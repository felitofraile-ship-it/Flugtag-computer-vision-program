"""Offline, keyframe-assisted, bidirectional neural tracking of a whole video.

The role labels and correction boxes in the input JSON are reviewed annotations.
They are not presented as automatic detections. No box is invented across an
explicitly occluded interval. Low-confidence tracking is left unboxed.
"""
from pathlib import Path
import argparse,csv,json,subprocess,time
import cv2
import numpy as np
from tracking import NeuralTracker
from engine import iou

COLORS={1:(150,235,80),2:(70,180,255),3:(250,190,80)}

def frame_ranges(segments,fps,total):
    result=[]
    for i,s in enumerate(segments):
        start=round(s['start']*fps);end=min(total-1,round(s['end']*fps))
        if i+1<len(segments):end=min(end,round(segments[i+1]['start']*fps)-1)
        result.append((start,end))
    return result

def trim_to_shots(data):
    ranges=frame_ranges(data['segments'],data['fps'],data['total_frames'])
    data['observations']=[r for r in data['observations'] if ranges[r['segment']][0]<=r['frame']<=ranges[r['segment']][1]]
    return data

def clipped(box,w=960,h=540):
    x,y,bw,bh=box;x=max(0,min(w-8,int(x)));y=max(0,min(h-8,int(y)))
    return [x,y,max(8,min(w-x,int(bw))),max(8,min(h-y,int(bh)))]

def propagate(frames,box,reverse=False):
    order=range(len(frames)-1,-1,-1) if reverse else range(len(frames))
    ids=list(order);tracker=NeuralTracker();tracker.init(frames[ids[0]],tuple(clipped(box)))
    result={ids[0]:(np.array(box,float),1.0)}
    for i in ids[1:]:
        ok,b=tracker.update(frames[i]);score=tracker.score or 0
        result[i]=(np.array(clipped(b),float),score if ok else 0)
    return result

def read_interval(cap,start,end):
    if isinstance(cap,list):return [cv2.imread(str(p)) for p in cap[start:end+1]]
    cap.set(cv2.CAP_PROP_POS_FRAMES,start);frames=[]
    for _ in range(start,end+1):
        ok,f=cap.read()
        if not ok:break
        frames.append(cv2.resize(f,(960,540),interpolation=cv2.INTER_AREA))
    return frames

def align_water_wing(frame,box):
    """Clip-specific appearance cue for this blue/pink wing, not a class model."""
    x,y,w,h=box;left=max(0,x-30);top=max(0,y-80);right=min(960,x+w+30);bottom=min(540,y+h+80)
    hsv=cv2.cvtColor(frame[top:bottom,left:right],cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,np.array([85,65,45]),np.array([179,255,255]))
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    n,_,stats,_=cv2.connectedComponentsWithStats(mask)
    candidates=[v for v in stats[1:] if v[4]>=45 and v[2]>=25 and v[2]/max(1,v[3])>=2.2]
    if not candidates:return box
    candidates=sorted(candidates,key=lambda v:v[4],reverse=True)
    base=candidates[0];cy=base[1]+base[3]/2
    chosen=[v for v in candidates if abs(v[1]+v[3]/2-cy)<24][:3]
    y0=min(v[1] for v in chosen);y1=max(v[1]+v[3] for v in chosen)
    return clipped([x,top+(y0+y1)/2-h/2,w,h])

def analyze(project,output,update_ids=None):
    cap=cv2.VideoCapture(project['video']);fps=cap.get(cv2.CAP_PROP_FPS);total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cache=Path('work/flightlab_frames');cache.mkdir(parents=True,exist_ok=True)
    source=Path(project['video']);signature=[str(source.resolve()),source.stat().st_size,source.stat().st_mtime_ns]
    manifest=cache/'source.json'
    files=[cache/f'{i:05}.jpg' for i in range(total)]
    if not manifest.exists() or json.loads(manifest.read_text())!=signature or not all(p.exists() for p in files):
        print('Caching source frames once for efficient bidirectional tracking...',flush=True)
        for path in files:
            ok,f=cap.read()
            if not ok:raise ValueError('Video ended while caching frames')
            cv2.imwrite(str(path),cv2.resize(f,(960,540),interpolation=cv2.INTER_AREA),[cv2.IMWRITE_JPEG_QUALITY,94])
        manifest.write_text(json.dumps(signature))
    cap.release();cap=files
    rows=[];sf=960/project['coordinate_size'][0]
    if update_ids:
        prior=json.loads((output/'review_tracks.json').read_text(encoding='utf-8'))
        if prior['video']!=project['video'] or prior['total_frames']!=total:raise ValueError('Cannot reuse tracks from another video')
        rows=[r for r in prior['observations'] if r['id'] not in update_ids]
    for segment_index,segment in enumerate(project['segments']):
        print('Tracking:',segment['name'],flush=True)
        for track in segment['tracks']:
            if update_ids and track['id'] not in update_ids:continue
            points=[(min(total-1,round(k['time']*fps)),np.array(k['box'],float)*sf if k['box'] else None) for k in track['keyframes']]
            for j,(start,box) in enumerate(points):
                if box is None:continue
                next_idx,next_box=points[j+1] if j+1<len(points) else (min(total-1,round(segment['end']*fps)),None)
                if next_idx<start:raise ValueError('Keyframes must be in increasing order')
                # The next manual keyframe owns its frame. An occlusion keyframe
                # explicitly terminates the preceding visible track.
                end=next_idx if j+1==len(points) else next_idx-1
                frames=read_interval(cap,start,next_idx)
                if not frames:continue
                forward=propagate(frames,box)
                backward=propagate(frames,next_box,True) if next_box is not None else None
                for offset in range(min(end-start+1,len(frames))):
                    b,score=forward[offset];method='NanoTrack';flag=False
                    if offset==0:b=box;score=1.;method='reviewed keyframe'
                    elif backward:
                        bb,bs=backward[offset];a=offset/max(1,len(frames)-1)
                        agreement=iou(b,bb)
                        if score>.5 and bs>.5 and agreement>.25:
                            b=(1-a)*b+a*bb;score=(1-a)*score+a*bs;method='bidirectional NanoTrack'
                        else:
                            # Prefer the nearer reliable anchor, never blend two
                            # unrelated tracked regions into a fake middle box.
                            if bs*(.25+a)>score*(1.25-a):b,score=bb,bs
                            flag=True;method='directional NanoTrack - review'
                        # Wide thin wings cause neural box-shape drift. Reviewed
                        # endpoint shapes constrain size; neural motion refines
                        # the center locally instead of growing onto bystanders.
                        expected=(1-a)*box+a*next_box
                        shift=b[:2]+b[2:]/2-(expected[:2]+expected[2:]/2)
                        # Compact subjects need freedom to follow camera pans;
                        # wide wings need tighter shape/center constraints.
                        if track['id']==2:
                            limit=np.maximum([12,12],expected[2:]*1.5)
                            shift=np.clip(shift,-limit,limit)
                        else:
                            limit=np.maximum([6,4],expected[2:]*.08)
                            shift=np.clip(shift,-limit,limit)*.5
                        b=np.r_[expected[:2]+shift,expected[2:]]
                        method='keyframe-constrained bidirectional NanoTrack'
                    elif offset>0:
                        # With no next visible anchor, avoid unchecked aspect
                        # ratio growth during a short visible tail.
                        center=b[:2]+b[2:]/2
                        size=np.clip(b[2:],box[2:]*.8,box[2:]*1.2)
                        b=np.r_[center-size/2,size]
                    status='visible' if score>=.5 else 'lost'
                    if status=='lost' and backward is not None:
                        b=expected;status='guided estimate';method='reviewed-keyframe interpolation';flag=True
                    rows.append(dict(frame=start+offset,time=(start+offset)/fps,segment=segment_index,
                        view=segment['view'],id=track['id'],label=track['label'],box=clipped(b) if status in ('visible','guided estimate') else None,
                        score=float(score),method=method,review_flag=flag,status=status))
    for r in rows:
        if r['view']=='water' and r['id']==1 and r['box'] is not None:
            r['box']=align_water_wing(cv2.imread(str(files[r['frame']])),r['box'])
            if 'wing-color alignment' not in r['method']:r['method']+=' + wing-color alignment'
    data=dict(video=project['video'],fps=fps,total_frames=total,coordinate_size=[960,540],
              method=project['identification'],segments=project['segments'],observations=rows)
    trim_to_shots(data);rows=data['observations']
    (output/'review_tracks.json').write_text(json.dumps(data,separators=(',',':')),encoding='utf-8')
    with (output/'review_tracks.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f);writer.writerow(['frame','video_seconds','shot','id','label','x','y','width','height','tracking_score','method','needs_review','status'])
        for r in sorted(rows,key=lambda r:(r['frame'],r['id'])):
            writer.writerow([r['frame'],r['time'],project['segments'][r['segment']]['name'],r['id'],r['label'],*(r['box'] or ['']*4),r['score'],r['method'],r['review_flag'],r['status']])
    return data

def render(data,output,work):
    fps=data['fps'];byframe={}
    for r in data['observations']:byframe.setdefault(r['frame'],[]).append(r)
    cap=cv2.VideoCapture(data['video']);raw=work/'tracking_silent.mp4'
    writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1280,784))
    histories={};previous_segment=None;index=0;ranges=frame_ranges(data['segments'],fps,data['total_frames'])
    while True:
        ok,f=cap.read()
        if not ok:break
        f=cv2.resize(f,(1280,720),interpolation=cv2.INTER_AREA);t=index/fps
        si=next((i for i,(start,end) in enumerate(ranges) if start<=index<=end),None)
        segment=data['segments'][si] if si is not None else None
        if si!=previous_segment:histories={};previous_segment=si
        for r in byframe.get(index,[]):
            if r['segment']!=si:continue
            if r['box'] is None:histories.pop(r['id'],None);continue
            x,y,w,h=[round(v*4/3) for v in r['box']];color=COLORS.get(r['id'],(210,190,100))
            history=histories.setdefault(r['id'],[])
            if history and history[-1][0]!=index-1:history.clear()
            history.append((index,(x+w//2,y+h//2)));history[:]=history[-30:]
            if segment and segment['view']=='side' and r['id'] in (1,2) and len(history)>1:
                path=np.int32([v[1] for v in history]);path=cv2.approxPolyDP(path,1.5,False)
                cv2.polylines(f,[path],False,color,2,cv2.LINE_AA)
            cv2.rectangle(f,(x,y),(x+w,y+h),color,2,cv2.LINE_AA)
            label=f'{r["label"].upper()}  {r["id"]:02}'
            tw,th=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.55,1)[0]
            ly=max(22,y-7);lx=min(x,1280-tw-10)
            cv2.rectangle(f,(lx,ly-th-6),(lx+tw+8,ly+5),(20,30,35),-1)
            cv2.putText(f,label,(lx+4,ly),cv2.FONT_HERSHEY_SIMPLEX,.55,color,1,cv2.LINE_AA)
        canvas=np.full((784,1280,3),(25,20,15),np.uint8);canvas[:720]=f
        title=segment['name'].upper() if segment else 'BROADCAST TRANSITION / TRACKS HIDDEN'
        cv2.putText(canvas,title,(20,746),cv2.FONT_HERSHEY_SIMPLEX,.62,(230,240,240),1,cv2.LINE_AA)
        note='Keyframe-assisted NanoTrack  |  video time %.2fs  |  measurements in separate graphs'%t
        cv2.putText(canvas,note,(20,771),cv2.FONT_HERSHEY_SIMPLEX,.45,(170,185,195),1,cv2.LINE_AA)
        if segment and not byframe.get(index):
            cv2.putText(canvas,'No reviewed subject box on this frame',(810,746),cv2.FONT_HERSHEY_SIMPLEX,.4,(180,190,205),1,cv2.LINE_AA)
        writer.write(canvas);index+=1
    writer.release();cap.release()
    import imageio_ffmpeg
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    cmd=[ffmpeg,'-y','-i',str(raw),'-i',data['video'],'-map','0:v:0','-map','1:a:0?',
         '-c:v','libx264','-threads','4','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-c:a','aac','-b:a','160k',
         '-movflags','+faststart','-shortest',str(output/'Flugtag_tracking_full.mp4')]
    subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    for name,start,end in [('Side_view',8.03,13.5),('Overhead_view',35,42.55),('Front_view',43.45,54.14)]:
        subprocess.run([ffmpeg,'-y','-ss',str(start),'-i',str(output/'Flugtag_tracking_full.mp4'),'-t',str(end-start),
            '-c:v','libx264','-threads','4','-preset','veryfast','-crf','20','-c:a','aac','-movflags','+faststart',str(output/f'{name}.mp4')],
            check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)

def graphs(data,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.style.use('dark_background')
    fig,axes=plt.subplots(3,2,figsize=(15,10),facecolor='#101a26')
    for row,view in enumerate(['side','top','front']):
        for ident,label,color in [(1,'Aircraft','#96eb50'),(2,'Pilot','#ffb446')]:
            points=[r for r in data['observations'] if r['view']==view and r['id']==ident]
            if not points:continue
            points=sorted(points,key=lambda r:r['frame']);xs=[];ys=[];ts=[]
            last=-2
            for r in points:
                if r['frame']!=last+1:ts.append(np.nan);xs.append(np.nan);ys.append(np.nan)
                ts.append(r['time']);b=r['box'];xs.append(b[0]+b[2]/2 if b else np.nan);ys.append(540-b[1]-b[3]/2 if b else np.nan);last=r['frame']
            axes[row,0].plot(ts,xs,color=color,label=label,lw=1.6)
            axes[row,1].plot(ts,ys,color=color,label=label,lw=1.6)
        for col,axis in enumerate(axes[row]):
            axis.set_facecolor('#162434');axis.grid(alpha=.12);axis.set_xlabel('Video time (s)')
            axis.set_ylabel('Image X (px)' if col==0 else 'Image height from bottom (px)')
            axis.set_title(view.title()+' view / '+('horizontal position' if col==0 else 'vertical position'),loc='left')
            axis.legend(loc='best',frameon=False)
    fig.suptitle('FlightLab | Object motion, separate from the video',fontsize=19,x=.06,ha='left')
    fig.text(.06,.025,'Image coordinates at 960 × 540. Camera motion is included. Gaps mean no reliable visible observation. Replays use video time, not verified physical time.',fontsize=10,color='#acbace')
    fig.tight_layout(rect=[.02,.055,.99,.95]);fig.savefig(output/'motion_graphs.png',dpi=160);fig.savefig(output/'motion_graphs.pdf');plt.close(fig)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',default=str(Path(__file__).with_name('flugtag_review_keyframes.json')))
    parser.add_argument('--output',default=str(Path(__file__).with_name('reviewed_flight')));parser.add_argument('--work',default='work/review_render')
    parser.add_argument('--render-only',action='store_true');parser.add_argument('--update-objects',type=int,nargs='+');args=parser.parse_args()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True);work=Path(args.work);work.mkdir(parents=True,exist_ok=True)
    if args.render_only:data=json.loads((output/'review_tracks.json').read_text(encoding='utf-8'))
    else:data=analyze(json.loads(Path(args.project).read_text(encoding='utf-8')),output,args.update_objects)
    print('Rendering complete video and separate graphs...',flush=True);render(data,output,work);graphs(data,output)
    print('Completed',len(data['observations']),'object observations',flush=True)

if __name__=='__main__':main()
