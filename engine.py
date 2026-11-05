"""FlightLab: local video tracking and explicitly planar measurements."""
from dataclasses import dataclass, field, asdict
from pathlib import Path
import csv
import json
import math
import urllib.request
import cv2
import numpy as np
from tracking import make_tracker

CLASSES = ('background aeroplane bicycle bird boat bottle bus car cat chair cow '
           'diningtable dog horse motorbike person pottedplant sheep sofa train tvmonitor').split()
COLORS = [(90, 230, 190), (70, 180, 255), (240, 150, 170), (250, 210, 80), (160, 120, 250)]

def transform(point, matrix):
    p = matrix @ np.array([*point, 1.0])
    return p[:2] / p[2]

def iou(a, b):
    x, y = max(a[0], b[0]), max(a[1], b[1])
    r, d = min(a[0]+a[2], b[0]+b[2]), min(a[1]+a[3], b[1]+b[3])
    inter = max(0, r-x)*max(0, d-y)
    return inter / max(1, a[2]*a[3]+b[2]*b[3]-inter)

@dataclass
class Calibration:
    units: str = 'ft'
    scale: float | None = None
    origin: list | None = None
    up: list = field(default_factory=lambda: [0.0, -1.0])
    reference: list | None = None
    known_length: float | None = None
    source: str = ''

    def set_reference(self, p1, p2, length, vertical=False, source=''):
        vec = np.asarray(p2, float)-p1
        distance = np.linalg.norm(vec)
        if not math.isfinite(length) or length <= 0 or distance < 8:
            raise ValueError('Use a positive length and a reference at least 8 pixels long.')
        self.scale = float(length/distance)
        self.reference = [list(p1), list(p2)]
        self.known_length = length
        self.source = source
        if vertical:
            self.up = (vec/distance).tolist()
            self.origin = list(p1)

    def position(self, point):
        up = np.asarray(self.up)
        right = np.array([-up[1], up[0]])
        p = np.asarray(point)-np.asarray(self.origin if self.origin is not None else [0, 0])
        return np.array([p@right, p@up])*(self.scale or 1)

@dataclass
class Sample:
    frame: int
    video_time: float
    elapsed: float
    image_x: float
    image_y: float
    ref_x: float | None
    ref_y: float | None
    status: str

@dataclass
class Track:
    id: int
    label: str
    box: tuple
    tracker: object
    source: str = 'manual'
    confidence: float | None = None
    status: str = 'tracking'
    samples: list = field(default_factory=list)
    tracking_score: float | None = None
    backend: str = 'CSRT'

class Detector:
    """VOC MobileNet-SSD. Class predictions are suggestions, not identities."""
    def __init__(self, folder):
        folder = Path(folder)
        self.net = cv2.dnn.readNetFromCaffe(str(folder/'deploy.prototxt'), str(folder/'mobilenet_iter_73000.caffemodel'))

    @staticmethod
    def download(folder):
        folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
        base = 'https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/'
        for name in ['deploy.prototxt', 'mobilenet_iter_73000.caffemodel', 'LICENSE']:
            target = folder/name
            if target.exists():
                continue
            temp = target.with_suffix(target.suffix+'.part')
            with urllib.request.urlopen(base+name, timeout=90) as response, temp.open('wb') as f:
                while chunk := response.read(1024*1024):
                    f.write(chunk)
            temp.replace(target)

    def detect(self, frame, threshold=.55):
        h, w = frame.shape[:2]
        self.net.setInput(cv2.dnn.blobFromImage(frame, 1/127.5, (300, 300), 127.5, swapRB=False))
        output = self.net.forward()[0, 0]
        boxes, labels, scores = [], [], []
        for det in output:
            score = float(det[2]); cls = int(det[1])
            if score < threshold or not 0 < cls < len(CLASSES):
                continue
            x1, y1, x2, y2 = det[3:7]*[w, h, w, h]
            x1, x2 = np.clip([x1, x2], 0, w-1); y1, y2 = np.clip([y1, y2], 0, h-1)
            box = tuple(map(int, [x1, y1, x2-x1, y2-y1]))
            if min(box[2:]) < 12:
                continue
            boxes.append(box); labels.append(CLASSES[cls]); scores.append(score)
        keep = []
        for label in set(labels):
            indices = [i for i, x in enumerate(labels) if x == label]
            selected = cv2.dnn.NMSBoxes([boxes[i] for i in indices], [scores[i] for i in indices], threshold, .4)
            keep.extend(indices[int(i)] for i in np.asarray(selected).reshape(-1))
        return [(labels[i], scores[i], boxes[i]) for i in keep]

class Motion:
    def __init__(self):
        self.matrix = np.eye(3)  # current frame -> segment reference frame
        self.valid = True
        self.quality = 'reference frame'

    def update(self, previous, current, boxes, region, enabled):
        if not enabled:
            self.quality = 'uncorrected camera motion'
            return False
        if not self.valid:
            return False
        a = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(a)
        if region:
            x,y,w,h = region; mask[y:y+h,x:x+w] = 255
        else:
            mask[:] = 255
        for x,y,w,h in boxes:
            cv2.rectangle(mask, (max(0,int(x)-12), max(0,int(y)-12)), (int(x+w)+12,int(y+h)+12), 0, -1)
        points = cv2.goodFeaturesToTrack(a, 350, .015, 12, mask=mask)
        if points is None or len(points) < 16:
            return self.fail('too few background features')
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(a, b, points, None)
        if nxt is None:
            return self.fail('background tracking failed')
        back, status2, _ = cv2.calcOpticalFlowPyrLK(b, a, nxt, None)
        if back is None:
            return self.fail('background tracking failed')
        good = (status.ravel()==1)&(status2.ravel()==1)&(np.linalg.norm(points-back,axis=2).ravel()<1.5)
        if good.sum() < 14:
            return self.fail('insufficient background matches')
        affine, inliers = cv2.estimateAffinePartial2D(points[good], nxt[good], method=cv2.RANSAC, ransacReprojThreshold=2.5)
        if affine is None or inliers.sum() < 12 or inliers.mean() < .55:
            return self.fail('inconsistent camera motion')
        scale = np.linalg.norm(affine[:,0])
        if not .92 < scale < 1.08 or np.linalg.norm(affine[:,2]) > max(a.shape)*.12:
            return self.fail('camera jump / possible cut')
        self.matrix = self.matrix @ np.linalg.inv(np.vstack([affine,[0,0,1]]))
        self.quality = f'{int(inliers.sum())} background matches; approximate'
        return True

    def fail(self, reason):
        self.valid = False
        self.quality = reason+'; new segment required'
        return False

def scene_cut(a, b):
    # Heuristic only: a cut should always be reviewed by the operator.
    a = cv2.resize(a, (160,90)); b = cv2.resize(b,(160,90))
    diff = np.mean(np.abs(a.astype(float)-b.astype(float)))/255
    ha = cv2.calcHist([cv2.cvtColor(a,cv2.COLOR_BGR2HSV)],[0,1],None,[24,24],[0,180,0,256])
    hb = cv2.calcHist([cv2.cvtColor(b,cv2.COLOR_BGR2HSV)],[0,1],None,[24,24],[0,180,0,256])
    cv2.normalize(ha,ha); cv2.normalize(hb,hb)
    return diff > .17 or (diff > .10 and cv2.compareHist(ha,hb,cv2.HISTCMP_BHATTACHARYYA) > .40)

class Engine:
    def __init__(self, fps=30):
        self.fps = fps
        self.time_factor = 1.0  # physical seconds per video second
        self.calibration = Calibration()
        self.motion = Motion()
        self.compensate = False
        self.region = None
        self.background = None
        self.tracks = []
        self.frame = None
        self.index = 0
        self.start_index = 0
        self.segment = 1
        self.cut = False
        self.measurements = []
        self.view_mode = 'Tracking'
        self.show_guides = False
        self.redetect = False
        self.detector = None
        self.corrections = []

    def reset(self, frame, index):
        self.frame = frame.copy(); self.index = index; self.start_index = index
        self.tracks = []; self.calibration = Calibration(); self.motion = Motion(); self.cut = False
        self.measurements = []
        self.corrections = []

    def add(self, box, label, source='manual', confidence=None):
        x,y,w,h = map(int,box)
        fh,fw = self.frame.shape[:2]
        if x<0 or y<0 or w<8 or h<8 or x+w>fw or y+h>fh:
            raise ValueError('Draw a box at least 8 pixels wide and tall, inside the video.')
        tracker = make_tracker((x,y,w,h)); tracker.init(self.frame,(x,y,w,h))
        t = Track(len(self.tracks)+1,label,(x,y,w,h),tracker,source,confidence)
        t.backend=getattr(tracker,'name','CSRT')
        self.tracks.append(t); self.sample(t)
        self.corrections.append(dict(frame=self.index,id=t.id,label=t.label,box=list(t.box),source=source))
        return t

    def reanchor(self,track_id,box,source='manual correction'):
        t=next(t for t in self.tracks if t.id==track_id)
        x,y,w,h=map(int,box);fh,fw=self.frame.shape[:2]
        if min(w,h)<8 or x<0 or y<0 or x+w>fw or y+h>fh:raise ValueError('Invalid correction box.')
        t.tracker=make_tracker((x,y,w,h));t.tracker.init(self.frame,(x,y,w,h));t.box=(x,y,w,h);t.status='tracking';t.tracking_score=None
        t.backend=getattr(t.tracker,'name','CSRT')
        if t.samples and t.samples[-1].frame==self.index:t.samples.pop()
        self.sample(t)
        self.corrections.append(dict(frame=self.index,id=t.id,label=t.label,box=list(t.box),source=source))
        return t

    def sample(self,t):
        x,y,w,h = t.box; center = (x+w/2,y+h/2)
        valid = t.status=='tracking' and (not self.compensate or self.motion.valid)
        ref = transform(center,self.motion.matrix) if valid else [None,None]
        t.samples.append(Sample(self.index,self.index/self.fps,(self.index-self.start_index)/self.fps*self.time_factor,
                                *center,*ref,t.status))

    def process(self, frame, index):
        if index != self.index+1:
            raise ValueError('Process consecutive frames. Seeking requires a new segment.')
        if scene_cut(self.frame,frame):
            self.cut = True
            for t in self.tracks: t.status='cut / stopped'
            self.frame = frame; self.index = index
            self.motion.fail('possible scene cut')
            return
        self.motion.update(self.frame,frame,[t.box for t in self.tracks],self.background or self.region,self.compensate)
        for t in self.tracks:
            if t.status == 'stopped':
                continue
            if t.status != 'tracking':
                self.index = index; self.sample(t); continue
            ok, box = t.tracker.update(frame)
            t.tracking_score=getattr(t.tracker,'score',None)
            if ok and min(box[2:]) > 0:
                t.box = tuple(map(int,box))
            else:
                t.status = 'lost / reselect'
            self.index = index; self.sample(t)
        self.frame = frame; self.index = index
        if self.redetect and self.detector is not None and index%15==0:
            if self.region:
                rx,ry,rw,rh=self.region;crop=frame[ry:ry+rh,rx:rx+rw]
            else:rx=ry=0;crop=frame
            detections=[(label,score,(b[0]+rx,b[1]+ry,b[2],b[3])) for label,score,b in self.detector.detect(crop)]
            used=set()
            for t in self.tracks:
                if t.label.lower() not in ('person','pilot') or t.status=='stopped':continue
                candidates=[(iou(t.box,b),i,b,score) for i,(label,score,b) in enumerate(detections) if label=='person' and i not in used]
                if candidates:
                    overlap,i,b,score=max(candidates,key=lambda v:v[0])
                    if overlap>=.35:
                        self.reanchor(t.id,b,source='detector local association');t.confidence=score;used.add(i)

    def metrics(self,t):
        if self.cut:
            return None
        valid = [s for s in t.samples if s.ref_x is not None]
        if not valid or t.samples[-1].ref_x is None:
            return None
        c = self.calibration
        p = [c.position([s.ref_x,s.ref_y]) for s in valid]
        delta = p[-1]-p[0]
        # Do not bridge missing samples when accumulating trajectory length.
        path = sum(float(np.linalg.norm(p[i]-p[i-1])) for i in range(1,len(p)) if valid[i].frame==valid[i-1].frame+1)
        speed = None
        if len(valid)>1:
            j=max(0,len(valid)-7); dt=valid[-1].elapsed-valid[j].elapsed
            if dt>0 and valid[-1].frame-valid[j].frame == len(valid)-1-j:
                speed=float(np.linalg.norm(p[-1]-p[j])/dt)
        return dict(dx=float(delta[0]),dy=float(delta[1]),displacement=float(np.linalg.norm(delta)),path=path,
                    height=float(p[-1][1]) if c.origin is not None and c.scale is not None else None,
                    speed=speed,unit=c.units if c.scale is not None else 'px')

    def overlay(self):
        out = self.frame.copy()
        inv = np.linalg.inv(self.motion.matrix)
        for t in self.tracks:
            color=COLORS[(t.id-1)%len(COLORS)]
            # Render old reference-frame points in the current camera view.
            chain=[]
            for s in (t.samples[-90:] if self.view_mode=='Side view' else []):
                if s.ref_x is None:
                    if len(chain)>1: cv2.polylines(out,[np.int32(chain)],False,color,2,cv2.LINE_AA)
                    chain=[]; continue
                chain.append(transform([s.ref_x,s.ref_y],inv))
            if len(chain)>1: cv2.polylines(out,[np.int32(chain)],False,color,2,cv2.LINE_AA)
            x,y,w,h=t.box
            if t.status not in ('tracking','stopped'):
                continue
            cv2.rectangle(out,(x,y),(x+w,y+h),color,2)
            cv2.circle(out,(int(x+w/2),int(y+h/2)),4,color,-1)
            cv2.putText(out,f'{t.id}: {t.label}',(max(0,x),max(18,y-8)),cv2.FONT_HERSHEY_SIMPLEX,.48,color,1,cv2.LINE_AA)
        if self.region and self.show_guides:
            x,y,w,h=self.region; cv2.rectangle(out,(x,y),(x+w,y+h),(180,180,180),1)
        if self.background and self.show_guides:
            x,y,w,h=self.background; cv2.rectangle(out,(x,y),(x+w,y+h),(255,180,60),2)
        c=self.calibration
        if self.show_guides and c.reference and (not self.compensate or self.motion.valid):
            a,b=[tuple(np.int32(transform(p,inv))) for p in c.reference]
            cv2.line(out,a,b,(80,220,255),2)
            cv2.putText(out,f'{c.known_length:g} {c.units}',a,cv2.FONT_HERSHEY_SIMPLEX,.6,(80,220,255),2)
        if self.show_guides and c.origin is not None and (not self.compensate or self.motion.valid):
            right=np.array([-c.up[1],c.up[0]])
            a,b=[tuple(np.int32(transform(np.asarray(c.origin)+right*k,inv))) for k in [-3000,3000]]
            cv2.line(out,a,b,(220,180,80),1)
        return out

    def export(self, folder, video, reported_ft=42):
        folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
        fields=['segment','id','label','source','detection_confidence','frame','video_time','elapsed','image_x','image_y',
                'ref_x','ref_y','status','planar_x','planar_y','unit','camera_mode','camera_chain_valid']
        with (folder/'tracks.csv').open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for t in self.tracks:
                for s in t.samples:
                    pos=self.calibration.position([s.ref_x,s.ref_y]) if s.ref_x is not None else [None,None]
                    writer.writerow(dict(segment=self.segment,id=t.id,label=t.label,source=t.source,detection_confidence=t.confidence,
                        **asdict(s),planar_x=pos[0],planar_y=pos[1],unit=self.calibration.units if self.calibration.scale else 'px',
                        camera_mode='approximate affine correction' if self.compensate else 'uncorrected',
                        camera_chain_valid=(s.ref_x is not None) if self.compensate else 'not applicable'))
        data=dict(video=str(video),fps=self.fps,segment=self.segment,start_frame=self.start_index,end_frame=self.index,
                  processing_size=list(self.frame.shape[1::-1]),physical_seconds_per_video_second=self.time_factor,
                  calibration=asdict(self.calibration),camera_correction=self.compensate,camera_status=self.motion.quality,
                  analysis_region=self.region,background_region=self.background,reported_pilot_distance_ft=reported_ft,
                  point_measurements=self.measurements,
                  correction_events=self.corrections,
                  known_references={'wingspan_ft_approx':20,'platform_height_ft':30},
                  caveats=['Single-view planar estimates, not reconstructed 3D or official flight distance.',
                           'Box centers are tracking anchors, not physical centers of mass.',
                           'Camera correction is approximate; moving water/overlays/parallax can fool it.',
                           'Elapsed physical time assumes the supplied replay timing factor.',
                           'Path length includes tracking jitter. Class labels are model suggestions.'],
                  tracks=[dict(id=t.id,label=t.label,source=t.source,status=t.status,metrics=self.metrics(t)) for t in self.tracks])
        (folder/'analysis.json').write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')
        cv2.imwrite(str(folder/'overlay.png'),self.overlay())
        return folder
