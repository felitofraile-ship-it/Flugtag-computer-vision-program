"""Run with python app.py [video.mp4]. All video analysis stays local."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import argparse
import json
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import cv2
import numpy as np
from PIL import Image, ImageTk
from engine import Engine, Detector, COLORS, transform, iou
from tracking import YoloDetector
import os

ROOT=Path(__file__).resolve().parent
BG='#0c1420'; PANEL='#142130'; FG='#e4edf7'; MUTED='#91a6bc'; ACCENT='#63e6bd'

class App:
    def __init__(self, root, video=None):
        self.root=root; root.title('FlightLab 2 | Object Tracking Studio'); root.geometry('1400x900'); root.minsize(1100,760)
        root.configure(bg=BG)
        self.pool=ThreadPoolExecutor(max_workers=1); self.future=None; self.pending=None
        self.cap=None; self.engine=Engine(); self.path=''; self.total=0; self.playing=False
        self.mode=None; self.points=[]; self.drag_start=None; self.writer=None; self.detector=None
        self.view_scale=1; self.offset=(0,0); self.photo=None; self.pending_frame=None; self.last_process=0; self.processing_fps=None
        self.status=tk.StringVar(value='Open a video to begin. All analysis runs on this computer.')
        self.info=tk.StringVar(value='No video loaded'); self.time_text=tk.StringVar(value='00:00.00')
        self.compensate=tk.BooleanVar(value=False); self.factor=tk.StringVar(value='1.0'); self.speed=tk.StringVar(value='1x')
        self.reference=tk.StringVar(value='Wingspan · 20 ft'); self.distance=tk.StringVar(value='20'); self.units=tk.StringVar(value='ft')
        self.label=tk.StringVar(value='Aircraft'); self.seek_text=tk.StringVar(value='9.0')
        self.metric_text=tk.StringVar(value='Select an object to inspect its measurements.')
        self.camera_text=tk.StringVar(value='Camera: uncorrected'); self.cal_text=tk.StringVar(value='PIXELS ONLY · no scale reference')
        self.view_mode=tk.StringVar(value='Tracking');self.guides=tk.BooleanVar(value=False);self.redetect=tk.BooleanVar(value=False)
        self.graph_window=None;self.graph_metric=tk.StringVar(value='Image vertical position');self.plot=None
        self.build(); root.protocol('WM_DELETE_WINDOW',self.close)
        root.bind('<space>',lambda e:self.toggle() if not isinstance(e.widget,(tk.Entry,ttk.Entry,ttk.Combobox)) else None)
        root.bind('<Escape>',lambda e:self.cancel_selection())
        root.after(25,self.tick)
        if video: self.open(video)

    def build(self):
        style=ttk.Style(); style.theme_use('clam')
        style.configure('.',background=PANEL,foreground=FG,font=('Segoe UI',10))
        style.configure('TButton',padding=(10,7),background='#23374b',borderwidth=0)
        style.map('TButton',background=[('active','#34526a')])
        style.configure('Accent.TButton',background='#226d60')
        style.configure('TEntry',fieldbackground='#0c1926',foreground=FG)
        style.configure('TCombobox',fieldbackground='#0c1926',foreground=FG,arrowcolor=FG)
        style.map('TCombobox',fieldbackground=[('readonly','#0c1926')],foreground=[('readonly',FG)])
        style.configure('Treeview',background='#0e1a27',fieldbackground='#0e1a27',foreground=FG,rowheight=30,borderwidth=0)
        style.configure('Treeview.Heading',background='#24374a',foreground=FG)
        head=tk.Frame(self.root,bg=BG); head.pack(fill='x',padx=20,pady=(16,10))
        tk.Label(head,text='FLIGHTLAB',font=('Segoe UI',23,'bold'),fg=ACCENT,bg=BG).pack(side='left')
        tk.Label(head,text='  /  OBJECT TRACKING STUDIO',font=('Segoe UI',10),fg=MUTED,bg=BG).pack(side='left',pady=(10,0))
        ttk.Button(head,text='Open video',command=lambda:self.open()).pack(side='right')
        tk.Label(self.root,textvariable=self.info,bg=BG,fg=MUTED,anchor='w').pack(fill='x',padx=22)
        body=tk.Frame(self.root,bg=BG); body.pack(fill='both',expand=True,padx=20,pady=12)
        left=tk.Frame(body,bg=BG);left.pack(side='left',fill='both',expand=True,padx=(0,14))
        sidebar=tk.Frame(body,bg=PANEL,width=370);sidebar.pack(side='right',fill='y');sidebar.pack_propagate(False)
        scroller=tk.Canvas(sidebar,bg=PANEL,highlightthickness=0)
        scrollbar=ttk.Scrollbar(sidebar,orient='vertical',command=scroller.yview);scrollbar.pack(side='right',fill='y')
        scroller.pack(side='left',fill='both',expand=True);scroller.configure(yscrollcommand=scrollbar.set)
        right=tk.Frame(scroller,bg=PANEL)
        content=scroller.create_window((0,0),window=right,anchor='nw')
        right.bind('<Configure>',lambda e:scroller.configure(scrollregion=scroller.bbox('all')))
        scroller.bind('<Configure>',lambda e:scroller.itemconfigure(content,width=e.width))
        def wheel(event):
            if event.widget.winfo_rootx()>=sidebar.winfo_rootx():scroller.yview_scroll(-int(event.delta/120),'units')
        self.root.bind('<MouseWheel>',wheel,add='+')
        self.canvas=tk.Canvas(left,bg='#070c13',highlightthickness=1,highlightbackground='#23374b',cursor='crosshair')
        self.canvas.pack(fill='both',expand=True)
        self.canvas.bind('<ButtonPress-1>',self.mouse_down); self.canvas.bind('<B1-Motion>',self.mouse_move); self.canvas.bind('<ButtonRelease-1>',self.mouse_up)
        self.canvas.bind('<Configure>',lambda e:self.draw())
        toolbar=tk.Frame(left,bg=BG);toolbar.pack(fill='x',pady=10)
        self.play_button=ttk.Button(toolbar,text='Play',command=self.toggle);self.play_button.pack(side='left')
        ttk.Button(toolbar,text='Step +1',command=self.step).pack(side='left',padx=5)
        ttk.Combobox(toolbar,textvariable=self.speed,values=['0.25x','0.5x','1x','2x'],width=6,state='readonly').pack(side='left')
        tk.Label(toolbar,textvariable=self.time_text,bg=BG,fg=ACCENT,font=('Consolas',12)).pack(side='left',padx=12)
        ttk.Button(toolbar,text='New segment',command=self.new_segment).pack(side='right')
        seek=tk.Frame(left,bg=BG);seek.pack(fill='x')
        tk.Label(seek,text='Jump to video second',bg=BG,fg=MUTED).pack(side='left')
        ttk.Entry(seek,textvariable=self.seek_text,width=8).pack(side='left',padx=6)
        ttk.Button(seek,text='Go',command=self.seek).pack(side='left')
        tk.Label(seek,text='Jumping archives tracks and starts a fresh segment.',bg=BG,fg=MUTED,font=('Segoe UI',9)).pack(side='left',padx=10)
        self.tree=ttk.Treeview(left,columns=('name','status','distance'),show='headings',height=4,selectmode='browse')
        for col,title,width in [('name','TRACKED OBJECT',210),('status','STATUS',160),('distance','TRACKER RESPONSE',190)]:
            self.tree.heading(col,text=title);self.tree.column(col,width=width)
        self.tree.pack(fill='x',pady=(12,0));self.tree.bind('<<TreeviewSelect>>',lambda e:self.update_metrics())
        self.section(right,'01  /  OBJECTS')
        row=self.row(right);ttk.Entry(row,textvariable=self.label,width=15).pack(side='left',fill='x',expand=True)
        ttk.Button(row,text='Draw box',style='Accent.TButton',command=lambda:self.select('track')).pack(side='right',padx=(6,0))
        row=self.row(right);ttk.Button(row,text='Identify objects',command=self.identify).pack(side='left',fill='x',expand=True)
        ttk.Button(row,text='Stop selected',command=self.stop_track).pack(side='right',padx=(6,0))
        self.note(right,'Draw separate boxes for aircraft and pilot. Model labels are suggestions; verify every track.')
        row=self.row(right)
        ttk.Button(row,text='Correct selected box',command=lambda:self.select('correct')).pack(fill='x')
        self.section(right,'02  /  VIEW')
        views=ttk.Combobox(right,textvariable=self.view_mode,values=['Tracking','Side view','Top view'],state='readonly')
        views.pack(fill='x',padx=14,pady=4);views.bind('<<ComboboxSelected>>',lambda e:self.change_view())
        self.note(right,'Top view: object boxes only. Side view: boxes and short trajectory lines. Measurements stay in a separate window.')
        ttk.Checkbutton(right,text='Show selection / calibration guides',variable=self.guides,command=self.change_view).pack(anchor='w',padx=14,pady=5)
        ttk.Button(right,text='Measurements & graphs',command=self.open_graphs).pack(fill='x',padx=14,pady=8)
        self.section(right,'03  /  TRACKING OPTIONS')
        row=self.row(right);ttk.Button(row,text='Analysis region',command=lambda:self.select('region')).pack(side='left')
        ttk.Button(row,text='Static background',command=lambda:self.select('background')).pack(side='right')
        ttk.Checkbutton(right,text='Camera correction for measurements',variable=self.compensate,command=self.camera_changed).pack(anchor='w',padx=14,pady=4)
        ttk.Checkbutton(right,text='Refresh recognized people every 15 frames',variable=self.redetect,command=self.set_redetect).pack(anchor='w',padx=14,pady=4)
        tk.Label(right,textvariable=self.camera_text,bg=PANEL,fg=MUTED,wraplength=320,justify='left',font=('Segoe UI',9)).pack(fill='x',padx=14,pady=5)
        self.note(right,'NanoTrack follows people and compact objects; CSRT handles wide wings. Correct selected box preserves its ID. Lost objects stay unboxed until corrected or locally re-detected.')
        self.section(right,'04  /  REVIEWED FLUGTAG CLIP')
        ttk.Button(right,text='Open full tracked video',command=self.open_review).pack(fill='x',padx=14,pady=4)
        self.note(right,'The prepared review uses manually identified roles and correction keyframes, with neural tracking between them. Hidden pilot intervals are omitted.')
        exports=tk.Frame(self.root,bg=BG);exports.pack(fill='x',padx=20,pady=(0,10))
        ttk.Button(exports,text='Export CSV + report + image',command=self.export).pack(side='left')
        self.record_button=ttk.Button(exports,text='Record overlay MP4',command=self.record);self.record_button.pack(side='left',padx=8)
        tk.Label(exports,text='Aircraft / pilot / components • measurements in separate graphs',bg=BG,fg=MUTED).pack(side='right')
        tk.Label(self.root,textvariable=self.status,bg='#172738',fg=FG,anchor='w',padx=20,pady=8,wraplength=1300).pack(fill='x')

    def section(self,parent,text):
        tk.Label(parent,text=text,bg=PANEL,fg=MUTED,font=('Segoe UI',9,'bold'),anchor='w').pack(fill='x',padx=14,pady=(10,4))
    def row(self,parent):
        row=tk.Frame(parent,bg=PANEL);row.pack(fill='x',padx=14,pady=3);return row
    def note(self,parent,text):
        tk.Label(parent,text=text,bg=PANEL,fg=MUTED,wraplength=320,justify='left',anchor='w',font=('Segoe UI',9)).pack(fill='x',padx=14,pady=3)
    def ready(self):
        if self.future is not None:
            self.status.set('Finishing the current frame; try again in a moment.');return False
        return self.engine.frame is not None
    def pause(self):
        self.playing=False;self.play_button.configure(text='Play')
    def cancel_selection(self):
        self.mode=None;self.points=[];self.drag_start=None;self.canvas.delete('selection');self.status.set('Selection cancelled.')

    def open(self,path=None):
        if self.future: return
        self.pause()
        if not path: path=filedialog.askopenfilename(filetypes=[('Video','*.mp4 *.mov *.avi *.mkv *.webm'),('All files','*.*')])
        if not path:return
        cap=cv2.VideoCapture(str(path));ok,frame=cap.read()
        if not ok:
            cap.release();messagebox.showerror('Cannot open video','The file could not be decoded.');return
        self.archive();self.stop_recording()
        if self.cap:self.cap.release()
        self.cap=cap;self.path=str(path);self.total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps=cap.get(cv2.CAP_PROP_FPS)
        if fps<=0:fps=30;self.status.set('Missing frame-rate metadata: using 30 fps. Verify timing.')
        self.engine=Engine(fps);self.engine.reset(self.resize(frame),0);self.compensate.set(False)
        self.engine.view_mode=self.view_mode.get();self.engine.show_guides=self.guides.get()
        self.info.set(f'{Path(path).name}   /   {int(cap.get(3))} × {int(cap.get(4))}   /   {fps:.3f} fps   /   {self.total/fps:.2f} seconds')
        self.status.set('Pause at the start of one continuous shot. Draw boxes, set scale, then play. Broadcast replays need timing correction.')
        self.refresh()

    def load_setup(self,path):
        """Load a user-authored setup at one segment's initial frame."""
        data=json.loads(Path(path).read_text(encoding='utf-8'))
        self.seek_text.set(str(data['start_seconds']));self.seek()
        self.engine.region=tuple(data['analysis_region']) if data.get('analysis_region') else None
        self.engine.background=tuple(data['background_region']) if data.get('background_region') else None
        self.engine.compensate=bool(data.get('camera_correction',False));self.compensate.set(self.engine.compensate)
        self.engine.view_mode=data.get('view_mode','Tracking');self.view_mode.set(self.engine.view_mode)
        if data.get('calibration'):
            c=data['calibration'];self.engine.calibration.set_reference(c['point1'],c['point2'],c['length'],c.get('vertical',False),c.get('source',''))
            self.engine.calibration.units=c.get('units','ft')
        for item in data.get('objects',[]):self.engine.add(tuple(item['box']),item['label'])
        self.refresh();self.status.set(data.get('note','Setup loaded. Press Play to track this segment.'))

    @staticmethod
    def resize(frame):
        h,w=frame.shape[:2];s=min(1,960/w)
        return cv2.resize(frame,(round(w*s),round(h*s)),interpolation=cv2.INTER_AREA)
    def archive(self):
        if self.engine.tracks:
            out=ROOT/'runs'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            self.engine.export(out,self.path)
    def new_segment(self):
        self.pause()
        if not self.ready():return
        self.archive();self.stop_recording();self.cancel_selection()
        self.engine.segment+=1;self.engine.reset(self.engine.frame,self.engine.index)
        self.engine.compensate=self.compensate.get();self.refresh()
        self.status.set('New segment. Previous tracks archived in runs/. Re-mark scale and objects for this shot.')
    def seek(self):
        self.pause()
        if not self.ready():return
        try:
            sec=float(self.seek_text.get())
            if not np.isfinite(sec) or sec<0:raise ValueError()
        except ValueError:
            messagebox.showerror('Invalid time','Enter a non-negative video time in seconds.');return
        self.archive();self.stop_recording();self.cancel_selection()
        idx=min(self.total-1,round(sec*self.engine.fps));self.cap.set(cv2.CAP_PROP_POS_FRAMES,idx)
        ok,f=self.cap.read()
        if ok:
            self.engine.segment+=1;self.engine.reset(self.resize(f),idx);self.engine.compensate=self.compensate.get();self.refresh()
            self.status.set('Fresh segment at selected time. Set scale and select objects before playing.')
    def configure_timing(self):
        try:
            factor=float(self.factor.get())
            if not np.isfinite(factor) or not .01<=factor<=10:raise ValueError()
        except ValueError:
            messagebox.showerror('Timing','Enter a physical-seconds/video-second factor between 0.01 and 10.');return False
        if self.engine.tracks and factor!=self.engine.time_factor:
            messagebox.showinfo('Timing','Start a new segment before changing the timing factor.');return False
        self.engine.time_factor=factor;return True
    def toggle(self):
        if self.playing:self.pause();return
        if not self.ready() or not self.configure_timing():return
        if self.engine.cut:
            self.status.set('Possible camera cut: start a new segment before continuing.');return
        self.cancel_selection();self.playing=True;self.play_button.configure(text='Pause')
        self.status.set('Tracking live during playback. Check boxes for drift; pause and reselect if the subject changes shape or is occluded.')
    def step(self):
        self.pause()
        if self.ready() and self.configure_timing() and not self.engine.cut:self.submit_frame()
    def submit_frame(self):
        ok,f=self.cap.read()
        if not ok:
            self.pause();self.stop_recording();self.status.set('End of video. Export this segment or jump to another shot.');return
        f=self.resize(f);idx=self.engine.index+1
        self.last_process=time.perf_counter();self.future=self.pool.submit(self.engine.process,f,idx);self.pending='frame'
    def tick(self):
        if self.future and self.future.done():
            kind=self.pending;future=self.future;self.future=None
            try:
                result=future.result()
                if kind=='detect':
                    if len(result)==2 and isinstance(result[0],(Detector,YoloDetector)):self.detector,result=result
                    count=0
                    for label,confidence,box in result:
                        if len([t for t in self.engine.tracks if t.status=='tracking'])>=8:break
                        if any(iou(box,t.box)>.4 for t in self.engine.tracks if t.status=='tracking'):continue
                        self.engine.add(box,label,'YOLOX class suggestion',confidence);count+=1
                    self.engine.detector=self.detector;self.engine.redetect=self.redetect.get()
                    self.status.set(f'Added {count} suggested objects. A custom Flugtag craft may need a manually labeled box.')
                elif kind=='frame':
                    self.processing_fps=1/max(.001,time.perf_counter()-self.last_process)
                    if self.engine.cut:
                        self.pause();self.stop_recording();self.status.set('Possible camera cut detected. Tracking stopped. Start a new segment and recalibrate.')
                    elif self.writer:
                        self.writer.write(self.record_frame())
                    if self.engine.compensate and not self.engine.motion.valid and not self.engine.cut:
                        self.status.set('Object tracking continues. Camera reference lost: calibrated measurements are withheld; image-coordinate graphs remain available.')
                self.refresh()
            except Exception as exc:
                self.pause();self.stop_recording();messagebox.showerror('Analysis error',str(exc));self.status.set('Operation failed; video remains local. Try a new segment.')
        if self.playing and self.future is None:
            interval=1/self.engine.fps/float(self.speed.get().replace('x',''))
            if time.perf_counter()-self.last_process>=interval:self.submit_frame()
        self.root.after(15,self.tick)

    def select(self,mode):
        self.pause()
        if not self.ready():return
        if mode in ('region','background') and self.engine.tracks:
            self.status.set('Set regions before creating tracks, or start a new segment.');return
        if mode in ('track','correct') and not self.configure_timing():return
        if mode=='correct' and not self.tree.selection():
            self.status.set('Select a track in the table first.');return
        if mode in ('scale','vertical','origin','measure') and self.engine.compensate and not self.engine.motion.valid:
            self.status.set('Start a new segment before calibrating: camera reference was lost.');return
        self.mode=mode;self.points=[]
        text={'correct':'Drag a new box for the selected object. Its track ID will be preserved.',
              'track':'Drag a tight box around one object. Keep aircraft and pilot separate.',
              'region':'Drag the main video region to exclude chat, black bars, and picture-in-picture.',
              'background':'Drag a static background patch (platform or buildings). Avoid people, aircraft, water, and overlays.',
              'scale':'Click two endpoints of the known length at comparable depth to the object.',
              'vertical':'Click the BASE, then the TOP of the known vertical height.',
              'measure':'Click two points to measure planar length and vertical separation using the current scale.',
              'origin':'Click the zero-height level at the object’s depth. This defines a planar baseline.'}
        self.status.set(text[mode]+'  Esc cancels.')
    def image_point(self,e):
        x=(e.x-self.offset[0])/self.view_scale;y=(e.y-self.offset[1])/self.view_scale
        if self.engine.frame is None:return None
        h,w=self.engine.frame.shape[:2]
        return (int(np.clip(x,0,w-1)),int(np.clip(y,0,h-1)))
    def mouse_down(self,e):
        if not self.mode or self.future:return
        p=self.image_point(e)
        if self.mode in ('track','correct','region','background'):self.drag_start=p
        else:
            self.points.append(p)
            x,y=self.to_canvas(p);self.canvas.create_oval(x-4,y-4,x+4,y+4,fill=ACCENT,tags='selection')
            if self.mode=='origin' or len(self.points)==2:self.finish_points()
    def mouse_move(self,e):
        if not self.drag_start:return
        self.canvas.delete('selection');a=self.to_canvas(self.drag_start);b=self.to_canvas(self.image_point(e))
        self.canvas.create_rectangle(*a,*b,outline=ACCENT,width=2,tags='selection')
    def mouse_up(self,e):
        if self.drag_start is None:return
        a=self.drag_start;b=self.image_point(e);self.drag_start=None
        box=(min(a[0],b[0]),min(a[1],b[1]),abs(a[0]-b[0]),abs(a[1]-b[1]))
        try:
            if min(box[2:])<8:raise ValueError('Draw a larger box.')
            if self.mode=='track':
                if len([t for t in self.engine.tracks if t.status=='tracking'])>=8:raise ValueError('Limit: 8 active tracks. Stop a track first.')
                self.engine.add(box,self.label.get().strip() or 'Object')
            elif self.mode=='correct':self.engine.reanchor(int(self.tree.selection()[0]),box)
            elif self.mode=='region':self.engine.region=box
            elif self.mode=='background':self.engine.background=box
            self.status.set('Selection saved. Press Play to continue.')
        except Exception as exc:messagebox.showerror('Selection',str(exc))
        self.mode=None;self.canvas.delete('selection');self.refresh()
    def finish_points(self):
        try:
            p=[transform(p,self.engine.motion.matrix).tolist() for p in self.points]
            c=self.engine.calibration
            if self.mode=='measure':
                delta=c.position(p[1])-c.position(p[0]);unit=c.units if c.scale else 'px'
                measurement=dict(frame=self.engine.index,reference_points=p,length=float(np.linalg.norm(delta)),
                                 horizontal_delta=float(delta[0]),vertical_delta=float(delta[1]),unit=unit,
                                 calibration=dict(scale=c.scale,origin=c.origin,up=c.up.copy()))
                self.engine.measurements.append(measurement)
                messagebox.showinfo('Planar measurement',f'Length: ~{measurement["length"]:.2f} {unit}\nVertical separation: {delta[1]:+.2f} {unit}\nHorizontal separation: {delta[0]:+.2f} {unit}\n\nSaved in analysis.json. Valid only under the current planar calibration.')
            elif self.mode=='origin':c.origin=p[0]
            else:
                c.set_reference(*p,float(self.distance.get()),vertical=self.mode=='vertical',source=self.reference.get())
                c.units=self.units.get()
            self.status.set('Calibration applied to this segment. Values remain approximate planar estimates.')
        except Exception as exc:messagebox.showerror('Calibration',str(exc))
        self.mode=None;self.points=[];self.refresh()
    def preset(self,e=None):
        if self.reference.get().startswith('Wingspan'):self.distance.set('20');self.units.set('ft')
        elif self.reference.get().startswith('Platform'):self.distance.set('30');self.units.set('ft')
    def camera_changed(self):
        self.pause()
        if self.future or self.engine.tracks:
            self.compensate.set(self.engine.compensate);self.status.set('Start a new segment before changing camera correction.');return
        self.engine.compensate=self.compensate.get();self.refresh()
    def identify(self):
        self.pause()
        if not self.ready() or not self.configure_timing():return
        frame=self.engine.frame.copy();region=self.engine.region
        if region:
            x,y,w,h=region;frame=frame[y:y+h,x:x+w]
        else:x=y=0
        detector=self.detector
        def run():
            nonlocal detector
            if detector is None:
                detector=YoloDetector() if (ROOT/'models'/'yolox.onnx').exists() else Detector(ROOT/'models')
            results=detector.detect(frame)
            results=[(label,score,(b[0]+x,b[1]+y,b[2],b[3])) for label,score,b in results]
            return detector,results
        self.status.set('Identifying common object classes locally with YOLOX. Assign role labels such as Pilot by selecting the correct subject.')
        self.future=self.pool.submit(run);self.pending='detect'
    def stop_track(self):
        self.pause()
        if not self.ready():return
        selected=self.tree.selection()
        if selected:
            t=self.engine.tracks[int(selected[0])-1];t.status='stopped'
            self.engine.corrections.append(dict(frame=self.engine.index,id=t.id,label=t.label,box=None,source='manual stop'))
            self.refresh()
    def to_canvas(self,p):return (p[0]*self.view_scale+self.offset[0],p[1]*self.view_scale+self.offset[1])
    def draw(self):
        if self.engine.frame is None or self.future:return
        frame=self.engine.overlay();h,w=frame.shape[:2];cw,ch=max(1,self.canvas.winfo_width()),max(1,self.canvas.winfo_height())
        self.view_scale=min(cw/w,ch/h);dw,dh=max(1,int(w*self.view_scale)),max(1,int(h*self.view_scale));self.offset=((cw-dw)/2,(ch-dh)/2)
        im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).resize((dw,dh),Image.Resampling.BILINEAR)
        self.photo=ImageTk.PhotoImage(im);self.canvas.delete('all');self.canvas.create_image(*self.offset,anchor='nw',image=self.photo)
    def refresh(self):
        self.draw();e=self.engine
        self.time_text.set(f'{e.index/e.fps:06.2f}s  |  frame {e.index}')
        throughput=f'\nProcessing: {self.processing_fps:.1f} fps (no skipped frames)' if self.processing_fps else ''
        self.camera_text.set('Camera: '+(e.motion.quality if e.compensate else 'uncorrected — assumes fixed camera')+throughput)
        c=e.calibration
        self.cal_text.set(f'APPROX. PLANAR · {c.scale:.5f} {c.units}/reference pixel' if c.scale else 'PIXELS ONLY · no scale reference')
        selected=self.tree.selection();selected=selected[0] if selected else None
        for item in self.tree.get_children():self.tree.delete(item)
        for t in e.tracks:
            dist=f'{t.backend} / {t.tracking_score:.2f}' if t.tracking_score is not None else t.backend
            self.tree.insert('', 'end',iid=str(t.id),values=(f'{t.id} · {t.label}',t.status,dist))
        if selected and self.tree.exists(selected):self.tree.selection_set(selected)
        elif e.tracks:self.tree.selection_set(str(e.tracks[0].id))
        self.update_metrics()
    def change_view(self):
        if self.future:return
        self.engine.view_mode=self.view_mode.get();self.engine.show_guides=self.guides.get();self.draw()
    def set_redetect(self):
        if self.future:
            self.redetect.set(self.engine.redetect);return
        self.engine.redetect=self.redetect.get();self.engine.detector=self.detector
        if self.redetect.get() and self.detector is None:self.status.set('Click Identify objects once to load the detector before automatic refresh.')
    def open_review(self):
        path=ROOT/'reviewed_flight'/'Flugtag_tracking_full.mp4'
        if path.exists():os.startfile(str(path))
        else:messagebox.showinfo('Review','Run review_video.py to generate the complete tracked review.')
    def open_graphs(self):
        if self.graph_window and self.graph_window.winfo_exists():self.graph_window.lift();return
        self.graph_window=tk.Toplevel(self.root);self.graph_window.title('FlightLab | Measurements & graphs');self.graph_window.geometry('850x730');self.graph_window.configure(bg=PANEL)
        tabs=ttk.Notebook(self.graph_window);tabs.pack(fill='both',expand=True,padx=12,pady=12)
        graph=tk.Frame(tabs,bg=PANEL);cal=tk.Frame(tabs,bg=PANEL);tabs.add(graph,text='Motion graphs');tabs.add(cal,text='Calibration & timing')
        chooser=ttk.Combobox(graph,textvariable=self.graph_metric,values=['Image vertical position','Image horizontal position','Planar displacement','Calibrated center height','Planar speed'],state='readonly')
        chooser.pack(fill='x',padx=15,pady=12);chooser.bind('<<ComboboxSelected>>',lambda e:self.update_metrics())
        self.plot=tk.Canvas(graph,bg='#0c1926',height=360,highlightthickness=0);self.plot.pack(fill='both',expand=True,padx=15,pady=5)
        self.plot.bind('<Configure>',lambda e:self.update_metrics())
        tk.Label(graph,textvariable=self.metric_text,bg=PANEL,fg=FG,justify='left',anchor='w',font=('Consolas',10)).pack(fill='x',padx=20,pady=10)
        self.note(graph,'Select an object in the main window. Image-coordinate graphs include camera motion. Calibrated values depend on your scale and timing assumptions.')
        right=cal
        self.section(right,'02  /  SCALE & HEIGHT')
        preset=ttk.Combobox(right,textvariable=self.reference,values=['Wingspan · 20 ft','Platform · 30 ft','Custom reference'],state='readonly')
        preset.pack(fill='x',padx=14,pady=3);preset.bind('<<ComboboxSelected>>',self.preset)
        row=self.row(right);ttk.Entry(row,textvariable=self.distance,width=9).pack(side='left')
        ttk.Combobox(row,textvariable=self.units,values=['ft','m'],width=5,state='readonly').pack(side='left',padx=5)
        ttk.Button(row,text='Mark scale',command=lambda:self.select('scale')).pack(side='right')
        row=self.row(right);ttk.Button(row,text='Mark vertical reference',command=lambda:self.select('vertical')).pack(side='left')
        ttk.Button(row,text='Zero height',command=lambda:self.select('origin')).pack(side='right')
        ttk.Button(right,text='Measure two points',command=lambda:self.select('measure')).pack(fill='x',padx=14,pady=3)
        tk.Label(right,textvariable=self.cal_text,bg=PANEL,fg=ACCENT,wraplength=320,justify='left',anchor='w',font=('Segoe UI',9)).pack(fill='x',padx=14,pady=4)
        self.note(right,'Vertical: click base, then top. Scale applies only at comparable depth. Reported pilot landing: 42 ft (independent reference).')
        row=self.row(right);tk.Label(row,text='Physical seconds / video second',bg=PANEL,fg=FG).pack(side='left')
        ttk.Entry(row,textvariable=self.factor,width=7).pack(side='right')
        self.note(right,'Set before creating tracks. 1 = normal speed; 0.25 = confirmed 4x slow motion. Reported pilot landing distance: 42 ft, independent of these estimates.')
        self.update_metrics()
    def update_metrics(self):
        if self.future or self.plot is None or not self.plot.winfo_exists():return
        self.plot.delete('all');selected=self.tree.selection()
        if not selected:self.metric_text.set('Select an object in the tracking table.');return
        t=self.engine.tracks[int(selected[0])-1];kind=self.graph_metric.get();c=self.engine.calibration
        valid=[s for s in t.samples if s.ref_x is not None];origin=c.position([valid[0].ref_x,valid[0].ref_y]) if valid else None
        series=[];last=None;unit='px';title=kind
        for s in t.samples:
            value=None;point=c.position([s.ref_x,s.ref_y]) if s.ref_x is not None else None
            if kind=='Image vertical position':value=self.engine.frame.shape[0]-s.image_y if s.status=='tracking' or s.ref_x is not None else None
            elif kind=='Image horizontal position':value=s.image_x if s.status=='tracking' or s.ref_x is not None else None
            elif kind=='Planar displacement':
                value=float(np.linalg.norm(point-origin)) if point is not None else None;unit=c.units if c.scale else 'px'
            elif kind=='Calibrated center height':
                value=float(point[1]) if point is not None and c.scale and c.origin is not None else None;unit=c.units
            elif kind=='Planar speed':
                if point is not None and last and last[0].frame==s.frame-1:
                    dt=s.elapsed-last[0].elapsed
                    if dt>0:value=float(np.linalg.norm(point-last[1])/dt)
                unit=(c.units if c.scale else 'px')+'/assumed s'
            series.append((s.video_time,value));last=(s,point) if point is not None else None
        values=[v for _,v in series if v is not None]
        self.metric_text.set(f'{t.label} | {t.status}\n{title}: '+(f'{values[-1]:.2f} {unit}' if values else 'no valid measurement')+'\nTracker response is not object-identity confidence.')
        if not values:return
        w=max(200,self.plot.winfo_width());h=max(150,self.plot.winfo_height());left,top,right,bottom=65,35,w-25,h-45
        x0=series[0][0];x1=max(x0+.1,series[-1][0]);y0=min(values);y1=max(values)
        pad=max(1,(y1-y0)*.1);y0-=pad;y1+=pad
        for i in range(6):
            xx=left+(right-left)*i/5;yy=bottom-(bottom-top)*i/5
            self.plot.create_line(xx,top,xx,bottom,fill='#213345');self.plot.create_line(left,yy,right,yy,fill='#213345')
            self.plot.create_text(xx,bottom+15,text=f'{x0+(x1-x0)*i/5:.1f}',fill=MUTED)
            self.plot.create_text(left-8,yy,text=f'{y0+(y1-y0)*i/5:.1f}',fill=MUTED,anchor='e')
        self.plot.create_text(left,14,text=f'{title} ({unit})',anchor='w',fill=FG)
        self.plot.create_text((left+right)/2,h-10,text='Video time (seconds)',fill=MUTED)
        chain=[]
        for x,y in series:
            if y is None:
                if len(chain)>=4:self.plot.create_line(*chain,fill=ACCENT,width=2)
                chain=[];continue
            chain.extend([left+(x-x0)/(x1-x0)*(right-left),bottom-(y-y0)/(y1-y0)*(bottom-top)])
        if len(chain)>=4:self.plot.create_line(*chain,fill=ACCENT,width=2)
    def export(self):
        self.pause()
        if not self.ready():return
        out=filedialog.askdirectory(title='Choose parent folder for a timestamped analysis export')
        if not out:return
        path=Path(out)/('FlightLab_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        try:
            self.engine.export(path,self.path);self.status.set(f'Exported CSV, analysis.json, and overlay.png to {path}')
        except Exception as exc:messagebox.showerror('Export failed',str(exc))
    def record_frame(self):
        out=self.engine.overlay();text=f'Video {self.engine.index/self.engine.fps:.2f}s | {self.engine.view_mode} | Object tracking'
        cv2.rectangle(out,(0,out.shape[0]-34),(out.shape[1],out.shape[0]),(20,30,42),-1)
        cv2.putText(out,text,(12,out.shape[0]-12),cv2.FONT_HERSHEY_SIMPLEX,.5,(230,240,240),1,cv2.LINE_AA)
        return out
    def record(self):
        self.pause()
        if not self.ready():return
        if self.writer:self.stop_recording();self.status.set('Overlay recording saved (silent video).');return
        path=filedialog.asksaveasfilename(defaultextension='.mp4',filetypes=[('MP4 video','*.mp4')])
        if not path:return
        if Path(path).resolve()==Path(self.path).resolve():
            messagebox.showerror('Choose another file','The original video cannot be overwritten.');return
        h,w=self.engine.frame.shape[:2];self.writer=cv2.VideoWriter(path,cv2.VideoWriter_fourcc(*'mp4v'),self.engine.fps,(w,h))
        if not self.writer.isOpened():self.stop_recording();messagebox.showerror('Recording failed','Cannot create MP4 file.');return
        self.writer.write(self.record_frame());self.record_button.configure(text='Stop recording')
        self.status.set('Recording overlays. Press Play. Output is silent and uses source-video timing.')
    def stop_recording(self):
        if self.writer:self.writer.release();self.writer=None
        self.record_button.configure(text='Record overlay MP4')
    def close(self):
        self.pause()
        if self.future:
            self.status.set('Finishing the current operation before closing…');self.root.after(200,self.close);return
        try:self.archive()
        finally:
            self.stop_recording()
            if self.cap:self.cap.release()
            self.pool.shutdown(wait=False);self.root.destroy()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('video',nargs='?');parser.add_argument('--setup');args=parser.parse_args()
    root=tk.Tk();app=App(root,args.video)
    if args.setup and app.cap:
        try:app.load_setup(args.setup)
        except Exception as exc:messagebox.showerror('Setup could not be loaded',str(exc))
    root.mainloop()
