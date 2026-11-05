"""Neural object tracking and higher-resolution class detection, all local."""
from pathlib import Path
import cv2
import numpy as np

MODELS=Path(__file__).resolve().parent/'models'
COCO=('person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,teddy bear,hair drier,toothbrush').split(',')

class NeuralTracker:
    name='NanoTrack'
    def __init__(self):
        p=cv2.TrackerNano_Params()
        p.backbone=str(MODELS/'nanotrack_backbone_sim.onnx')
        p.neckhead=str(MODELS/'nanotrack_head_sim.onnx')
        self.tracker=cv2.TrackerNano_create(p);self.score=None
    def init(self,frame,box):
        self.tracker.init(frame,tuple(map(int,box)))
    def update(self,frame):
        ok,box=self.tracker.update(frame);self.score=float(self.tracker.getTrackingScore())
        return bool(ok and self.score>=.45),box

def make_tracker(box=None):
    # Very wide wings are a poor fit for NanoTrack's square search crop. CSRT
    # preserves the selected aspect ratio and is steadier for these shapes.
    if box is not None and box[2]/max(1,box[3])>3.5:
        return cv2.TrackerCSRT_create()
    if (MODELS/'nanotrack_head_sim.onnx').exists():return NeuralTracker()
    return cv2.TrackerCSRT_create()

class YoloDetector:
    """YOLOX ONNX decoding, per the model's published stride/grid output format."""
    name='YOLOX'
    def __init__(self,path=None):
        self.net=cv2.dnn.readNetFromONNX(str(path or MODELS/'yolox.onnx'))
        grids=[];strides=[]
        for stride in (8,16,32):
            y,x=np.mgrid[:640//stride,:640//stride]
            grids.append(np.c_[x.ravel(),y.ravel()]);strides.extend([stride]*x.size)
        self.grid=np.concatenate(grids);self.stride=np.array(strides)[:,None]
    def detect(self,frame,threshold=.4):
        h,w=frame.shape[:2];ratio=min(640/w,640/h)
        canvas=np.full((640,640,3),114,np.uint8)
        resized=cv2.resize(frame,(int(w*ratio),int(h*ratio)))
        canvas[:resized.shape[0],:resized.shape[1]]=resized
        self.net.setInput(cv2.dnn.blobFromImage(canvas,swapRB=True))
        p=self.net.forward().reshape(-1,85)
        scores=p[:,4,None]*p[:,5:];classes=scores.argmax(1);conf=scores.max(1)
        centers=(p[:,:2]+self.grid)*self.stride
        sizes=np.exp(np.clip(p[:,2:4],-15,15))*self.stride
        boxes=np.c_[centers-sizes/2,sizes]/ratio
        good=np.where(conf>=threshold)[0]
        if not len(good):return []
        keep=cv2.dnn.NMSBoxesBatched(boxes[good].tolist(),conf[good].tolist(),classes[good].tolist(),threshold,.45)
        result=[]
        for k in np.asarray(keep).reshape(-1):
            i=good[int(k)];x,y,bw,bh=boxes[i]
            x2=min(w,int(x+bw));y2=min(h,int(y+bh));x=max(0,int(x));y=max(0,int(y))
            if x2-x>=8 and y2-y>=8:result.append((COCO[int(classes[i])],float(conf[i]),(x,y,x2-x,y2-y)))
        return sorted(result,key=lambda x:x[1],reverse=True)
