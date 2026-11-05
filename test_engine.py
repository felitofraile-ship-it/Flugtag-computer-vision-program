import csv
import json
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
from engine import Calibration, Engine, Motion, Sample, Track, scene_cut, transform

class GeometryTests(unittest.TestCase):
    def test_scale_and_vertical_height(self):
        c=Calibration();c.set_reference([100,400],[100,100],30,True)
        np.testing.assert_allclose(c.position([200,200]),[10,20])
        self.assertEqual(c.scale,.1)
        with self.assertRaises(ValueError):c.set_reference([0,0],[0,0],20)
        with self.assertRaises(ValueError):c.set_reference([0,0],[100,0],float('nan'))

    def test_rotated_vertical(self):
        c=Calibration();c.set_reference([0,0],[100,-100],20,True)
        np.testing.assert_allclose(c.position([50,-50]),[0,10],atol=1e-9)

    def test_camera_motion_translation(self):
        rng=np.random.default_rng(20)
        a=rng.integers(0,256,(300,480,3),dtype=np.uint8)
        b=cv2.warpAffine(a,np.float32([[1,0,4],[0,1,3]]),(480,300))
        motion=Motion();self.assertTrue(motion.update(a,b,[],None,True))
        np.testing.assert_allclose(transform([204,103],motion.matrix),[200,100],atol=.3)

    def test_camera_failure_latches(self):
        a=np.zeros((200,300,3),np.uint8);m=Motion()
        self.assertFalse(m.update(a,a,[],None,True));self.assertFalse(m.valid)

    def test_scene_cut(self):
        a=np.full((100,100,3),(0,0,255),np.uint8)
        b=np.full((100,100,3),(255,0,0),np.uint8)
        self.assertTrue(scene_cut(a,b));self.assertFalse(scene_cut(a,a))
        # A major rearrangement may preserve the color histogram (broadcast edit).
        a=np.zeros((100,100,3),np.uint8);a[:50]=200
        self.assertTrue(scene_cut(a,np.flipud(a).copy()))

    def test_measurement_export_and_time(self):
        e=Engine(30);e.reset(np.zeros((200,300,3),np.uint8),0)
        e.calibration.set_reference([0,100],[100,100],20)
        e.calibration.origin=[0,100]
        t=Track(1,'pilot',(0,0,20,20),None)
        t.samples=[Sample(0,0,0,0,100,0,100,'tracking'),Sample(1,1/30,.25,3,96,3,96,'tracking')]
        e.tracks=[t];m=e.metrics(t)
        self.assertAlmostEqual(m['displacement'],1);self.assertAlmostEqual(m['path'],1)
        self.assertAlmostEqual(m['speed'],4);self.assertAlmostEqual(m['height'],.8)
        with tempfile.TemporaryDirectory() as folder:
            e.export(folder,'example.mp4')
            data=json.loads((Path(folder)/'analysis.json').read_text())
            self.assertEqual(data['reported_pilot_distance_ft'],42)
            with (Path(folder)/'tracks.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),2);self.assertEqual(rows[1]['unit'],'ft')

    def test_missing_points_not_measured(self):
        e=Engine();e.reset(np.zeros((100,100,3),np.uint8),0)
        t=Track(1,'pilot',(0,0,20,20),None)
        t.samples=[Sample(0,0,0,10,10,10,10,'tracking'),Sample(1,.03,.03,10,10,None,None,'lost')]
        self.assertIsNone(e.metrics(t))

    def test_tracker_moves_with_object(self):
        rng=np.random.default_rng(2);patch=rng.integers(0,255,(48,48,3),dtype=np.uint8)
        a=np.full((240,320,3),50,np.uint8);a[80:128,60:108]=patch
        e=Engine();e.reset(a,0);t=e.add((60,80,48,48),'synthetic object')
        for i in range(1,9):
            b=np.full_like(a,50);b[80+i:128+i,60+2*i:108+2*i]=patch;e.process(b,i)
        self.assertEqual(t.status,'tracking')
        self.assertAlmostEqual(t.box[0],76,delta=5);self.assertAlmostEqual(t.box[1],88,delta=5)
        self.assertGreater(e.metrics(t)['displacement'],12)
        with self.assertRaises(ValueError):e.process(b,20)

    def test_stopped_track_keeps_endpoint(self):
        rng=np.random.default_rng(5);a=rng.integers(0,255,(120,160,3),dtype=np.uint8)
        e=Engine();e.reset(a,0);t=e.add((20,20,30,30),'stopped')
        t.status='stopped';e.process(a,1)
        self.assertEqual(len(t.samples),1);self.assertIsNotNone(e.metrics(t))
        e.cut=True;self.assertIsNone(e.metrics(t))

    def test_correction_preserves_identity(self):
        a=np.random.default_rng(6).integers(0,255,(160,240,3),dtype=np.uint8)
        e=Engine();e.reset(a,0);t=e.add((30,30,35,35),'Pilot')
        e.reanchor(t.id,(36,32,35,35))
        self.assertEqual(len(e.tracks),1);self.assertEqual(t.id,1);self.assertEqual(t.box,(36,32,35,35))
        self.assertEqual(len(t.samples),1)

    def test_measurement_failure_does_not_stop_tracking(self):
        a=np.random.default_rng(8).integers(0,255,(180,280,3),dtype=np.uint8)
        e=Engine();e.reset(a,0);t=e.add((40,40,40,40),'Pilot');e.compensate=True;e.motion.valid=False
        e.process(a,1)
        self.assertEqual(t.status,'tracking');self.assertEqual(t.samples[-1].status,'tracking')
        self.assertIsNone(t.samples[-1].ref_x);self.assertIsNone(e.metrics(t))

    def test_top_view_has_no_trajectory_line(self):
        a=np.zeros((150,200,3),np.uint8);e=Engine();e.reset(a,0);e.view_mode='Top view'
        t=Track(1,'Pilot',(100,80,20,20),None)
        t.samples=[Sample(0,0,0,20,20,20,20,'tracking'),Sample(1,1,1,70,50,70,50,'tracking')];e.tracks=[t]
        top=e.overlay();e.view_mode='Side view';side=e.overlay()
        self.assertEqual(int(top[35,45].sum()),0);self.assertGreater(int(side[35,45].sum()),0)

if __name__=='__main__':unittest.main()
