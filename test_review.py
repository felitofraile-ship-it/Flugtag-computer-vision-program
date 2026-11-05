import unittest
import numpy as np
import cv2
from review_video import clipped,trim_to_shots,align_water_wing

class ReviewTests(unittest.TestCase):
    def test_bounds(self):
        self.assertEqual(clipped([959,539,300,300]),[952,532,8,8])
    def test_cut_has_one_owner(self):
        d=dict(fps=30,total_frames=100,segments=[dict(start=0,end=1),dict(start=1,end=2)],
               observations=[dict(frame=30,segment=0,id=1),dict(frame=30,segment=1,id=1)])
        trim_to_shots(d)
        self.assertEqual(d['observations'],[dict(frame=30,segment=1,id=1)])
    def test_wing_color_alignment_rejects_vertical_subject(self):
        frame=np.full((540,960,3),100,np.uint8)
        cv2.rectangle(frame,(310,200),(490,210),(220,80,30),-1)
        cv2.rectangle(frame,(350,260),(370,300),(220,80,30),-1)
        result=align_water_wing(frame,[300,250,200,20])
        self.assertAlmostEqual(result[1],195,delta=2)

if __name__=='__main__':unittest.main()
