from manim import *
from more_geometry import BGrid

class BGridTest(MovingCameraScene):
    def construct(self):
        copy = [[0.0, 5.0, 3.0],
                [9.0, 'wow', 2.5],
                [0.2, 0.3, 44]]
        b = BGrid(3, 3, copy_data=copy)
        b.shift(np.array((-1., 1., 0.0)))
        self.add(b)
        self.wait(5)

s = BGridTest()
s.render()
open_file("./media/videos/1080p60/BGridTest.mp4")