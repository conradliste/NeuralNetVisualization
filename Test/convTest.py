from manim import *
from visualize_net import ConvVisual

class cnnTest(MovingCameraScene):
    def construct(self):
        self.wait(1)
        c = ConvVisual([10, 10, 10], 64, 4, 2, plain=True)
        self.add(c)
        c = ConvVisual([10, 10, 10], 64, 4, 2)
        c.shift(np.array((-10, 5, 0)))
        self.add(c)
        self.wait(5)

s = cnnTest()
s.render()
open_file("./media/videos/1080p60/cnnTest.mp4")
