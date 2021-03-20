from manim import *

class BGrid(VGroup):
    def __init__(self, rows, cols, unit=1.0, stroke_width=1, stroke_color=WHITE, stroke_opacity=1, fill_color=BLACK, fill_opacity=1, copy_data=None, *vmobjects, **kwargs):
        super().__init__(*vmobjects, **kwargs)
        self.rows = rows
        self.cols = cols
        self.unit = unit
        if copy_data is not None:
            assert(len(copy_data) == self.rows and len(copy_data[0]) == self.cols)
        for r in range (0, self.rows):
            for c in range (0, self.cols):
                square = Square(
                    stroke_width=stroke_width,
                    stroke_color=stroke_color,
                    stroke_opacity=stroke_opacity,
                    side_length=self.unit,
                    fill_color=fill_color,
                    fill_opacity=fill_opacity
                )
                if copy_data is not None:
                    if isinstance(copy_data[r][c], float) or isinstance(copy_data[r][c], int):
                        n = DecimalNumber(copy_data[r][c])
                        n.scale(0.5)
                        square.add(n)
                    else:
                        t = Text(str(copy_data[r][c]))
                        t.scale(0.5)
                        square.add(t)
                square.shift(np.array((c * self.unit, -1 * r * self.unit, 0.0)))
                self.add(square)

