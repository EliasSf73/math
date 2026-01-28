# Linear Algebra Visualizations with Manim

Create 3Blue1Brown-style mathematical animations for your blog!

## Quick Start

```bash
# Navigate to this directory
cd /home/user/math/linear_algebra

# Run your first visualization
manimgl visualizations.py VectorBasics -o
```

## What is Manim?

Manim is the animation engine created by Grant Sanderson (3Blue1Brown) to make his famous math videos. It lets you create precise, programmatic animations of mathematical concepts.

## Available Scenes

| Scene | Description | Difficulty |
|-------|-------------|------------|
| `VectorBasics` | Introduction to vectors and their components | Beginner |
| `VectorAddition` | Tip-to-tail method for adding vectors | Beginner |
| `LinearTransformation2D` | How matrices transform 2D space | Beginner |
| `MatrixMultiplication` | Composition of transformations | Intermediate |
| `DeterminantVisualization` | Determinants and area scaling | Intermediate |
| `EigenvectorVisualization` | Vectors that only get scaled | Intermediate |
| `SpanVisualization` | Linear combinations and span | Intermediate |
| `NullSpaceVisualization` | Vectors that map to zero | Advanced |
| `DotProductVisualization` | Dot product as projection | Intermediate |
| `CrossProductVisualization` | Cross product in 3D | Advanced |

## Command Line Flags

```bash
manimgl visualizations.py SceneName [flags]
```

| Flag | Description |
|------|-------------|
| `-o` | Write to file AND open it when done |
| `-w` | Write to file only |
| `-s` | Skip to end, show only final frame |
| `-so` | Save final frame as image and show it |
| `-n 5` | Skip to animation #5 |
| `-f` | Fullscreen playback |

## Creating Your Own Scenes

### Basic Structure

Every Manim scene follows this pattern:

```python
from manimlib import *

class MyScene(Scene):
    def construct(self):
        # Your animation code goes here
        pass
```

### Core Concepts

#### 1. Creating Objects (Mobjects)

```python
# Basic shapes
circle = Circle(radius=1, color=BLUE)
square = Square(side_length=2, color=RED)
line = Line(start=LEFT, end=RIGHT)

# Text and math
text = Text("Hello World", font_size=48)
equation = Tex(r"E = mc^2", font_size=72)
matrix = IntegerMatrix([[1, 2], [3, 4]])

# Vectors and arrows
arrow = Arrow(start=ORIGIN, end=UP + RIGHT, color=YELLOW)
vector = Vector([2, 1], color=GREEN)

# Coordinate systems
plane = NumberPlane(x_range=(-5, 5), y_range=(-4, 4))
axes = Axes(x_range=(-3, 3), y_range=(-2, 2))
```

#### 2. Positioning Objects

```python
# Move to specific location
circle.move_to(RIGHT * 2 + UP)

# Position relative to other objects
text.next_to(circle, UP)
text.to_edge(DOWN)
text.to_corner(UL)  # Upper Left

# Built-in directions: UP, DOWN, LEFT, RIGHT, ORIGIN
# Corners: UL, UR, DL, DR
```

#### 3. Animations

```python
# Basic animations
self.play(FadeIn(circle))           # Fade in
self.play(FadeOut(circle))          # Fade out
self.play(Write(text))              # Write text
self.play(ShowCreation(line))       # Draw a line
self.play(GrowArrow(arrow))         # Grow an arrow

# Transform between objects
self.play(Transform(circle, square))
self.play(ReplacementTransform(a, b))

# Animate properties with .animate
self.play(circle.animate.shift(RIGHT * 2))
self.play(circle.animate.scale(2))
self.play(circle.animate.set_color(RED))

# Wait between animations
self.wait(2)  # Wait 2 seconds
```

#### 4. Linear Algebra Specific

```python
# Create a grid/plane
plane = NumberPlane(x_range=(-5, 5), y_range=(-5, 5))

# Apply a matrix transformation
matrix = [[2, 1], [0, 1]]
self.play(plane.animate.apply_matrix(matrix), run_time=3)

# Create vectors
v = Arrow(plane.c2p(0, 0), plane.c2p(3, 2), buff=0, color=YELLOW)

# Add coordinate labels
plane.add_coordinate_labels(font_size=20)
```

### Example: Your First Custom Scene

```python
from manimlib import *

class MyFirstScene(Scene):
    def construct(self):
        # Create a title
        title = Text("My First Manim Scene!", font_size=48)
        title.to_edge(UP)

        # Create a circle
        circle = Circle(radius=1.5, color=BLUE)
        circle.set_fill(BLUE, opacity=0.5)

        # Animate!
        self.play(Write(title))
        self.play(ShowCreation(circle))
        self.wait()

        # Transform circle to square
        square = Square(side_length=3, color=RED)
        square.set_fill(RED, opacity=0.5)

        self.play(Transform(circle, square))
        self.wait(2)
```

Save this in a file and run:
```bash
manimgl your_file.py MyFirstScene -o
```

## Colors

Manim has built-in colors:

```
BLUE, BLUE_A, BLUE_B, BLUE_C, BLUE_D, BLUE_E
RED, GREEN, YELLOW, ORANGE, PURPLE, PINK
TEAL, MAROON, GOLD, GREY, WHITE, BLACK
```

## Tips for Great Visualizations

1. **Start Simple**: Begin with basic shapes and animations
2. **Use `.set_backstroke()`**: Makes text readable over grids
3. **Control Timing**: Use `run_time=3` for slower animations
4. **Add Pauses**: `self.wait()` lets viewers absorb information
5. **Use Comments**: Document your scenes for future reference

## Resources

- [Manim Documentation](https://3b1b.github.io/manim/)
- [3Blue1Brown Video Code](https://github.com/3b1b/videos)
- [Manim Discord](https://discord.com/invite/bYCyhM9Kz2)
- [Example Scenes](/home/user/math/manim/example_scenes.py)

## Troubleshooting

**"No module named manimlib"**
```bash
cd /home/user/math/manim
pip install -e .
```

**Window doesn't open**
- Make sure you have a display connected
- Try using `-w` to write to file instead

**LaTeX errors**
- Install LaTeX: `apt-get install texlive-full`
- Or use `Text()` instead of `Tex()` for non-math text

## File Structure

```
/home/user/math/
├── manim/                 # Manim library (cloned from 3b1b)
│   ├── manimlib/          # Core library code
│   ├── example_scenes.py  # Official examples
│   └── ...
└── linear_algebra/
    ├── README.md          # This file
    └── visualizations.py  # Your linear algebra scenes
```

Happy animating!
