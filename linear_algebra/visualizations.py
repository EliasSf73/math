"""
Linear Algebra Visualizations with Manim
=========================================

This file contains 3Blue1Brown-style visualizations for linear algebra concepts.
Each scene is designed to be educational and beginner-friendly.

To run a specific scene:
    manimgl visualizations.py SceneName -o

Flags:
    -o  : Write to file and open it when done
    -w  : Write to file only
    -s  : Skip to end, show final frame
    -n 5: Skip to the 5th animation

Example:
    manimgl visualizations.py VectorBasics -o
"""

from manimlib import *
import numpy as np


# =============================================================================
# SCENE 1: Vector Basics
# =============================================================================
class VectorBasics(Scene):
    """
    Introduction to vectors: What they are and how to visualize them.

    This scene shows:
    - A 2D coordinate plane
    - Vectors as arrows from the origin
    - Vector components (x, y)
    """
    def construct(self):
        # Create the coordinate plane
        plane = NumberPlane(
            x_range=(-5, 5),
            y_range=(-4, 4),
            background_line_style={
                "stroke_color": BLUE_D,
                "stroke_width": 1,
                "stroke_opacity": 0.5
            }
        )
        plane.add_coordinate_labels(font_size=20)

        # Title
        title = Text("Vectors: The Building Blocks of Linear Algebra", font_size=36)
        title.to_edge(UP)
        title.set_backstroke(width=5)

        self.play(ShowCreation(plane), Write(title), run_time=2)
        self.wait()

        # Create a vector
        vector = Arrow(
            plane.c2p(0, 0),
            plane.c2p(3, 2),
            buff=0,
            color=YELLOW,
            stroke_width=6
        )

        # Vector label
        vector_label = Tex(r"\vec{v} = \begin{bmatrix} 3 \\ 2 \end{bmatrix}", font_size=36)
        vector_label.next_to(vector.get_end(), RIGHT)
        vector_label.set_backstroke(width=4)

        self.play(GrowArrow(vector))
        self.play(Write(vector_label))
        self.wait()

        # Show components
        x_component = DashedLine(
            plane.c2p(0, 0),
            plane.c2p(3, 0),
            color=RED
        )
        y_component = DashedLine(
            plane.c2p(3, 0),
            plane.c2p(3, 2),
            color=GREEN
        )

        x_label = Tex("3", color=RED, font_size=30)
        x_label.next_to(x_component, DOWN)
        y_label = Tex("2", color=GREEN, font_size=30)
        y_label.next_to(y_component, RIGHT)

        component_text = Text("A vector has components in each direction", font_size=24)
        component_text.to_edge(DOWN)
        component_text.set_backstroke(width=3)

        self.play(
            ShowCreation(x_component),
            ShowCreation(y_component),
            Write(x_label),
            Write(y_label),
            FadeIn(component_text)
        )
        self.wait(2)

        # Show another vector
        vector2 = Arrow(
            plane.c2p(0, 0),
            plane.c2p(-2, 3),
            buff=0,
            color=TEAL,
            stroke_width=6
        )

        vector2_label = Tex(r"\vec{w} = \begin{bmatrix} -2 \\ 3 \end{bmatrix}", font_size=36)
        vector2_label.next_to(vector2.get_end(), LEFT)
        vector2_label.set_backstroke(width=4)

        self.play(
            GrowArrow(vector2),
            Write(vector2_label)
        )
        self.wait(2)


# =============================================================================
# SCENE 2: Vector Addition
# =============================================================================
class VectorAddition(Scene):
    """
    Visual demonstration of vector addition.

    Shows the "tip-to-tail" method of adding vectors.
    """
    def construct(self):
        plane = NumberPlane(
            x_range=(-6, 6),
            y_range=(-4, 4),
            background_line_style={
                "stroke_color": BLUE_D,
                "stroke_width": 1,
                "stroke_opacity": 0.4
            }
        )
        plane.add_coordinate_labels(font_size=18)

        title = Text("Vector Addition: Tip-to-Tail Method", font_size=36)
        title.to_edge(UP)
        title.set_backstroke(width=5)

        self.play(ShowCreation(plane), Write(title))
        self.wait()

        # First vector v
        v = Arrow(plane.c2p(0, 0), plane.c2p(3, 1), buff=0, color=YELLOW, stroke_width=5)
        v_label = Tex(r"\vec{v}", color=YELLOW, font_size=36)
        v_label.next_to(v.get_center(), DOWN)

        self.play(GrowArrow(v), Write(v_label))
        self.wait()

        # Second vector w
        w = Arrow(plane.c2p(0, 0), plane.c2p(1, 2), buff=0, color=TEAL, stroke_width=5)
        w_label = Tex(r"\vec{w}", color=TEAL, font_size=36)
        w_label.next_to(w.get_center(), LEFT)

        self.play(GrowArrow(w), Write(w_label))
        self.wait()

        # Move w to tip of v (tip-to-tail)
        explanation = Text("Place the tail of w at the tip of v", font_size=24)
        explanation.to_edge(DOWN)
        explanation.set_backstroke(width=3)

        w_shifted = Arrow(
            plane.c2p(3, 1),
            plane.c2p(4, 3),
            buff=0,
            color=TEAL,
            stroke_width=5
        )
        w_label_shifted = Tex(r"\vec{w}", color=TEAL, font_size=36)
        w_label_shifted.next_to(w_shifted.get_center(), RIGHT)

        self.play(FadeIn(explanation))
        self.play(
            Transform(w.copy(), w_shifted),
            Transform(w_label.copy(), w_label_shifted)
        )
        self.wait()

        # Show the sum
        result_text = Text("The sum goes from origin to the final tip", font_size=24)
        result_text.to_edge(DOWN)
        result_text.set_backstroke(width=3)

        sum_vector = Arrow(
            plane.c2p(0, 0),
            plane.c2p(4, 3),
            buff=0,
            color=RED,
            stroke_width=6
        )
        sum_label = Tex(r"\vec{v} + \vec{w}", color=RED, font_size=36)
        sum_label.next_to(sum_vector.get_center(), UP + LEFT)
        sum_label.set_backstroke(width=4)

        self.play(FadeTransform(explanation, result_text))
        self.play(GrowArrow(sum_vector), Write(sum_label))
        self.wait()

        # Show equation
        equation = Tex(
            r"\begin{bmatrix} 3 \\ 1 \end{bmatrix} + "
            r"\begin{bmatrix} 1 \\ 2 \end{bmatrix} = "
            r"\begin{bmatrix} 4 \\ 3 \end{bmatrix}",
            font_size=36
        )
        equation.to_corner(DR)
        equation.set_backstroke(width=4)

        self.play(Write(equation))
        self.wait(2)


# =============================================================================
# SCENE 3: Linear Transformations
# =============================================================================
class LinearTransformation2D(Scene):
    """
    Visualize how matrices transform space.

    This is the core insight from 3Blue1Brown's Essence of Linear Algebra:
    - A matrix represents a linear transformation
    - The columns show where the basis vectors land
    """
    def construct(self):
        # Create a grid
        grid = NumberPlane(
            x_range=(-5, 5),
            y_range=(-5, 5)
        )

        title = Text("Linear Transformations: How Matrices Move Space", font_size=32)
        title.to_edge(UP)
        title.set_backstroke(width=5)

        self.play(ShowCreation(grid), Write(title))
        self.wait()

        # Show basis vectors
        i_hat = Arrow(grid.c2p(0, 0), grid.c2p(1, 0), buff=0, color=GREEN, stroke_width=6)
        j_hat = Arrow(grid.c2p(0, 0), grid.c2p(0, 1), buff=0, color=RED, stroke_width=6)

        i_label = Tex(r"\hat{i}", color=GREEN, font_size=36)
        i_label.next_to(i_hat, DOWN)
        j_label = Tex(r"\hat{j}", color=RED, font_size=36)
        j_label.next_to(j_hat, LEFT)

        basis_text = Text("These are the basis vectors", font_size=24)
        basis_text.to_edge(DOWN)
        basis_text.set_backstroke(width=3)

        self.play(
            GrowArrow(i_hat),
            GrowArrow(j_hat),
            Write(i_label),
            Write(j_label),
            FadeIn(basis_text)
        )
        self.wait()

        # Show transformation matrix
        matrix = [[1, 1], [0, 1]]  # Shear transformation

        matrix_tex = VGroup(
            Text("Apply the matrix:", font_size=24),
            IntegerMatrix(matrix, h_buff=0.8)
        )
        matrix_tex.arrange(RIGHT)
        matrix_tex.next_to(title, DOWN)
        matrix_tex.set_backstroke(width=4)

        self.play(FadeOut(basis_text), Write(matrix_tex))
        self.wait()

        # Apply the transformation
        transform_text = Text("Watch how space transforms!", font_size=24)
        transform_text.to_edge(DOWN)
        transform_text.set_backstroke(width=3)

        self.play(FadeIn(transform_text))

        # Transform the grid
        self.play(
            grid.animate.apply_matrix(matrix),
            i_hat.animate.put_start_and_end_on(
                grid.c2p(0, 0),
                grid.c2p(matrix[0][0], matrix[1][0])
            ),
            j_hat.animate.put_start_and_end_on(
                grid.c2p(0, 0),
                grid.c2p(matrix[0][1], matrix[1][1])
            ),
            run_time=3
        )
        self.wait()

        # Explain what happened
        explanation = VGroup(
            Tex(r"\hat{i} \rightarrow \begin{bmatrix} 1 \\ 0 \end{bmatrix}", color=GREEN, font_size=30),
            Tex(r"\hat{j} \rightarrow \begin{bmatrix} 1 \\ 1 \end{bmatrix}", color=RED, font_size=30)
        )
        explanation.arrange(DOWN, aligned_edge=LEFT)
        explanation.to_corner(DL)
        explanation.set_backstroke(width=4)

        column_text = Text("Matrix columns = where basis vectors land!", font_size=24)
        column_text.to_edge(DOWN)
        column_text.set_backstroke(width=3)

        self.play(
            FadeTransform(transform_text, column_text),
            Write(explanation)
        )
        self.wait(2)


# =============================================================================
# SCENE 4: Matrix Multiplication as Composition
# =============================================================================
class MatrixMultiplication(Scene):
    """
    Show that matrix multiplication = composing transformations.

    Applying matrix A then B is the same as applying (BA).
    """
    def construct(self):
        grid = NumberPlane(x_range=(-5, 5), y_range=(-5, 5))

        title = Text("Matrix Multiplication = Composition of Transformations", font_size=28)
        title.to_edge(UP)
        title.set_backstroke(width=5)

        self.play(ShowCreation(grid), Write(title))
        self.wait()

        # First transformation: Rotation by 90 degrees
        rotation = [[0, -1], [1, 0]]
        rotation_tex = VGroup(
            Text("First: Rotation", font_size=24),
            IntegerMatrix(rotation)
        )
        rotation_tex.arrange(DOWN)
        rotation_tex.to_corner(UL)
        rotation_tex.shift(DOWN)
        rotation_tex.set_backstroke(width=4)

        self.play(Write(rotation_tex))
        self.play(grid.animate.apply_matrix(rotation), run_time=2)
        self.wait()

        # Second transformation: Shear
        shear = [[1, 1], [0, 1]]
        shear_tex = VGroup(
            Text("Then: Shear", font_size=24),
            IntegerMatrix(shear)
        )
        shear_tex.arrange(DOWN)
        shear_tex.to_corner(UR)
        shear_tex.shift(DOWN)
        shear_tex.set_backstroke(width=4)

        self.play(Write(shear_tex))
        self.play(grid.animate.apply_matrix(shear), run_time=2)
        self.wait()

        # Show the combined matrix
        combined = np.dot(shear, rotation)

        result_text = VGroup(
            Text("Combined matrix:", font_size=24),
            IntegerMatrix(combined.tolist())
        )
        result_text.arrange(DOWN)
        result_text.to_edge(DOWN)
        result_text.set_backstroke(width=4)

        self.play(Write(result_text))
        self.wait(2)


# =============================================================================
# SCENE 5: Determinant - Area Scaling
# =============================================================================
class DeterminantVisualization(Scene):
    """
    The determinant tells you how much a transformation scales area.

    - det > 1: Area increases
    - det = 1: Area preserved
    - 0 < det < 1: Area decreases
    - det = 0: Collapses to a line (or point)
    - det < 0: Orientation flips
    """
    def construct(self):
        plane = NumberPlane(x_range=(-5, 5), y_range=(-4, 4))

        title = Text("The Determinant: How Much Does Area Change?", font_size=32)
        title.to_edge(UP)
        title.set_backstroke(width=5)

        self.play(ShowCreation(plane), Write(title))
        self.wait()

        # Create a unit square
        square = Square(side_length=1)
        square.set_fill(YELLOW, opacity=0.5)
        square.set_stroke(YELLOW, width=3)
        square.move_to(plane.c2p(0.5, 0.5))

        area_label = Tex("Area = 1", font_size=30)
        area_label.next_to(square, RIGHT)
        area_label.set_backstroke(width=3)

        self.play(FadeIn(square), Write(area_label))
        self.wait()

        # Transformation that doubles area
        matrix = [[2, 0], [0, 1]]  # Stretch by 2 in x-direction
        det = 2

        matrix_tex = VGroup(
            Text("Matrix:", font_size=24),
            IntegerMatrix(matrix),
            Tex(f"det = {det}", font_size=30, color=GREEN)
        )
        matrix_tex.arrange(RIGHT, buff=0.3)
        matrix_tex.to_corner(DL)
        matrix_tex.set_backstroke(width=4)

        self.play(Write(matrix_tex))

        # Transform
        new_area_label = Tex(f"Area = {det}", font_size=30)
        new_area_label.set_backstroke(width=3)

        self.play(
            plane.animate.apply_matrix(matrix),
            square.animate.apply_matrix(matrix),
            run_time=2
        )
        new_area_label.next_to(square, RIGHT)
        self.play(Transform(area_label, new_area_label))
        self.wait()

        # Reset and show negative determinant
        self.play(FadeOut(VGroup(plane, square, area_label, matrix_tex)))

        plane2 = NumberPlane(x_range=(-5, 5), y_range=(-4, 4))
        square2 = Square(side_length=1)
        square2.set_fill(YELLOW, opacity=0.5)
        square2.set_stroke(YELLOW, width=3)
        square2.move_to(plane2.c2p(0.5, 0.5))

        flip_title = Text("Negative Determinant: Orientation Flips!", font_size=28)
        flip_title.to_edge(UP)
        flip_title.set_backstroke(width=5)

        self.play(ShowCreation(plane2), FadeIn(square2), Transform(title, flip_title))

        # Reflection matrix
        reflection = [[-1, 0], [0, 1]]

        ref_tex = VGroup(
            Text("Reflection:", font_size=24),
            IntegerMatrix(reflection),
            Tex("det = -1", font_size=30, color=RED)
        )
        ref_tex.arrange(RIGHT, buff=0.3)
        ref_tex.to_corner(DL)
        ref_tex.set_backstroke(width=4)

        self.play(Write(ref_tex))
        self.play(
            plane2.animate.apply_matrix(reflection),
            square2.animate.apply_matrix(reflection),
            run_time=2
        )
        self.wait(2)


# =============================================================================
# SCENE 6: Eigenvectors - The Special Directions
# =============================================================================
class EigenvectorVisualization(Scene):
    """
    Eigenvectors are vectors that only get scaled (not rotated) by a transformation.

    Av = λv where λ is the eigenvalue (the scaling factor)
    """
    def construct(self):
        plane = NumberPlane(x_range=(-5, 5), y_range=(-4, 4))

        title = Text("Eigenvectors: Directions That Don't Rotate", font_size=32)
        title.to_edge(UP)
        title.set_backstroke(width=5)

        self.play(ShowCreation(plane), Write(title))
        self.wait()

        # Show several vectors
        vectors = []
        colors = [RED, BLUE, GREEN, ORANGE, PURPLE]
        angles = [0, PI/6, PI/4, PI/3, PI/2]

        for angle, color in zip(angles, colors):
            v = Arrow(
                plane.c2p(0, 0),
                plane.c2p(2*np.cos(angle), 2*np.sin(angle)),
                buff=0,
                color=color,
                stroke_width=4
            )
            vectors.append(v)

        self.play(*[GrowArrow(v) for v in vectors])
        self.wait()

        # Apply a transformation
        # Use a matrix with eigenvector along x-axis (eigenvalue 2) and y-axis (eigenvalue 0.5)
        matrix = [[2, 0], [0, 0.5]]

        matrix_tex = VGroup(
            Text("Matrix:", font_size=24),
            Tex(r"\begin{bmatrix} 2 & 0 \\ 0 & 0.5 \end{bmatrix}", font_size=30)
        )
        matrix_tex.arrange(RIGHT)
        matrix_tex.to_corner(UL).shift(DOWN)
        matrix_tex.set_backstroke(width=4)

        self.play(Write(matrix_tex))

        watch_text = Text("Watch which vectors only get scaled (not rotated)!", font_size=22)
        watch_text.to_edge(DOWN)
        watch_text.set_backstroke(width=3)
        self.play(FadeIn(watch_text))

        # Apply transformation
        self.play(
            plane.animate.apply_matrix(matrix),
            *[v.animate.apply_matrix(matrix) for v in vectors],
            run_time=3
        )
        self.wait()

        # Highlight eigenvectors
        eigen_text = VGroup(
            Tex(r"\text{Red (x-axis): Eigenvalue } \lambda = 2", color=RED, font_size=24),
            Tex(r"\text{Purple (y-axis): Eigenvalue } \lambda = 0.5", color=PURPLE, font_size=24)
        )
        eigen_text.arrange(DOWN, aligned_edge=LEFT)
        eigen_text.to_corner(DR)
        eigen_text.set_backstroke(width=3)

        self.play(
            FadeTransform(watch_text, eigen_text),
            vectors[0].animate.set_stroke(width=8),
            vectors[4].animate.set_stroke(width=8)
        )
        self.wait(2)


# =============================================================================
# SCENE 7: Span of Vectors
# =============================================================================
class SpanVisualization(Scene):
    """
    The span of vectors is all possible linear combinations.

    For two non-parallel 2D vectors, the span is the entire plane.
    """
    def construct(self):
        plane = NumberPlane(x_range=(-5, 5), y_range=(-4, 4))
        plane.set_opacity(0.3)

        title = Text("Span: All Possible Linear Combinations", font_size=32)
        title.to_edge(UP)
        title.set_backstroke(width=5)

        self.play(ShowCreation(plane), Write(title))
        self.wait()

        # Two vectors
        v1 = Arrow(plane.c2p(0, 0), plane.c2p(2, 1), buff=0, color=YELLOW, stroke_width=5)
        v2 = Arrow(plane.c2p(0, 0), plane.c2p(1, 2), buff=0, color=TEAL, stroke_width=5)

        v1_label = Tex(r"\vec{v}", color=YELLOW, font_size=30)
        v1_label.next_to(v1.get_end(), RIGHT)
        v2_label = Tex(r"\vec{w}", color=TEAL, font_size=30)
        v2_label.next_to(v2.get_end(), UP)

        self.play(GrowArrow(v1), GrowArrow(v2), Write(v1_label), Write(v2_label))
        self.wait()

        # Show linear combination
        formula = Tex(r"a\vec{v} + b\vec{w}", font_size=36)
        formula.to_corner(UL).shift(DOWN)
        formula.set_backstroke(width=4)

        self.play(Write(formula))

        # Animate different combinations
        combinations = [
            (1, 0),
            (0, 1),
            (1, 1),
            (2, 0.5),
            (-1, 1),
            (0.5, -0.5),
            (1.5, 1.5),
        ]

        result_vector = None
        for a, b in combinations:
            end_x = a * 2 + b * 1
            end_y = a * 1 + b * 2

            new_result = Arrow(
                plane.c2p(0, 0),
                plane.c2p(end_x, end_y),
                buff=0,
                color=RED,
                stroke_width=4
            )

            coeff_label = Tex(f"a={a}, b={b}", font_size=24)
            coeff_label.to_edge(DOWN)
            coeff_label.set_backstroke(width=3)

            if result_vector is None:
                self.play(GrowArrow(new_result), FadeIn(coeff_label), run_time=0.5)
            else:
                self.play(
                    Transform(result_vector, new_result),
                    FadeIn(coeff_label),
                    run_time=0.5
                )
                self.remove(coeff_label)

            result_vector = new_result
            self.wait(0.3)

        # Show that span is the whole plane
        span_text = Text("Span = The entire 2D plane!", font_size=28, color=GREEN)
        span_text.to_edge(DOWN)
        span_text.set_backstroke(width=4)

        self.play(
            FadeIn(span_text),
            plane.animate.set_opacity(0.7)
        )
        self.wait(2)


# =============================================================================
# SCENE 8: Null Space
# =============================================================================
class NullSpaceVisualization(Scene):
    """
    The null space is all vectors that get sent to zero by a matrix.

    For a singular matrix, the null space is non-trivial.
    """
    def construct(self):
        plane = NumberPlane(x_range=(-5, 5), y_range=(-4, 4))

        title = Text("Null Space: Vectors That Map to Zero", font_size=32)
        title.to_edge(UP)
        title.set_backstroke(width=5)

        self.play(ShowCreation(plane), Write(title))
        self.wait()

        # Matrix that collapses to a line
        # This sends (1, 2) to zero since [1,2]·[1,2] = [1,2] and [1,2]·[-2,1] = 0
        matrix = [[1, 2], [2, 4]]  # Rank 1 matrix

        matrix_tex = VGroup(
            Text("Singular matrix:", font_size=24),
            Tex(r"\begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}", font_size=30)
        )
        matrix_tex.arrange(RIGHT)
        matrix_tex.to_corner(UL).shift(DOWN)
        matrix_tex.set_backstroke(width=4)

        self.play(Write(matrix_tex))

        # Show the null space direction
        null_direction = Arrow(
            plane.c2p(-2, 1),
            plane.c2p(2, -1),
            buff=0,
            color=RED,
            stroke_width=5
        )
        null_label = Text("Null Space", color=RED, font_size=24)
        null_label.next_to(null_direction.get_center(), UP)
        null_label.set_backstroke(width=3)

        self.play(ShowCreation(null_direction), Write(null_label))

        explanation = Text("All vectors on this line map to zero", font_size=22)
        explanation.to_edge(DOWN)
        explanation.set_backstroke(width=3)
        self.play(FadeIn(explanation))
        self.wait()

        # Apply the transformation
        apply_text = Text("Watch the transformation collapse space to a line", font_size=22)
        apply_text.to_edge(DOWN)
        apply_text.set_backstroke(width=3)

        self.play(FadeTransform(explanation, apply_text))
        self.play(
            plane.animate.apply_matrix(matrix),
            run_time=3
        )

        det_text = Tex(r"\det(A) = 0 \text{ means null space is non-trivial}", font_size=28)
        det_text.to_edge(DOWN)
        det_text.set_backstroke(width=4)

        self.play(FadeTransform(apply_text, det_text))
        self.wait(2)


# =============================================================================
# SCENE 9: Dot Product Visualization
# =============================================================================
class DotProductVisualization(Scene):
    """
    The dot product measures how much two vectors point in the same direction.

    v · w = |v| |w| cos(θ)
    """
    def construct(self):
        plane = NumberPlane(x_range=(-5, 5), y_range=(-4, 4))
        plane.set_opacity(0.3)

        title = Text("Dot Product: Measuring Alignment", font_size=32)
        title.to_edge(UP)
        title.set_backstroke(width=5)

        self.play(ShowCreation(plane), Write(title))
        self.wait()

        # Fixed vector v
        v = Arrow(plane.c2p(0, 0), plane.c2p(3, 0), buff=0, color=YELLOW, stroke_width=5)
        v_label = Tex(r"\vec{v}", color=YELLOW, font_size=30)
        v_label.next_to(v.get_end(), DOWN)

        self.play(GrowArrow(v), Write(v_label))

        # Moving vector w
        w_end = plane.c2p(2, 2)
        w = Arrow(plane.c2p(0, 0), w_end, buff=0, color=TEAL, stroke_width=5)
        w_label = Tex(r"\vec{w}", color=TEAL, font_size=30)
        w_label.next_to(w.get_end(), UP)

        self.play(GrowArrow(w), Write(w_label))

        # Angle arc
        angle = Angle(v, w, radius=0.5, color=WHITE)
        angle_label = Tex(r"\theta", font_size=24)
        angle_label.move_to(angle.point_from_proportion(0.5) + 0.3 * (UP + RIGHT))

        self.play(ShowCreation(angle), Write(angle_label))

        # Formula
        formula = Tex(r"\vec{v} \cdot \vec{w} = |\vec{v}||\vec{w}|\cos\theta", font_size=32)
        formula.to_corner(UL).shift(DOWN)
        formula.set_backstroke(width=4)

        self.play(Write(formula))
        self.wait()

        # Show projection
        projection_text = Text("Dot product = (length of projection) × (length of v)", font_size=22)
        projection_text.to_edge(DOWN)
        projection_text.set_backstroke(width=3)

        # Project w onto v
        proj_length = 2  # Since w = (2, 2) projected onto x-axis
        proj_line = DashedLine(
            plane.c2p(2, 2),
            plane.c2p(2, 0),
            color=WHITE
        )
        proj_point = Dot(plane.c2p(2, 0), color=RED)
        proj_label = Text("projection", font_size=18)
        proj_label.next_to(proj_point, DOWN)

        self.play(
            FadeIn(projection_text),
            ShowCreation(proj_line),
            FadeIn(proj_point),
            Write(proj_label)
        )
        self.wait(2)


# =============================================================================
# SCENE 10: Cross Product (3D)
# =============================================================================
class CrossProductVisualization(ThreeDScene):
    """
    The cross product produces a vector perpendicular to both input vectors.

    The magnitude equals the area of the parallelogram they span.
    """
    def construct(self):
        # Set up 3D axes
        axes = ThreeDAxes(
            x_range=(-4, 4),
            y_range=(-4, 4),
            z_range=(-4, 4)
        )

        title = Text("Cross Product: The Perpendicular Vector", font_size=28)
        title.to_edge(UP)
        title.fix_in_frame()
        title.set_backstroke(width=5)

        self.add(title)
        self.play(ShowCreation(axes))

        # Two vectors in the xy-plane
        v = Arrow3D(
            start=axes.c2p(0, 0, 0),
            end=axes.c2p(2, 0, 0),
            color=YELLOW
        )
        w = Arrow3D(
            start=axes.c2p(0, 0, 0),
            end=axes.c2p(1, 2, 0),
            color=TEAL
        )

        v_label = Tex(r"\vec{v}", color=YELLOW, font_size=30)
        v_label.move_to(axes.c2p(2.3, 0, 0))
        v_label.fix_in_frame()

        w_label = Tex(r"\vec{w}", color=TEAL, font_size=30)
        w_label.move_to(axes.c2p(1, 2.3, 0))
        w_label.fix_in_frame()

        self.play(ShowCreation(v), ShowCreation(w))
        self.wait()

        # Cross product result
        cross = Arrow3D(
            start=axes.c2p(0, 0, 0),
            end=axes.c2p(0, 0, 4),  # 2×2 - 0×1 = 4 in z-direction
            color=RED
        )

        cross_label = Tex(r"\vec{v} \times \vec{w}", color=RED, font_size=30)
        cross_label.move_to(axes.c2p(0.3, 0, 4.3))
        cross_label.fix_in_frame()

        result_text = Text("Cross product is perpendicular to both!", font_size=22)
        result_text.to_edge(DOWN)
        result_text.fix_in_frame()
        result_text.set_backstroke(width=3)

        self.play(ShowCreation(cross), FadeIn(result_text))
        self.wait()

        # Rotate camera to show perpendicularity
        self.play(
            self.frame.animate.set_euler_angles(phi=70 * DEG, theta=30 * DEG),
            run_time=2
        )
        self.wait(2)


# =============================================================================
# HOW TO RUN THESE SCENES
# =============================================================================
"""
To run any of these scenes, use the terminal:

    cd /home/user/math/linear_algebra
    manimgl visualizations.py VectorBasics -o

Available scenes:
1. VectorBasics          - Introduction to vectors
2. VectorAddition        - Tip-to-tail vector addition
3. LinearTransformation2D - How matrices transform space
4. MatrixMultiplication  - Composition of transformations
5. DeterminantVisualization - Determinants and area scaling
6. EigenvectorVisualization - Eigenvectors don't rotate
7. SpanVisualization     - Linear combinations and span
8. NullSpaceVisualization - Vectors that map to zero
9. DotProductVisualization - Dot product and projections
10. CrossProductVisualization - Cross product in 3D

Tips:
- Use -o to write to file and open automatically
- Use -w to just write to file
- Use -s to skip to the final frame
- Use -n 3 to skip to animation #3
"""
