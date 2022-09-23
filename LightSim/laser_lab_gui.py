from manimlib import *
import numpy as np


class lab(Scene):
    def construct(self):
        """Manim uses a constuct method to generate a Scene"""

        for x in range(-15, 15):
            for y in range(-15, 15):
                self.add(Dot(np.array([x, y, 0]), color=GREY))

        self.add(
            NumberPlane(
                background_line_style={
                    "stroke_color": TEAL,
                    "stroke_width": 1,
                    "stroke_opacity": 0.3,
                }
            ).set_stroke(opacity=0.1)
        )

        self.lasers = []
        self.thin_lenses = []
        self.prisms = []
        self.elements = []
        self.snap = True
        self.rotation = 0
        self.add_element = self.laser
        self.cursor_opacity = 0.3
        self.cursor, _ = self.add_element(opacity=self.cursor_opacity)
        always(self.cursor.move_to, self.mouse_point)
        self.elements = []

    def thin_lens(self, opacity: float = 0.6, creation_method: str = "show creation"):

        return self.create_element(
            RoundedRectangle(
                corner_radius=0.05,
                height=1,
                width=0.1,
                color=BLUE,
                fill_opacity=opacity,
                stroke_opacity=opacity,
            ),
            "Thin Lens",
            creation_method,
        )

    def prism(self, opacity: float = 0.6, creation_method: str = "show creation"):
        # return self.create_element(Triangle(width = 0.2, color=BLUE, opacity=1), creation_method)
        return self.create_element(
            Polygon(
                *[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                color=BLUE,
                fill_opacity=opacity,
                stroke_opacity=opacity
            ),
            "Prism",
            creation_method,
            # offset=[0.175, 0.175, 0],
        )

    def laser(self, opacity: float = 1, creation_method: str = "show creation"):

        return self.create_element(
            Polygon(
                *[[0, 0.8, 0], [1, 0.8, 0], [1.1, 0.9, 0], [1, 1, 0], [0, 1, 0]],
                color=GREY,
                fill_color=GREY,
                fill_opacity=opacity,
                stroke_opacity=opacity
            ),
            "Laser",
            creation_method,
        )

    def flat_mirror(self, opacity: float = 1, creation_method: str = "show creation"):
        return self.create_element(
            Rectangle(width=0.2, height=1, color=GREY, fill_opacity=opacity),
            "Flat Mirror",
            creation_method,
        )

    def create_element(
        self, element, element_name: str, creation_method: str, offset=[0, 0, 0]
    ):
        """initialises an element

        Args:
            element (_type_): The geometry and object reference to the element
            element_name (str): a short description of the element
            creation_method (str): the method it is shown to the user
            offset (list, optional): Defaults to [0, 0, 0].

        Returns:
            _type_: _description_
        """
        if creation_method == "none":
            return element, element_name
        if creation_method == "show creation":
            create_bool = self.snap_to_position(
                element, element_name, self.mouse_point, offset
            )
            if create_bool:
                self.play(ShowCreation(element))
            return element, element_name
        if creation_method == "fade in":
            create_bool = self.snap_to_position(
                element, element_name, self.mouse_point, offset
            )
            if create_bool:
                self.play(FadeIn(element))
            return element, element_name

    def turn_on_laser(self):
        print("Checking if laser should turn on")
        mp = self.mouse_point
        pos = self.round_loc(mp)

        laser_here = False
        for i, el in enumerate(self.elements):
            if self.round_loc(el[0]) == pos:
                laser_here = True
                self.propagate_beam(pos, el[1])
        if not laser_here:
            t = Text("No laser found")
            self.play(FadeIn(t))
            self.wait()
            self.play(FadeOut(t))

    def propagate_beam(self, start, rotation):
        multiplier = 0.6
        a = np.add(
            start, [multiplier * np.cos(rotation), multiplier * np.sin(rotation), 0]
        )
        if self.test_angled(rotation):
            b = np.round(
                np.add(
                    a,
                    [
                        np.cos(rotation) * (1 - multiplier) * 2,
                        np.sin(rotation) * (1 - multiplier) * 2,
                        0,
                    ],
                )
            )
        else:
            b = np.round(
                np.add(
                    a,
                    [
                        np.cos(rotation) * (1 - multiplier),
                        np.sin(rotation) * (1 - multiplier),
                        0,
                    ],
                )
            )
        beam = Line(start=a, end=b, color=RED)
        self.play(ShowCreation(beam))

        for i in range(20):
            el_bool, el = self.element_here(b)
            if el_bool:
                pass
            else:
                c = np.round(np.add(b, [np.cos(rotation), np.sin(rotation), 0]))
                beam = Line(start=b, end=c, color=RED)
                self.play(ShowCreation(beam), run_time=1 / (i + 1))
                b = c

    def test_angled(self, rotation):
        for angle in [
            round(np.pi / 4, 3),
            round(3 * np.pi / 4, 3),
            round(5 * np.pi / 4, 3),
            round(7 * np.pi / 4, 3),
        ]:
            if round(rotation, 3) == angle:
                return True

    def element_here(self, loc):
        for el in self.elements:
            equal_positions = True
            for i, el_loc in enumerate(np.array(self.round_loc(el[0]))):
                if el_loc != loc[i]:
                    equal_positions = False
            if equal_positions:
                return True, el
        return False, None

    def snap_to_position(self, element: mobject, element_name, unsnapped_point, offset):
        """Takes in an object and point and snaps it to the closest integer point
        Returns:
            element: The snapped object
        """
        if self.snap:
            pos = Point(
                location=[
                    round(unsnapped_point.get_x()) + offset[0],
                    round(unsnapped_point.get_y()) + offset[1],
                    round(unsnapped_point.get_z()) + offset[2],
                ]
            )
        else:
            pos = self.mouse_point
        print(self.elements)
        # check if element is already in this position
        loc_available = True
        for el in self.elements:
            if self.round_loc(el[0]) == self.round_loc(pos):
                loc_available = False
                t = Text("Cannot place here")
                self.play(FadeIn(t))
                self.play(FadeOut(t))
        if loc_available:
            element.move_to(pos).rotate(self.rotation, np.array([0, 0, 1]))
            if element_name == "Laser":
                self.lasers.append(self.round_loc(pos))
            elif element_name == "Prism":
                self.prisms.append(self.round_loc(pos))
            elif element_name == "Thin Lens":
                self.thin_lenses.append(self.round_loc(pos))
            self.elements.append([element, self.rotation])
            return True
        else:
            del element
            return False

    def round_loc(self, loc):
        return [round(loc.get_x()), round(loc.get_y()), round(loc.get_z())]

    def on_key_press(self, symbol, modifiers):
        """Manim uses this method for accepting input, by writing it here the method is overridden

        Args:
            symbol (_type_): A normal char on a keyboard
            modifiers (_type_): Something like ctrl or enter
        """
        try:
            char = chr(symbol)
        except OverflowError:
            logger.warning("The value of the pressed key is too large.")
            return
        if char == "r":
            if self.rotation < 5.49:  # keep between 0 and 2pi
                self.rotation += np.pi / 4
            else:
                self.rotation = 0
            self.cursor.rotate(np.pi / 4, np.array([0, 0, 1]))
        elif char == "q":
            self.quit_interaction = True
        elif char == "a":
            self.add_element()
        elif char == "l":
            self.turn_on_laser()
        elif char == "s":
            # * Toggles snap
            if self.snap == False:
                self.snap = True
            else:
                self.snap = False
        elif char == "0":
            self.change_cursor(self.laser)
        elif char == "1":
            self.change_cursor(self.thin_lens)
        elif char == "2":
            self.change_cursor(self.prism)
        elif char == "3":
            self.change_cursor(self.flat_mirror)

    def change_cursor(self, new_element_method: classmethod):
        new_cursor, new_name = new_element_method(
            creation_method="none", opacity=self.cursor_opacity
        )
        new_name = Text(new_name).move_to(TOP + DOWN)
        always(new_cursor.move_to, self.mouse_point)
        self.add_element = new_element_method
        self.play(
            ReplacementTransform(self.cursor, new_cursor.rotate(self.rotation)),
            Write(new_name),
        )
        self.cursor = new_cursor
        self.wait(1)
        self.play(FadeOut(new_name))
