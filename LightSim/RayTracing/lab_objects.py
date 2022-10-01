from manimlib import *
import numpy as np


class lab_objects:
    def __init__(self, mouse_point, debug):
        self.mouse_point = mouse_point
        self.snap = True
        self.rotation = 0
        self.elements = []
        self.lasers = []
        self.thin_lenses = []
        self.prisms = []
        self.debug = debug
        self.last_method_name = ""

    def thin_lens(
        self,
        mouse_point,
        opacity: float = 0.6,
        creation_method: str = "show creation",
        is_cursor=False,
    ):
        self.mouse_point = mouse_point
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
            is_cursor=is_cursor,
        )

    def prism(
        self,
        mouse_point,
        opacity: float = 0.6,
        creation_method: str = "show creation",
        is_cursor=False,
    ):
        self.mouse_point = mouse_point
        return self.create_element(
            Polygon(
                *[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                color=BLUE,
                fill_opacity=opacity,
                stroke_opacity=opacity,
            ),
            "Prism",
            creation_method,
            is_cursor=is_cursor,
        )

    def laser(
        self,
        mouse_point,
        opacity: float = 1,
        creation_method: str = "show creation",
        is_cursor=False,
    ):
        self.mouse_point = mouse_point
        return self.create_element(
            Polygon(
                *[[0, 0.8, 0], [1, 0.8, 0], [1.1, 0.9, 0], [1, 1, 0], [0, 1, 0]],
                color=GREY,
                fill_color=GREY,
                fill_opacity=opacity,
                stroke_opacity=opacity,
            ),
            "Laser",
            creation_method,
            is_cursor=is_cursor,
        )

    def flat_mirror(
        self,
        mouse_point,
        opacity: float = 1,
        creation_method: str = "show creation",
        is_cursor=False,
    ):
        self.mouse_point = mouse_point
        return self.create_element(
            Rectangle(width=0.2, height=1, color=GREY, fill_opacity=opacity),
            "Flat Mirror",
            creation_method,
            is_cursor=is_cursor,
        )

    def create_element(
        self,
        element,
        element_name: str,
        creation_method: str,
        offset=[0, 0, 0],
        is_cursor=False,
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
        element.__name__ = element_name
        self.debugger(f"Creation method: {creation_method}", "Create element method", 1)

        creation = None
        create_bool = self.snap_to_position(
            element, self.mouse_point, offset, is_cursor=is_cursor
        )
        if creation_method == "none":
            self.debugger(f"Placement allowed: {create_bool}", "Create element method", 1)
            if create_bool:
                creation = element

        if creation_method == "show creation":
            self.debugger(f"Placement allowed: {create_bool}", "Create element method", 1)
            if create_bool:
                creation = ShowCreation(element)

        if creation_method == "fade in":
            self.debugger(f"Placement allowed: {create_bool}", "Create element method", 1)
            if create_bool:
                creation = FadeIn(element)
        self.debugger(f"Element to be created: {creation}", "Create element method", 1)
        return element, creation, create_bool

    def snap_to_position(
        self, element: mobject, unsnapped_point, offset, is_cursor=False
    ):
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
        self.debugger(
            f"Number of elements: {len(self.elements)}", "snap position method", 2
        )
        # check if element is already in this position
        loc_available = True
        for el in self.elements:
            self.debugger(
                f"Placed element's location: {self.round_loc(el[0])}",
                "snap position method",
                2,
            )
            self.debugger(
                f"Desired element's location: {self.round_loc(pos)}",
                "snap position method",
                2,
            )
            if self.round_loc(el[0]) == self.round_loc(pos):
                self.debugger("Location not available", "snap position method", 2)
                loc_available = False
                del element
                return False
        if not is_cursor:
            self.elements.append([element, self.rotation])
            if loc_available:
                element.move_to(pos).rotate(self.rotation, np.array([0, 0, 1]))
                if element.__name__ == "Laser":
                    self.lasers.append(self.round_loc(pos))
                elif element.__name__ == "Prism":
                    self.prisms.append(self.round_loc(pos))
                elif element.__name__ == "Thin Lens":
                    self.thin_lenses.append(self.round_loc(pos))
        return True

    def round_loc(self, loc):
        """Performs the snap

        Args:
            loc (Point): An unrounded point object

        Returns:
            Point: A rounded point object
        """
        return [round(loc.get_x()), round(loc.get_y()), round(loc.get_z())]

    def element_here(self, loc):
        for el in self.elements:
            equal_positions = True
            for i, el_loc in enumerate(np.array(self.round_loc(el[0]))):
                if el_loc != loc[i]:
                    equal_positions = False
            if equal_positions:
                return True, el
        return False, None

    def debugger(self, debug_str: str, method_name: str = "", method_int=0):
        if self.debug:
            if self.last_method_name is not method_name:
                print("")
            print(str(" " * method_int * 4) + "~" + method_name + " ... " + str(debug_str))
            self.last_method_name = method_name
