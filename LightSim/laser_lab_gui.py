import this
from manimlib import *
import numpy as np


class lab(Scene):
    def construct(self):
        """Manim uses a constuct method to generate a Scene"""
        for x in range(-15, 15):
            for y in range(-15, 15):
                self.add(Dot(np.array([x, y, 0]), color=GREY))
        self.lasers = []
        self.thin_lenses = []
        self.prisms = []
        self.elements = []
        self.snap = True
        self.rotation = 0
        self.add_element = self.laser
        self.cursor_opacity = 0.3
        self.cursor,_ = self.add_element(opacity=self.cursor_opacity)
        always(self.cursor.move_to, self.mouse_point)
        self.elements = []

    def thin_lens(self, opacity:float = 0.6, creation_method: str = "show creation"):
        
        return self.create_element(
            RoundedRectangle(
                corner_radius=0.05, height=1, width=0.1, color=BLUE,fill_opacity=opacity, stroke_opacity=opacity
            ),"Thin Lens",
            creation_method,
        )
        

    def prism(self, opacity:float = 0.6, creation_method: str = "show creation"):
        # return self.create_element(Triangle(width = 0.2, color=BLUE, opacity=1), creation_method)
        return self.create_element(
            Polygon(*[[0, 0, 0], [1, 0, 0], [0, 1, 0]], color=BLUE,fill_opacity=opacity, stroke_opacity=opacity),"Prism",
            creation_method,
            #offset=[0.175, 0.175, 0],
        )

    def laser(self, opacity:float = 1, creation_method: str = "show creation"):

        return self.create_element(
            Polygon(
                *[[0, 0.8, 0], [1, 0.8, 0],[1.1, 0.9, 0], [1, 1, 0], [0, 1, 0]], color=GREY,fill_color=GREY, fill_opacity=opacity, stroke_opacity=opacity
            ),"Laser",
            creation_method,
        )

    def create_element(self, element,element_name:str, creation_method:str, offset=[0, 0, 0]):
        if creation_method == "none":
            return element,element_name
        if creation_method == "show creation":
            create_bool = self.snap_to_position(element,element_name, self.mouse_point, offset)
            if create_bool:
                self.play(ShowCreation(element))
            return element,element_name
        if creation_method == "fade in":
            create_bool = self.snap_to_position(element,element_name, self.mouse_point, offset)
            if create_bool:
                self.play(FadeIn(element))
            return element,element_name
    
    def turn_on_laser(self):
        print("Checking if laser should turn on")
        mp = self.mouse_point
        pos = [round(mp.get_x()), round(mp.get_y()), round(mp.get_z())]
        
        laser_here = False
        for i, el in enumerate(self.elements):
            if el == pos:
                laser_here = True
                self.propagate_beam(np.add(pos,[0.6,0,0]))
        if not laser_here:
            t = Text("No laser found")
            self.play(FadeIn(t))
            self.wait()
            self.play(FadeOut(t))

            
    def propagate_beam(self,start):
        print(start)
        print(np.add(start,[1,0,0]))
        self.play(ShowCreation(Line(start = start,end = np.add(start,[1,0,0]))))

    def snap_to_position(self, element:mobject, element_name, unsnapped_point, offset):
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
        
        #check if element is already in this position
        loc_available = True
        for el in self.elements:
            if el == [pos.get_x(), pos.get_y(), pos.get_z()]:
                loc_available = False
        if loc_available:
            element.move_to(pos).rotate(self.rotation, np.array([0, 0, 1]))
            if element_name=="Laser":
                self.lasers.append([pos.get_x(),pos.get_y(),pos.get_z()])
            elif element_name=="Prism":
                self.prisms.append([pos.get_x(),pos.get_y(),pos.get_z()])
            elif element_name=="Thin Lens":
                self.thin_lenses.append([pos.get_x(),pos.get_y(),pos.get_z()])
            self.elements.append([pos.get_x(),pos.get_y(),pos.get_z()])
            return True
        else:
            del element
            return False

        

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
            self.rotation += np.pi / 4
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

    def change_cursor(self, new_element_method: classmethod):
        new_cursor,new_name = new_element_method(creation_method="none",opacity=self.cursor_opacity)
        new_name = Text(new_name).move_to(TOP+DOWN)
        always(new_cursor.move_to, self.mouse_point)
        self.add_element = new_element_method
        self.play(ReplacementTransform(self.cursor, new_cursor.rotate(self.rotation)), Write(new_name))
        self.cursor = new_cursor
        self.wait(1)
        self.play(FadeOut(new_name))
        
