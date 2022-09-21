from manimlib import *
import numpy as np

class lab(Scene):

    def construct(self):
        """Manim uses a constuct method to generate a Scene
        """
        for x in range(-10, 10):
            for y in range(-10, 10):
                self.add(Dot(np.array([x, y, 0]), color=GREY))
        self.add_object_type = 1
        self.snap = True
        self.rotation = 0
    
    def thin_lens(self):
        o = self.snap_to_position(RoundedRectangle(corner_radius=0.05,height=1, width=0.1, color=BLUE), self.mouse_point) 
        self.play(ShowCreation(o))
    
    def prism(self):
        pass

    def snap_to_position(self, mobject, unsnapped_point):
        """Takes in an object and point and snaps it to the closest integer point
        Returns:
            mobject: The snapped object
        """
        if self.snap:
            pos = Point(location = [round(unsnapped_point.get_x()),round(unsnapped_point.get_y()),round(unsnapped_point.get_z())])
        else:
            pos = self.mouse_point
        return mobject.move_to(pos).rotate(self.rotation, np.array([0,0,1]))

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
            #print(dir(self.camera))
            #self.camera.to_default_state()
            #self.camera_target = np.array([0, 0, 0], dtype=np.float32)
            self.rotation += np.pi/4
        elif char == "q":
            self.quit_interaction = True
        elif char == "a":
            if self.add_object_type == 1:
                self.thin_lens()
        elif char == "s":
            #* Toggles snap
            if self.snap == False:
                self.snap = True
            else:
                self.snap = False
            