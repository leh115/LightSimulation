import sys

import numpy as np
import time

from manimlib import *

sys.path.append("D:/Uni/Uni Masters/Project/code/LightSimulation/LightSim/RayTracing")
from ray_transfer_matrix_class import ray_transfer as rays
from lab_objects import lab_objects


class lab(Scene):
    def construct(self):
        """Manim uses a constuct method to generate a Scene"""

        for x in range(-15, 16):
            for y in range(-15, 16):
                self.add(Dot(np.array([x, y, 0]), color=GREY))

        self.add(
            NumberPlane(
                x_range=(-15, 15),
                y_range=(-15, 15),
                background_line_style={
                    "stroke_color": TEAL,
                    "stroke_width": 1,
                    "stroke_opacity": 0.3,
                },
            ).set_stroke(opacity=0.1)
        )
        self.last_method_name = ""
        self.debug = True
        self.labjects = lab_objects(self.mouse_point, self.debug)
        self.cursor_element = self.labjects.laser
        self.cursor_opacity = 0.3
        self.cursor, player, _ = self.cursor_element(
            self.mouse_point, opacity=self.cursor_opacity
        )
        self.play(player)
        always(self.cursor.move_to, self.mouse_point)
        self.labjects.elements = []

        self.rays = rays(self.debug)
        
        self.mouse_press_callbacks = [self.interactive_add]
        

    def turn_on_laser(self):
        """Triggers an animation to show the beam from any given laser"""
        self.debugger("Checking if laser should turn on", "Turn on laser")
        mp = self.mouse_point
        pos = self.labjects.round_loc(mp)

        laser_here = False
        for i, el in enumerate(self.labjects.elements):
            if self.labjects.round_loc(el[0]) == pos:
                laser_here = True
                self.debugger("Laser turning on", "Turn on laser")
                self.propagate_beam(pos, el[1])
        if not laser_here:
            t = Text("No laser found")
            self.play(FadeIn(t))
            self.wait()
            self.play(FadeOut(t))

    def propagate_beam(self, start, rotation, steps=20):
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
        element_interaction = None
        el_location = [0, 0]
        rad_angle = 0
        for i in range(steps):
            el_bool, el = self.labjects.element_here(b)
            if el_bool:
                rad_angle = rotation - el[1]
                rad_angle = np.abs(rad_angle)
                self.debugger(
                    f"Laser beam hit a {el[0].__name__.lower()} at a {180*(rad_angle)/np.pi} degree angle",
                    "Propagation method",
                    1,
                )
                element_interaction = el[0].__name__
                el_location = self.labjects.round_loc(el[0])
                break
            else:
                
                c_x_r = self.rays.free_space(
                    beam_matrix = np.array([b[0], np.sin(rotation+np.pi/2)]), distance=1
                )
                unit_x = c_x_r[0] - b[0]
                unit_y = np.sin( np.arcsin(c_x_r[1]) + np.sign(-(rotation - np.pi + 1e-4))*np.pi/2)
                
                new_x = b[0] + unit_x
                new_y = b[1] + unit_y

                self.debugger(f"Free space output: {c_x_r}", "Propagation method", 1)
                self.debugger(
                    f"x [{b[0]} --> {new_x}], y [{b[1]} --> {new_y}], theta [{rotation} --> {c_x_r[1]}]",
                    "Propagation method",
                    1,
                )
                c = [new_x, new_y, 0]
                beam = Line(start=b, end=c, color=RED)
                self.play(ShowCreation(beam), run_time=1 / (i + 1))
                b = c

        beam_x_mat, beam_y_mat = self.rays.loc_rot_2_mats(
            location=el_location, rotation=rotation
        )

        if element_interaction is None:
            pass
        elif element_interaction == "Flat Mirror":
            self.debugger(
                f"Calculating flat mirror interaction", "Propagation method", 1
            )
            x_rot = self.rays.flat_mirror(beam_x_mat)
            y_rot = self.rays.flat_mirror(beam_y_mat)
            self.debugger(x_rot[1] * np.pi, "Propagation method", 1)
            self.debugger(y_rot[1] * np.pi, "Propagation method", 1)
            c = np.round(np.add(b, [x_rot[1], y_rot[1], 0]))
            beam = Line(start=b, end=c, color=RED)
            self.play(ShowCreation(beam), run_time=1 / (i + 1))

        elif element_interaction == "Thin Lens":
            self.debugger(f"Calculating thin lens interaction", "Propagation method", 1)
            x_thin_lens = self.rays.thin_lens([el_location[0], rotation])

            self.debugger(
                f"Location: x = {el_location[0]}, y = {el_location[1]}  Rotation: {rotation}rad",
                "Propagation method",
                1,
            )
            if (rotation - np.pi) < 0:
                new_x = x_thin_lens[0]
            else:
                new_x = el_location[0] - np.abs(x_thin_lens[0] - el_location[0])
            new_y = el_location[1] + (x_thin_lens[0] - el_location[0]) * np.sin(
                x_thin_lens[1]
            )  #% np.pi)
            self.debugger(f"New x: {new_x}, New y: {new_y}", "Propagation method", 1)
            # [x_1, theta_1] = M[x_0, theta_0]
            # y_1 = y_0 + (x_1 - x_0)*theta_1
            print(x_thin_lens)
            c = [new_x, new_y, 0]
            beam = Line(start=b, end=c, color=RED)
            self.play(ShowCreation(beam), run_time=1 / (i + 1))

    #! this is outdated and needs to work for all angles
    def test_angled(self, rotation):
        """checks the angle of the element is not horizontal or vertical

        Args:
            rotation: The rotation of the element when it was placed

        Returns:
            Bool: True if angle is 45 degrees to an axis
        """
        for angle in [
            round(np.pi / 4, 3),
            round(3 * np.pi / 4, 3),
            round(5 * np.pi / 4, 3),
            round(7 * np.pi / 4, 3),
        ]:
            if round(rotation, 3) == angle:
                return True

    def on_mouse_press(self, point, button, modifiers):
        for func in self.mouse_press_callbacks:
            func()
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
        try: 
            mods = modifiers
        except Exception as e:
            print("Well that didn't work")
            print(e)
       
        if mods !=0:
            self.debugger(f"Modifier pressed: {mods}","key press")
        else:
            self.debugger(f"Character key pressed: {char}","key press")

        if char == "r":
            if self.labjects.rotation < 5.49:  # keep between 0 and 2pi
                self.labjects.rotation += np.pi / 4
            else:
                self.labjects.rotation = 0
            self.cursor.rotate(np.pi / 4, np.array([0, 0, 1]))
        elif char == "q":
            self.quit_interaction = True
        elif char == "a":
            self.interactive_add()
        elif char == "l":
            self.turn_on_laser()
        elif char == "s":
            # * Toggles snap
            if self.snap == False:
                self.snap = True
            else:
                self.snap = False
        elif char == "0":
            self.change_cursor(self.labjects.laser)
        elif char == "1":
            self.change_cursor(self.labjects.thin_lens)
        elif char == "2":
            self.change_cursor(self.labjects.prism)
        elif char == "3":
            self.change_cursor(self.labjects.flat_mirror)
        elif char == "w" and mods ==2:
            print("Clearing all")
            self.clear_everything()
        
        

    def change_cursor(self, new_element_method: classmethod):
        new_cursor, player, _ = new_element_method(
            self.mouse_point,
            creation_method="none",
            opacity=self.cursor_opacity,
            is_cursor=True,
        )
        new_name = Text(new_cursor.__name__).move_to(TOP + DOWN)
        always(new_cursor.move_to, self.mouse_point)
        self.cursor_element = new_element_method
        self.play(
            ReplacementTransform(
                self.cursor, new_cursor.rotate(self.labjects.rotation)
            ),
            Write(new_name),
        )
        self.cursor = new_cursor
        self.wait(1)
        self.play(FadeOut(new_name))
    
    def interactive_add(self):
        print(" ")
        element, player, create_bool = self.cursor_element(self.mouse_point)
        if create_bool:
            self.play(player)
        else:
            t = Text("Cannot place here")
            self.play(FadeIn(t))
            self.play(FadeOut(t))
    
    def clear_everything(self):
        fade_outs = []
        for i, element in enumerate(self.labjects.elements):
            fade_outs.append(FadeOut(element[0]))
            del element
        self.play(*fade_outs)
        self.labjects.elements = []

    def debugger(self, debug_str: str, method_name: str = "", method_int=0):

        if self.debug:
            if self.last_method_name is not method_name:
                print("")
            print(
                str(" " * method_int * 4) + "~" + method_name + " ... " + str(debug_str)
            )
            self.last_method_name = method_name
