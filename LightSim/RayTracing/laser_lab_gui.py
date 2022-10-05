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
        self.input_text = ""
        self.text_shift = False
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
        self.laser_steps = 3
        self.mouse_press_callbacks = [self.interactive_add]
        self.on_key_press = self.key_press
        

    def turn_on_laser(self):
        """Triggers an animation to show the beam from any given laser"""
        self.debugger("Checking if laser should turn on", "Turn on laser")
        mp = self.mouse_point
        pos = self.labjects.round_loc(mp)

        laser_here = False
        for _, el in enumerate(self.labjects.elements):
            if self.labjects.round_loc(el[0]) == pos:
                laser_here = True
                self.debugger("Laser turning on", "Turn on laser")
                self.propagate_beam(pos, el[1])
                break
        if not laser_here:
            t = Text("No laser found")
            self.play(FadeIn(t))
            self.wait()
            self.play(FadeOut(t))

    def propagate_beam(self, start, rotation):
        #b = start
        multiplier = 0.6
        b = np.add(
            start, [multiplier * np.cos(rotation), multiplier * np.sin(rotation), 0]
        )
        element_interaction = None
        el_location = [0, 0]
        rad_angle = 0
        for step in range(self.laser_steps):
            el_bool, el = self.labjects.element_here(b)
            print(step)
            if step>0 and el_bool and el[0].__name__!="Beam":
                rad_angle = np.abs(rotation - el[1])
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
                
                if unit_x!=0:
                    unit_y /= np.abs(unit_x)
                    unit_x /= np.abs(unit_x)

                new_x = b[0] + unit_x
                new_y = b[1] + unit_y
                if step == 0:
                    new_x = round(new_x)
                    new_y = round(new_y)

                self.debugger(f"Free space output: {c_x_r}", "Propagation method", 1)
                self.debugger(
                    f"x [{b[0]} --> {new_x}], y [{b[1]} --> {new_y}], theta [{rotation} --> {c_x_r[1]}]",
                    "Propagation method",
                    1,
                )
                c = [np.round(new_x, 1), np.round(new_y, 1), np.round(0, 1)]
                self.make_beam(b,c,step)
                b = c
        else:
            self.debugger(f"No interactions", "Propagation method", 1)

        beam_x_mat, beam_y_mat = self.rays.loc_rot_2_mats(
            location=el_location, rotation=rotation
        )
        

        if element_interaction == "Flat Mirror":
            self.debugger(
                f"Calculating flat mirror interaction", "Propagation method", 1
            )
            x_rot = self.rays.flat_mirror(beam_x_mat)
            y_rot = self.rays.flat_mirror(beam_y_mat)
            self.debugger(x_rot[1] * np.pi, "Propagation method", 1)
            self.debugger(y_rot[1] * np.pi, "Propagation method", 1)
            c = np.round(np.add(b, [x_rot[1], y_rot[1], 0]))

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
            ) 
            self.debugger(f"New x: {new_x}, New y: {new_y}", "Propagation method", 1)
            # [x_1, theta_1] = M[x_0, theta_0]
            # y_1 = y_0 + (x_1 - x_0)*theta_1
            print(x_thin_lens)
            c = [new_x, new_y, 0]
        self.make_beam(b, c, self.laser_steps)

    def make_beam(self, b, c, i):
        beam = Line(start=b, end=c, color=RED)
        beam.__name__ = "Beam"
        self.labjects += [beam,0]
        self.play(ShowCreation(beam), run_time= 1 / (i + 1))

    def on_mouse_press(self, point, button, modifiers):
        print(button)
        print(modifiers)
        if button == 1:
            for func in self.mouse_press_callbacks:
                func()
        elif button == 4:
            self.turn_on_laser()
    
    def on_mouse_scroll(self, point, offset):
        if offset[1]<0:
            self.rotate_cursor()
            return
        self.rotate_cursor(direction=-1)
        

    def type_input(self, symbol, modifiers):
        try:
            char = chr(symbol)
        except Exception as e:
            print(e)
        if char == "（":
            self.input_text = self.input_text[:-1]
            print(self.input_text)
            return
        elif char == "－":
            self.on_key_press = self.key_press
            t = Text(self.input_text).move_to(TOP).shift(DOWN)
            self.play(FadeIn(t))
            self.play(FadeOut(t))
            self.input_text = ""
            return
        elif char == "￡" or char == "￢":
            self.text_shift = True
            return
        if self.text_shift:
            self.input_text += char.upper()
            self.text_shift = False
        else:
            self.input_text += char
        print(self.input_text)

    def key_press(self, symbol, modifiers):
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
            print(e)
       
        if mods !=0:
            self.debugger(f"Modifier pressed: {mods}","key press")
        else:
            self.debugger(f"Character key pressed: {char}","key press")

        if char == "r":
            self.rotate_cursor()
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
        elif char == "w" and mods == 2:
            self.clear_everything()
        elif char == "=" and mods == 2:
            self.laser_steps += 1
            self.laser_distance()
        elif char == "-" and mods == 2:
            if self.laser_steps>1:
                self.laser_steps -= 1
                self.laser_distance()
        elif char == "t" and mods == 1:
            self.override_keyboard()
            print("- ------ - Text Mode - ------ -")
        
        

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

    def rotate_cursor(self, direction=1):
        self.labjects.rotation = (self.labjects.rotation + direction * np.pi / 4) % (2*np.pi)
        self.cursor.rotate(direction * np.pi / 4, np.array([0, 0, 1]))

    def interactive_add(self):
        print(" ")
        element, player, create_bool = self.cursor_element(self.mouse_point)
        if create_bool:
            self.play(player)
        else:
            t = Text("Cannot place here")
            self.play(FadeIn(t))
            self.play(FadeOut(t))
    def laser_distance(self):
        t = Text("Laser distance: " + str(self.laser_steps)).move_to(TOP).shift(DOWN)
        self.play(FadeIn(t))
        self.play(FadeOut(t))

    def clear_everything(self):
        print("Clearing all")
        fade_outs = []
        for i, element in enumerate(self.labjects.elements):
            fade_outs.append(FadeOut(element[0]))
            del element
        self.play(*fade_outs)
        self.labjects.elements = []

    def override_keyboard(self):
        self.on_key_press = self.type_input

    def debugger(self, debug_str: str, method_name: str = "", method_int=0):

        if self.debug:
            if self.last_method_name is not method_name:
                print("")
            print(
                str(" " * method_int * 4) + "~" + method_name + " ... " + str(debug_str)
            )
            self.last_method_name = method_name
