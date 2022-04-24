from turtle import update
import unittest
from Propagate import Propagate
from MultiMode import ModePosition as mulmo
from Beam_Analysis import Beam_Analyser
from UpdatePlanes import Updater
from Visualiser import Visualiser
import numpy as np
import cv2


class Test_Simulation(unittest.TestCase):
    PlaneSetUp = [20e-3, 25e-3, 25e-3, 25e-3, 25e-3, 25e-3, 25e-3, 25e-3]
    Number_Of_Modes = 6
    InitialBeamWaist = 40e-6
    spotSeparation = np.sqrt(4) * InitialBeamWaist

    # * Set up all of the class instances
    mode_maker = mulmo(PlaneSetUp, Number_Of_Modes, 5)
    propagator1 = Propagate(PlaneSetUp, Number_Of_Modes)
    propagator2 = Propagate(PlaneSetUp, Number_Of_Modes, override_dz=True)
    analyse = Beam_Analyser(PlaneSetUp, Number_Of_Modes)
    plane_update = Updater(PlaneSetUp,Number_Of_Modes)
    visual = Visualiser(PlaneSetUp,Number_Of_Modes,False)
    Modes = mode_maker.makeModes(
        InitialBeamWaist,
        spotSeparation,
        "Square",
        "Spot",
    )

    # * Choose a Mode to test
    test_mode1 = Modes[3][0]

    # * Choose a Plane to test
    test_plane1 = 0

    # Test 1
    def test_Forwards_Propagation(self):
        """Tests that forward propagation generates a larger beam in free space"""
        z_distance = 5 * 20e-3

        self.propagator1.Beam_Cross_Sections = self.test_mode1
        self.propagator1 >> z_distance
        self.assertGreater(
            self.analyse.FWHM(self.propagator1.Beam_Cross_Sections[-1])[1],
            self.analyse.FWHM(self.test_mode1)[1],
        )

        #* Now test that by overriding the dz value only 1 calculation is made (so there is only an initial (1) and final (1) cross section (1+1=2 sections)) and also that this produces roughly equivalent results to calculating a larger number of cross sections without the override
        self.propagator2.Beam_Cross_Sections = self.test_mode1
        self.propagator2 >> z_distance
        self.assertEqual(len(self.propagator2.Beam_Cross_Sections),2)
        self.assertAlmostEqual(np.sum(self.propagator1.Beam_Cross_Sections[-1]), np.sum(self.propagator2.Beam_Cross_Sections[-1]))

    # Test 2
    def test_reproducible_propagation(self):
        """Tests that forwards propagation followed by equal distance backwards propagation reproduces the same beam cross section in free space with a phase shift"""

        z_distance = 5 * 20e-3
        self.propagator1.Beam_Cross_Sections = self.test_mode1
        self.propagator1 >> z_distance
        z_distance << self.propagator1

        # * Test the beam width is equal
        self.assertAlmostEqual(
            self.analyse.FWHM(self.propagator1.Beam_Cross_Sections[-1])[1],
            self.analyse.FWHM(self.test_mode1)[1],
        )

        # * Test the real components are equal to 7 decimal places
        self.assertAlmostEqual(
            np.real(np.sum(self.propagator1.Beam_Cross_Sections[-1])),
            np.real(np.sum(self.test_mode1)),
        )

        # * Test that the imaginary component returns to its initial value (which should be 0 for modes generated by mulmo)
        self.assertAlmostEqual(
            np.imag(np.sum(self.propagator1.Beam_Cross_Sections[-1])),
            np.imag(np.sum(self.test_mode1)),
        )

    # Test 3
    def test_reproducible_planes(self):
        """Tests that forwards and backwards through a plane produces equivalent results"""

        self.propagator1.Beam_Cross_Sections = self.test_mode1
        self.propagator1 | self.test_plane1  # this is less clear than for >>, but this is forwards
        self.test_plane1 | self.propagator1  # and this is backwards

        # * Test that there are only 3 cross sections: initial, after going through plane, after coming back through plane
        self.assertEqual(len(self.propagator1.Beam_Cross_Sections), 3)

        # * Test the real components are equal to 7 decimal places
        self.assertAlmostEqual(
            np.real(np.sum(self.propagator1.Beam_Cross_Sections[-1])),
            np.real(np.sum(self.test_mode1)),
        )

        # * Test that the imaginary component returns to its initial value (which should be 0 for modes generated by mulmo)
        self.assertAlmostEqual(
            np.imag(np.sum(self.propagator1.Beam_Cross_Sections[-1])),
            np.imag(np.sum(self.test_mode1)),
        )

    # Test 4
    def test_reproducible_planes_and_propagation(self):
        """Tests that combining planes with propagation still works by going forwards and then backwards through the system"""
        
        for _ in range(2):
            self.propagator1.Beam_Cross_Sections = self.test_mode1
            self.propagator1.Propagate_FromPlane_ToPlane(0, len(self.PlaneSetUp))
            self.propagator1.Propagate_FromPlane_ToPlane(
                0, len(self.PlaneSetUp), Forwards=False
            )

            self.propagator2.Beam_Cross_Sections = self.test_mode1
            self.propagator2.Propagate_FromPlane_ToPlane(0, len(self.PlaneSetUp))

            # * Test that for override dz a system with n planes has 2n + 1 cross sections + initial cross section
            self.assertEqual(len(self.propagator2.Beam_Cross_Sections), 2 * (len(self.PlaneSetUp) - 1 ) + 2)

            self.propagator2.Beam_Cross_Sections = self.test_mode1
            self.propagator2.Propagate_FromPlane_ToPlane(0, len(self.PlaneSetUp),Forwards = False)

            # * Test that for override dz a system with n planes has 2n + 1 cross sections + initial cross section when going backwards
            self.assertEqual(len(self.propagator2.Beam_Cross_Sections), 2 * (len(self.PlaneSetUp) - 1 ) + 2)

            # * Test the real components are equal to 7 decimal places
            self.assertAlmostEqual(
                np.real(np.sum(self.propagator1.Beam_Cross_Sections[-1])),
                np.real(np.sum(self.test_mode1)),
            )

            # * Test that the imaginary component returns to its initial value (which should be 0 for modes generated by mulmo)
            self.assertAlmostEqual(
                np.imag(np.sum(self.propagator1.Beam_Cross_Sections[-1])),
                np.imag(np.sum(self.test_mode1)),
            )

            Noisy_Plane = np.zeros((self.propagator1.Nx, self.propagator1.Ny), dtype=np.complex128)
            Noisy_Plane.real = np.random.normal(0, 1, (self.propagator1.Nx, self.propagator1.Ny))
            Noisy_Plane.imag = np.random.normal(0, 1, (self.propagator1.Nx, self.propagator1.Ny))
            self.propagator1.Planes[self.test_plane1] = Noisy_Plane

    # Test 5
    def test_update_intermediate_value_theorem(self):
        """The value of the updated plane should lie somewhere between the values of the Forward and Backward Fields.
        Important note: changing maskOffset (commented out below) may cause test to fail, this is both a good thing (helps us test mask offset in system) and bad because too high mask offset will stop the value from being intermediate
        """
        z_distance = 20e-3
        self.propagator1.Beam_Cross_Sections = self.test_mode1
        InitialPlane = self.propagator1.Planes[self.test_plane1].copy()
        self.propagator1 >> z_distance
        F = self.propagator1.Beam_Cross_Sections.copy()
        self.propagator1 >> z_distance # go forward another dz so now at 2 * dz
        z_distance << self.propagator1 # and return so that now field is conjugate
        B = self.propagator1.Beam_Cross_Sections.copy()
        #self.plane_update.maskOffset = 0.0001
        self.plane_update.UpdatePhasePlane(F[-1], B[-1], self.propagator1.Planes[self.test_plane1], self.test_plane1)       
        self.plane_update.UpdatePhasePlane(None, None, None, None, push_to_class_var=True) # needs to be pushed to class because this is the only mode
        self.propagator1 = Propagate(self.PlaneSetUp, self.Number_Of_Modes) # must re-start the class for new class variables to take effect
        
        #* Test first that there has been an update
        updated_plane = self.propagator1.Planes[self.test_plane1]
        self.assertNotEqual( np.sum(np.real(updated_plane)), np.sum(np.real(InitialPlane)) )
        self.assertNotEqual( np.sum(np.imag(updated_plane)), np.sum(np.imag(InitialPlane)) )

        #* Find the more positive field
        if np.sum(np.real(F))>np.sum(np.real(B)):
            more_positive_real = np.sum(np.real(F))
            less_positive_real = np.sum(np.real(B))
        else:
            more_positive_real = np.sum(np.real(B))
            less_positive_real = np.sum(np.real(F))

        if np.sum(np.imag(F))>np.sum(np.imag(B)):
            more_positive_imag = np.sum(np.imag(F))
            less_positive_imag = np.sum(np.imag(B))
        else:
            more_positive_imag = np.sum(np.imag(B))
            less_positive_imag = np.sum(np.imag(F))
        
        #* Test now that the new plane is intermediate between the two fields in both real and imaginary components
        self.assertGreater(np.sum(np.real(updated_plane)) , less_positive_real)
        self.assertLess(np.sum(np.real(updated_plane)) , more_positive_real)

        self.assertGreater(np.sum(np.imag(updated_plane)) , less_positive_imag)
        self.assertLess(np.sum(np.imag(updated_plane)) , more_positive_imag)
        
    def test_state_normalisation(self):
        """The maximum value of all of the modes should be 1.
        """
        mode_maker = mulmo(self.PlaneSetUp, self.Number_Of_Modes, 1)
        modes = mode_maker.makeModes(self.InitialBeamWaist,self.spotSeparation,"Square","Spot")
        #* all following modes should be equal normalisation to the test normalisation
        test_Normalisation = np.max(np.abs(modes))
        for patterns in ["Square","Central","Fib"]:
            for mode_type in ["HG","Spot"]:
                modes = mode_maker.makeModes(self.InitialBeamWaist,self.spotSeparation, patterns, mode_type)
                self.assertEqual(np.max(np.abs(modes)),test_Normalisation, patterns + " " + mode_type)

    def test_FWHM(self):
        mode_maker = mulmo([0.1], 1, 1)
        Modes = mode_maker.makeModes(
        self.InitialBeamWaist,
        0,
        "Central",
        "Spot",
        )

        self.visual.show_Initial(Modes,self.Modes,500)
        
        fwhm0,w0 = self.analyse.FWHM(Modes)
        self.propagator2.Beam_Cross_Sections = Modes[0][0]
        self.propagator2 >> 10 * 20e-1
        fwhm1,w1 = self.analyse.FWHM(self.propagator2.Beam_Cross_Sections)
        self.assertAlmostEqual(w0,w1)