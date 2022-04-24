from LightProp import LightSim
from matplotlib import pyplot as plt
from Propagate import Propagate
from Beam_Analysis import Beam_Analyser
from Visualiser import Visualiser
import numpy as np
import cv2

class Updater(LightSim):
    def __init__(self, PlaneSetUp, modeNum):
        super().__init__(PlaneSetUp, modeNum)
        self.modeNum = modeNum
        self.updated_planes = [None] * (len(self.Planes))
        self.analyser = Beam_Analyser(PlaneSetUp, modeNum)
        self.show_modes_at_start = False

    def UpdatePhasePlane(
        self,
        F: np.ndarray,
        B: np.ndarray,
        Plane: np.ndarray,
        plane_number: int,
        push_to_class_var: bool = False,
    ):
        """Updates a single phase plane by comparing backwards and forwards propagation

        Args:
            F (np.ndarray): The cross section before the phase plane
            B (np.ndarray): The cross section after the phase plane
            Plane (np.ndarray): The phase plane to update
            plane_number (int): The index of the phase plane to update
        """

        # * Really this method should be within the <LightSim> class because it is changing a class variable of <LightSim> NOT <Updater>, however this lets me separate out the process of updating into its own neat class and tidies up the <LightSim> class a lot
        if not push_to_class_var:
            Current_Mask = np.exp(1j * np.angle(Plane))
            Mask2 = F * B.conj()
            Normaliser = np.sqrt(np.sum(np.abs(F) ** 2) * np.sum(np.abs(B.conj()) ** 2))
            Mask2 /= Normaliser

            if not isinstance(self.updated_planes, np.ndarray):
                self.updated_planes[plane_number] = (
                    Mask2 * np.exp(-1j * np.angle(np.sum(Mask2 * Current_Mask.conj())))
                    + self.maskOffset
                )
            else:
                self.updated_planes[plane_number] = np.add(
                    self.updated_planes[plane_number],
                    (
                        Mask2
                        * np.exp(-1j * np.angle(np.sum(Mask2 * Current_Mask.conj())))
                        + self.maskOffset
                    ),
                )

        else:
            for plane_num in range(len(self.Planes)):
                self.Planes[plane_num] = self.updated_planes[plane_num]

            Ls = LightSim(self.PlaneSetUp, self.modeNum)
            Ls.update_plane(self.Planes)

    def Update_All_Planes(
        self, Input: np.ndarray, Output: np.ndarray, showPrePostPlane: bool = False
    ):
        """Propagates Backwards and Forwards through whole system once with all modes and updates the phase planes

        Args:
            Input (np.ndarray): The Input modes to the system
            Output (np.ndarray): The Output modes to the system
            showPrePostPlane (bool): Show fields immediately before and after plane. Defaults to False.
        """
        Complex_Difference = 0
        Coupling_Matrix = np.zeros((Input.shape[0]), dtype=np.float)
        modeCentreDifferences = []
        Forwards_propagator = Propagate(
            self.PlaneSetUp, self.modeNum, override_dz=True, show_beam=False
        )
        Backwards_propagator = Propagate(
            self.PlaneSetUp, self.modeNum, override_dz=True, show_beam=False
        )
        self.updated_planes = [None] * (len(self.Planes))
        for mode in range(Input.shape[0]):

            Forwards_propagator.Beam_Cross_Sections = Input[mode]
            Backwards_propagator.Beam_Cross_Sections = Output[mode]

            Forwards_propagator.Propagate_FromPlane_ToPlane(0, len(self.PlaneSetUp))

            Backwards_propagator.Propagate_FromPlane_ToPlane(
                0, len(self.PlaneSetUp), Forwards=False
            )

            # * compares the centre of mass of desired output modes (not propagated) and the actual output modes (propagated from start to finish)
            modeCentreDifferences.append(
                self.analyser.Centre_difference(
                    Backwards_propagator.Beam_Cross_Sections[0],
                    Forwards_propagator.Beam_Cross_Sections[-1],
                )
            )

            for i, F in enumerate(Forwards_propagator.Beam_Cross_Sections):
                # * Phase planes are at even indices. The first even index (0) is not a phase plane, its the initial beam
                if i > 0:
                    if i % 2 == 0:

                        plane_number = i // 2 - 1
                        B = Backwards_propagator.Beam_Cross_Sections[-i]

                        [
                            Coupling_Matrix[mode],
                            Complex_Difference,
                        ] = self.analyser.coupling_analysis(
                            F, B, Coupling_Matrix[mode], Complex_Difference
                        )

                        if showPrePostPlane:
                            cv2.imshow("Final Forwards", np.abs(F))
                            cv2.imshow("Final Backwards", np.abs(B))
                            cv2.waitKey(10)

                        self.UpdatePhasePlane(
                            F, B, self.Planes[plane_number], plane_number
                        )

        self.UpdatePhasePlane(
            F, B, np.ones((1, 1), dtype=bool), -1, push_to_class_var=True
        )

        self.analyser.coupleMat.append(np.sum(Coupling_Matrix))
        self.analyser.avgcoupler = np.sum(
            self.analyser.coupleMat / np.max(self.analyser.coupleMat)
        ) / len(self.analyser.coupleMat)

        self.analyser.save_current_complex_convergence(Complex_Difference)

        if showPrePostPlane:
            plt.show(block=False)
        self.analyser.save_Centre_Of_Mass_Convergence(modeCentreDifferences)

    def GradientDescent(
        self,
        Input: np.ndarray,
        Output: np.ndarray,
        EpochNumber: int,
        samplingRate: int,
        showAllModes: bool = False,
        show_Propagation_live: bool = True,
    ):
        """Attempts to find the global minima of the phase planes using a wavematching technique.

        Args:
            Input (np.ndarray): An array that propagates forwards to try and approximate the output modes
            Output (np.ndarray): An array that propagates backwards to try and approximate the input modes
            EpochNumber (int): The number of iterations to run for
            samplingRate (int): How often to display a visual of the propagation
            showAllModes (bool, optional): When visualising, should each mode be shown individually? Defaults to False.
            showAnyProgress (bool, optional): If False, nothing will be shown. Defaults to True.
        """
        visual = Visualiser(self.PlaneSetUp, self.modeNum, show_all_modes=showAllModes,show_Propagation_live=show_Propagation_live)
        
        if self.show_modes_at_start:
            visual.show_Initial(Input,Output)

        for GradDescent in range(EpochNumber):
            In = Input.copy()
            Out = Output.copy()

            self.Update_All_Planes(
                In, Out, showPrePostPlane=False
            )  # Correct for all of the planes
            print("Epoch: %d" % (GradDescent))

            if GradDescent % samplingRate == 0:
                visual.showProgress(In, Out)


if __name__ == "__main__":
    from MultiMode import ModePosition as mulmo

    PlaneSetUp = [20e-3, 25e-3, 25e-3, 25e-3, 25e-3, 25e-3, 25e-3, 25e-3]
    Number_Of_Modes = 10
    InitialBeamWaist = 40e-6
    spotSeparation = np.sqrt(4) * InitialBeamWaist

    mode_maker = mulmo(PlaneSetUp, Number_Of_Modes, 1)
    modes = mode_maker.makeModes(InitialBeamWaist, spotSeparation, "Fib", "Spot")
    output_modes = np.zeros(
        (mode_maker.modeNum, 1, mode_maker.Nx, mode_maker.Ny), dtype=np.complex128
    )

    propagator = Propagate(PlaneSetUp, Number_Of_Modes, override_dz=True)
    for i, mode in enumerate(modes):
        propagator.Beam_Cross_Sections = mode[0]
        propagator >> np.sum(PlaneSetUp)
        output_modes[i][0] = propagator.Beam_Cross_Sections[-1]

    updater = Updater(PlaneSetUp, Number_Of_Modes)
    updater.GradientDescent(modes, output_modes, 100, samplingRate=10)
    # conclusion is that once the secondary starts updating itself it initialises its variables and stops checking main anymore: so the warning is to only use the main variable and make sure that secondary is actually updating it.
