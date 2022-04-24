import numpy as np
import cv2
import scipy.fftpack as sfft
from LightProp import LightSim


class Propagate(LightSim):
    def __init__(self, PlaneSetUp, modeNum, override_dz=False, show_beam=True):
        super().__init__(PlaneSetUp, modeNum)
        self.Beam_Cross_Sections = None
        self.override_dz = override_dz
        self.show_beam = show_beam

    def PropagateFreeSpace(
        self,
        Distance=0.1,
        Forwards=True,
        FWHM=False,
        printFWHM=False,
    ):
        """Propagates a beam through free space for a total distance D broken into increments of length dz

        Args:
            Distance (float): The total distance to travel in the z direction. Defaults to 0.1.
            showBeam (bool): Defaults to True.
            Forwards (bool): Denotes direction of propagation. Defaults to True.
            FWHM (bool): Shows the Full Width Half Maxima of every step along a beam. Defaults to False.
            printFWHM (bool): Defaults to False.

        Returns:
            _type_: _description_
        """
        if not isinstance(self.Beam_Cross_Sections, list):
            if len(self.Beam_Cross_Sections.shape) == 2:
                self.Beam_Cross_Sections = [self.Beam_Cross_Sections]

        if self.override_dz:
            dz = Distance
        else:
            dz = self.dz

        TransferFunction = np.exp(-1j * self.bowl * dz)

        if not Forwards:
            TransferFunction = TransferFunction.conj()

        # TransferFunction *= self.Rho<self.kFilter*self.maxRho
        num_Counts = abs(int(Distance / dz))
        for c in range(num_Counts):
            # The fourier transform to get to k-space
            F = sfft.fft2(self.Beam_Cross_Sections[-1])
            F = sfft.fftshift(F)

            # Applying the propagation multiplication to the fourier space
            KF = F * TransferFunction  # applying the transfer function

            # Inversing the result to get back to real space
            self.Beam_Cross_Sections = np.append(
                self.Beam_Cross_Sections, [np.fft.ifft2(sfft.fftshift(KF))], axis=0
            )

            if self.show_beam:
                cv2.imshow("Beam Cross section", np.abs(self.Beam_Cross_Sections[-1]))
                cv2.waitKey(1)
                cv2.imshow("Beam Phase", np.angle(self.Beam_Cross_Sections[-1]))
                cv2.waitKey(1)
            if FWHM:
                fwhm, w0 = self.FWHM(
                    np.abs(self.Beam_Cross_Sections[-1]), c, (c + 1) * dz
                )
                if printFWHM:
                    print(
                        "z = %.6f, Full Width Half Maxima: %.6f, Beam Waist: %.6f"
                        % ((c + 1) * dz, fwhm, w0)
                    )

    def PropagatePhasePlane(
        self, Plane: np.ndarray, Forwards: bool = True
    ) -> np.ndarray:
        """Applying the thin lens paraxial approximation to the beam.

        Args:
            Plane (np.ndarray): The plane it is passing through
            Forwards (bool,): The direction the beam is travelling in. Defaults to True.

        Returns:
            np.ndarray: The cross section of the beam after the plane
        """
        if not isinstance(self.Beam_Cross_Sections, list):
            if len(self.Beam_Cross_Sections.shape) == 2:
                self.Beam_Cross_Sections = [self.Beam_Cross_Sections]

        if Forwards:
            self.Beam_Cross_Sections = np.append(
                self.Beam_Cross_Sections,
                [self.Beam_Cross_Sections[-1] * np.exp(1j * np.angle(Plane))],
                axis=0,
            )
        else:
            self.Beam_Cross_Sections = np.append(
                self.Beam_Cross_Sections,
                [self.Beam_Cross_Sections[-1] * np.exp(-1j * np.angle(Plane))],
                axis=0,
            )

    # TODO Turn this into a dunder which parses a given plane set up: forwards propagate += [0.1,0.1,0.1] , backwards propagate -= [0.1,0.1,0.1]
    def Propagate_FromPlane_ToPlane(
        self,
        PlaneStart: int,
        PlaneEnd: int,
        Forwards: bool = True,
    ):
        """Propagates through multiple planes and free space distances at once

        Args:
            PlaneStart (int): The index of initial plane to start propagating from
            PlaneEnd (int): The index of final plane to propagate to
            Forwards (bool): Direction of propagation. Defaults to False.
        """

        if Forwards:
            for i, dists in enumerate(self.PlaneSetUp[PlaneStart:PlaneEnd]):
                self >> dists
                
                if i < len(self.PlaneSetUp[PlaneStart:PlaneEnd]) - 1:
                    self | i

        else:
            for i, dists in reversed(
                list(enumerate(self.PlaneSetUp[PlaneStart:PlaneEnd]))
            ):
                
                dists << self
                if i > 0:
                    i - 1 | self

    # * This following code makes everything look cool by adding "Class Operators" using dunder.
    # * For example: Forwards propagation now looks like >> and backwards looks like <<

    def __rshift__(self, other):
        """Replacement of the true bitwise function >> (which is not useful in this context) to show forwards propagation.

        Args:
            other (float): The distance to travel forwards in the axial direction
        """
        self.PropagateFreeSpace(other)

    def __rlshift__(self, other):
        """Replacement of the true bitwise function << (which is not useful in this context) to show backwards propagation.

        Args:
            other (float): The distance to travel backwards in the axial direction
        """
        self.PropagateFreeSpace(other, Forwards=False)

    def __or__(self, other: int):
        """The or funciton | is not useful in the context of this class and so is replaced because it looks like a plane.

        Args:
            other (int): The index of the plane to travel through
        """
        self.PropagatePhasePlane(self.Planes[other])

    def __ror__(self, other: int):
        """The ror funciton | is also not useful in the context of this class, ror defines when the class is on the opposite side of the operator (like rlshift above)

        Args:
            other (int): The index of the plane to travel through
        """
        self.PropagatePhasePlane(self.Planes[other], Forwards=False)

    def __str__(self) -> str:
        if len(self.Beam_Cross_Sections.shape) > 2:
            return f"<Propagate(Plane set up, Mode Number) object> with Beam Shape {self.Beam_Cross_Sections[-1].shape} and {len(self.Beam_Cross_Sections)} sections"
        else:
            return f"<Propagate(Plane set up, Mode Number) object> with Beam Shape {self.Beam_Cross_Sections.shape} and 1 section"


if __name__ == "__main__":
    import MultiMode as mulmo

    PlaneSetUp = [20e-3, 25e-3, 25e-3, 25e-3, 25e-3, 25e-3, 25e-3, 25e-3]
    Number_Of_Modes = 6
    InitialBeamWaist = 40e-6
    spotSeparation = np.sqrt(4) * InitialBeamWaist

    propagator = Propagate(PlaneSetUp, Number_Of_Modes,override_dz=True)
    

    mode_maker = mulmo.ModePosition(PlaneSetUp, Number_Of_Modes, 5)
    Modes = mode_maker.makeModes(
        InitialBeamWaist,
        spotSeparation,
        "Square",
        "Spot",
    )

    test_mode = Modes[3][0]
    Noisy_Plane = np.zeros((propagator.Nx, propagator.Ny), dtype=np.complex128)
    Noisy_Plane.real = np.random.normal(0, 1, (propagator.Nx, propagator.Ny))
    Noisy_Plane.imag = np.random.normal(0, 1, (propagator.Nx, propagator.Ny))
    print(Noisy_Plane)
    propagator.Beam_Cross_Sections = test_mode
    print(propagator)
    propagator.Propagate_FromPlane_ToPlane(0,len(PlaneSetUp))
    print(propagator)
    propagator.Propagate_FromPlane_ToPlane(0,4,Forwards=False)