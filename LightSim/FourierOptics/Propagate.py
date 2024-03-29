import numpy as np
import cv2
import scipy.fftpack as sfft
from LightProp import LightSim

class Propagate(LightSim):
    def __init__(self, override_dz=False, show_beam=True):
        super().__init__()
        self.Beam_Cross_Sections = None
        self.override_dz = override_dz
        self.show_beam = show_beam

    #TODO make matrix handle modes too
    def PropagateFreeSpace(
        self,
        Distance=0.1,
        Forwards=True,
    ):
        """Propagates a beam through free space for a total distance D broken into increments of length dz

        Args:
            Distance (float): The total distance to travel in the z direction. Defaults to 0.1.
            showBeam (bool): Defaults to True.
            Forwards (bool): Denotes direction of propagation. Defaults to True.
        """
        if not isinstance(self[:], list):
            if len(self[:].shape) == 2:
                self[:] = [self[:]]

        if self.override_dz:
            dz = Distance
        else:
            dz = self.dz
        
        epsilon = 1e-5
        dzs = (np.ones((len(np.arange(dz, Distance + epsilon, dz)), self.Nx, self.Ny)) * np.expand_dims(np.arange(dz, Distance + epsilon, dz), (1,2)) )
        
        TransferFunction = np.exp(-1j * dzs * np.expand_dims(self.bowl, 0))

        if self.reverse_time:
            if not Forwards:
                TransferFunction = TransferFunction.conj()

        if self.filter_on:
            TransferFunction = TransferFunction * np.expand_dims(self.Filter, 0)

        # The fourier transform to get to k-space
        F = np.expand_dims(sfft.fftshift(sfft.fft2(self[-1])), 0)
        # Applying the propagation multiplication to the fourier space
        KF = TransferFunction * F  # applying the transfer function
        KF = np.fft.ifft2(sfft.fftshift(KF,axes=(-1,-2))) # kf is shifted temporally

        self.append(KF)
        if self.show_beam:
            cv2.imshow("Beam Cross section", np.abs(self[-1])**2)
            cv2.waitKey(50)
            cv2.imshow("Beam Phase", np.angle(self[-1]))
            cv2.waitKey(50)

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
        if not isinstance(self[:], list):
            if len(self[:].shape) == 2:
                self[:] = [self[:]]

        if Forwards:
            self.append([self[-1] * np.exp(1j * np.angle(Plane))])
        else:
            self.append([self[-1] * np.exp(-1j * np.angle(Plane))])

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

    def append(self, cross_sections):
        self[:] = np.append(
                self[:],
                cross_sections,
                axis=0,
            )
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

    def __or__(self, other: int or np.ndarray):
        """The or funciton | is not useful in the context of this class and so is replaced because it looks like a plane.

        Args:
            other (int): The index of the plane to travel through
        """
        if isinstance(other,int):
            self.PropagatePhasePlane(self.Planes[other])
        elif isinstance(other,np.ndarray):
            self.PropagatePhasePlane(other)

    def __ror__(self, other: int):
        """The ror funciton | is also not useful in the context of this class, ror defines when the class is on the opposite side of the operator (like rlshift above)

        Args:
            other (int): The index of the plane to travel through
        """
        if isinstance(other,int or np.ndarray):
            self.PropagatePhasePlane(self.Planes[other], Forwards=False)
        elif isinstance(other,np.ndarray):
            self.PropagatePhasePlane(other, Forwards=False)
    
    def __getitem__(self, i: int) -> np.ndarray:
        """When splicing a propagator object, the beam cross section is returned

        Args:
            i (int): the slice index

        Returns:
            np.ndarray: the slice at index i
        """
        return self.Beam_Cross_Sections[i]

    def __setitem__(self, i, other):
        if isinstance(i, slice):
            self.Beam_Cross_Sections = other
        if isinstance(i, int):
            self.Beam_Cross_Sections[i] = other

    def __str__(self) -> str:
        if len(self[:].shape) > 2:
            return f"<Propagate(Plane set up, Mode Number) object> with Beam Shape {self[-1].shape} and {len(self[:])} sections"
        else:
            return f"<Propagate(Plane set up, Mode Number) object> with Beam Shape {self[:].shape} and 1 section"


if __name__ == "__main__":
    from MultiMode import ModePosition as mulmo
    from Visualiser import Visualiser
    InitialBeamWaist = 40e-6
    spotSeparation = np.sqrt(4) * InitialBeamWaist
    propagator = Propagate(override_dz=False,show_beam=False)
    
    LightSim.number_of_modes = 20
    LightSim.reverse_time = True
    mode_maker = mulmo(Amplitude = 5)
    Modes = mode_maker.makeModes(
        InitialBeamWaist,
        0,
        "square",
        "hg"
    )
    
    # test_mode = Modes[3][0]
    # propagator.Beam_Cross_Sections = test_mode
    # print(propagator)
    # propagator.Propagate_FromPlane_ToPlane(0,len(propagator.PlaneSetUp))
    # print(propagator)
    # propagator.Propagate_FromPlane_ToPlane(0,4,Forwards=False)

    spiral = np.arctan2(propagator.X, propagator.Y) + np.pi

    cv2.imshow("Spiral",  spiral)
    cv2.waitKey(100)

    LightSim.dz /= 10
    m = 1
    cv2.imshow("Spirals",  np.real(np.exp(1j * m * spiral)))
    cv2.waitKey(100)
    
    spiral = np.exp(1j * m * spiral)
    z_dist = 0.03
    visual = Visualiser(save_to_file=False, show_Propagation_live=True)
    propagator[:] = Modes[1]
    print(propagator)
    rand_plane = np.random.rand(propagator.Nx,propagator.Ny) + 1J*np.random.rand(propagator.Nx,propagator.Ny)
    print(rand_plane)
    propagator >> 0.03
    propagator | rand_plane
    propagator >> 0.01
    propagator | rand_plane
    propagator >> 0.01

    visual.VisualiseBeam(np.abs(propagator[:])**2, "Testing speed update")
    
    propagator[:] = np.sum(Modes,axis=0)
    for mode in Modes:
        propagator[:] = mode
        LightSim.VERSION += 1
        #propagator.Beam_Cross_Sections = np.ones((propagator.Nx, propagator.Ny), dtype=np.complex128)
        propagator >> 0.05
        #propagator | spiral
        #propagator >> z_dist
        visual.VisualiseBeam(np.abs([propagator[-1]]), "transparent33", on_white="alpha")
    #z_dist << propagator
    #spiral | propagator
    #propagator | spiral
    #propagator >> 0.16
    # 0.16 << propagator
    # spiral.conj() | propagator
    # 0.16 << propagator
    # spiral.conj() | propagator
    # 0.01 << propagator

    