import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy.fftpack as sfft
import threading as TH
import multiprocessing as mp
from tqdm import tqdm

class LightSim:
    Planes = None
    VERSION = 1
    Variable_Name = ""
    ROOTDIR = "C:/Users/Unimatrix Zero/Documents/Uni Masters/Project/"
    kFilter = 1
    wavelength = 1565e-9
    resolution = 1
    k = 2 * np.pi / wavelength # Wavenumber of light
    dz = 5e-3
    ccd_size_factor = 2
    pixelSize = 8e-6 * ccd_size_factor
    Nx = int(2.192e-3 / pixelSize) * ccd_size_factor * resolution
    Ny = int(2.192e-3 / pixelSize) * ccd_size_factor * resolution
    number_of_modes = 5
    PlaneSetUp = [20e-3, 25e-3, 25e-3, 25e-3, 25e-3, 25e-3, 25e-3, 25e-3]
    phase_plane_components = [1, 0]
    filter_on = True
    reverse_time = True
    make_windows_top = True

    def __init__(self):
        
        if self.Planes == None:
            self.Planes = []
            for _ in range(len(self.PlaneSetUp) - 1):
                self.Planes.append(self.make_phase_plane(self.phase_plane_components[0], self.phase_plane_components[1]))

        self.x = (
            np.linspace(-self.Nx / 2, self.Nx / 2, self.Nx) * self.pixelSize
        )  # center around the coordinates (0,0)
        self.y = np.linspace(-self.Ny / 2, self.Ny / 2, self.Ny) * self.pixelSize
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.Theta, self.Rho = self.cart2pol(self.X, self.Y)
        self.maxRho = np.max(self.Rho)
        nx = self.X.shape[0]
        ny = self.X.shape[1]

        linearKx = (
            np.linspace(-nx / 2, nx / 2 - 1, nx)
            / nx
            * (nx / (np.max(self.X) - np.min(self.X)))
        )
        linearKy = (
            np.linspace(-ny / 2, ny / 2 - 1, ny)
            / ny
            * (ny / (np.max(self.Y) - np.min(self.Y)))
        )
        self.vx, self.vy = np.meshgrid(linearKx, linearKy)

        self.bowl = 0.7 * (2 * np.pi) * np.sqrt(
            (
                (1 / self.wavelength) ** 2
                - (self.vx**2 + self.vy**2) 
            )
        )#.T
        if self.filter_on:
            self.Filter = np.array(np.abs(self.Rho < self.kFilter*self.maxRho), dtype=np.float32)

    def __repr__(self) -> str:
        return f"<LightSim({self.PlaneSetUp}, {self.number_of_modes}) object>"

    def cart2pol(self, x, y):
        rho = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        return (theta, rho)

    def make_phase_plane(self, re:float = 1, im:float = 0):
        """Generates a complex valued, repeated value phase plane

        Args:
            re (float, optional): A single value for the real components of the phase plane. Defaults to 1.
            im (float, optional): A single value for the imaginary components of the phase plane. Defaults to 0.

        Returns:
            np.ndarray: The complex valued phase plane
        """
        real_components = re * np.ones((self.Nx, self.Ny), dtype=np.float)
        imaginary_components = im * np.ones((self.Nx, self.Ny), dtype=np.float)
        Plane = real_components + 1j * imaginary_components
        #Plane = np.ones((self.Nx, self.Ny), dtype=np.complex128)
        return Plane
    
    def Theoretical_BeamWaist(self, w0:float, z:float, rayleigh_length:float=-1,wavelength:float=-1):
        """Returns the beam waist at a given value of z

        Args:
            w0 (float): The initial beam waist
            z (float): The distance in the axial direction to take a measurement of the beam waist
            rayleigh_length (float, optional): The known Rayleigh length of the beam. Defaults to the inferred Rayleigh length from provided parameters.
            wavelength (float, optional): The wavelength of the beam. Defaults to the systems current wavelength.

        Returns:
            float: The expected beam waist
        """
        if wavelength == -1:
            wavelength = self.wavelength
        if rayleigh_length == -1:
            rayleigh_length = self.Rayleigh_Length(w0, wavelength)
        return w0 * np.sqrt(1 + (z / rayleigh_length) ** 2)
    
    def Rayleigh_Length(self,w0:float,wavelength:float = None):
        """Calculates the Rayleigh length for a given wavelength and input beam waist

        Args:
            w0 (float): Initial beam waist
            wavelength (float): The wavelength of light

        Returns:
            float: The Rayleigh length
        """
        if wavelength == None:
            wavelength = self.wavelength
        return np.pi * w0**2 / wavelength

if __name__ == "__main__":
    # For N planes there will be N+1 distances through free space e.g: ----|-------|-----|------------|----
    # Here are 5 propagation distances to calculate and only 4 planes    1      2     3         4        5

    # Boolean Switches
    validateBeamWaist = False

    ConvergenceResults = []
    CouplingResults = []
    Variable_Name = "Number of modes 2"
    Variable_Tests = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
    ] 
    for Var_Num, Variable in enumerate(Variable_Tests):
        VERSION = Var_Num
        
        Ls = LightSim()
        Ls.kFilter = 100
        Ls.Variable_Name = Variable_Name
        Ls.maskOffset *= 40
        print("Total distance: %.3fm" % sum(PlaneSetUp))
        print("Number of planes: %d" % len(Ls.Planes))
        RayLen = np.pi * InitialBeamWaist**2 / Ls.wavelength
        print(
            "Rayleigh length: %.4fm, Rayleigh Beam Width: %.6f"
            % (RayLen, np.sqrt(2) * InitialBeamWaist)
        )
        print(
            "Dimensions of image: (%.2fcm, %.2fcm)"
            % (Ls.Nx * Ls.pixelSize * 100, Ls.Ny * Ls.pixelSize * 100)
        )

        ConvergenceResults.append(Ls.avgFieldConv)
        plt.close()
        plt.scatter(Variable_Tests[: len(ConvergenceResults)], ConvergenceResults)
        plt.title("Average Field Phase Convergence")
        plt.xlabel(Variable_Name)
        plt.ylabel("Normalised Phase difference")
        plt.savefig(
            ROOTDIR + "/Results/Average Convergence Graph " + Variable_Name + ".png"
        )

        CouplingResults.append(Ls.avgcoupler)
        plt.close()
        plt.scatter(Variable_Tests[: len(CouplingResults)], CouplingResults)
        plt.title("Average Field Coupling")
        plt.xlabel(Variable_Name)
        plt.ylabel("Normalised Coupling")
        plt.savefig(
            ROOTDIR + "/Results/Average Coupling Graph " + Variable_Name + ".png"
        )
