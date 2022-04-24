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
    ROOTDIR = "C:/Users/Unimatrix Zero/Documents/Uni Masters/Project/"

    def __init__(self, PlaneSetUp, modeNum):
        self.modeNum = modeNum
        self.PlaneSetUp = PlaneSetUp
        self.pixelSize = (
            8e-6  # JC chooses 8e-6 for a width of 274pixels so plane width is 2.192e-3m
        )

        self.Nx = int(2.192e-3 / self.pixelSize)
        self.Ny = int(2.192e-3 / self.pixelSize)

        self.dz = 5e-3  # speed of propagation through space
        self.wavelength = 1565e-9  # choose values between 380nm and 750nm
        self.kFilter = 0.1
        self.maskOffset = np.sqrt(1e-4 / (self.Nx * self.Ny * self.modeNum))
        self.Variable_Name = ""
        
        if self.Planes == None:
            self.Planes = []
            for _ in range(len(PlaneSetUp) - 1):
                self.Planes.append(self.MakePhasePlane())

        

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

        #self.bowl = (2 * np.pi) * np.sqrt(  ( ( 1/self.wavelength )**2 - self.vx**2 - self.vy**2 )).T
        self.bowl = 2 * (2 * np.pi) * np.sqrt(
            (
                (1 / self.wavelength) ** 2
                - (self.vx**2 + self.vy**2) / (2 * np.pi) ** 2
            )
        ).T

    def __repr__(self) -> str:
        return f"<LightSim({self.PlaneSetUp}, {self.modeNum}) object>"

    def cart2pol(self, x, y):
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return (rho, phi)

    @classmethod
    def update_plane(cls, new_planes):
        cls.Planes = new_planes

    def MakePhasePlane(self):
        planeSize = (self.Nx, self.Ny)
        Plane = np.ones(planeSize, dtype=np.complex128)
        return Plane

if __name__ == "__main__":
    # For N planes there will be N+1 distances through free space e.g: ----|-------|-----|------------|----
    # Here are 5 propagation distances to calculate and only 4 planes    1      2     3         4        5

    # Boolean Switches
    show_Initial_States = False
    Show_FWHM = False
    validateBeamWaist = False
    NormaliseInputs = True

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
    ]  # np.linspace(1e-6,300e-6,2000)
    for Var_Num, Variable in enumerate(Variable_Tests):
        VERSION = Var_Num
        # The set-up chosen by Joel Carpenter is a set of seven planes with a distance of 20mm before the first plane and 12.5mm*2 between each mirror
        PlaneSetUp = [
            20e-3,
            25e-3,
            25e-3,
            25e-3,
            25e-3,
            25e-3,
            25e-3,
            25e-3,
        ]  # JC plane set-up
        Number_Of_Modes = Variable  # 12
        InitialBeamWaist = 40e-6  # Joel Carpenter chooses w0 = 30e-6 and an exit beam of wz = 200e-6            #### I tested on 35e-6 for this
        spotSeparation = (
            np.sqrt(4) * InitialBeamWaist
        )  # Joel Carpenter chooses separation of 89.8e-6

        # The question becomes, how do we get an exit beam of wz = 200e-6 ???
        # My assumption is that we have to give it a different input beam waist so that it propagates to a smaller beam...
        # I found that the closest to 200e-6 I could get from some simple envelope calcs was 400e-6 using an initial beam waist larger than my other one??!!
        # Exit_InitialBeamWaist = 3e-4
        Exit_InitialBeamWaist = (
            spotSeparation * np.sqrt(Number_Of_Modes) / 2
        )  # I want the beam to interact with every part of the spot array, the array has a diameter of approx. one side length which is (separation dist) * sqrt(number of modes)

        Ls = LightSim(PlaneSetUp, Number_Of_Modes)
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

        if validateBeamWaist:
            for i in tqdm(range(50)):
                z = (i + 1) * Ls.dz
                X = Ls.PropagateFreeSpace(
                    np.array([HGs.makeHG(0, 0, Ls.wavelength, InitialBeamWaist)]),
                    Dist=z,
                    showBeam=False,
                    overridedz=True,
                    newdz=z,
                )[-1]
                fwhm, wz_fwhm = Ls.FWHM(np.abs(X), 0, z)
                plt.scatter(z, wz_fwhm, color="red")
                plt.scatter(
                    z,
                    Ls.Theoretical_BeamWaist(InitialBeamWaist, z, RayLen),
                    color="blue",
                )
            plt.title("Beam waist at distance z")
            plt.xlabel("Distance z (m)")
            plt.ylabel("Beam Waist (m)")
            plt.show()

        # Initialise Forward and Backward Propagating Fields:
        if show_Initial_States:
            BProp2 = np.zeros((Number_Of_Modes, 1, Ls.Nx, Ls.Ny), dtype=np.complex128)
            FProp2 = np.zeros((Number_Of_Modes, 1, Ls.Nx, Ls.Ny), dtype=np.complex128)
        FProp = Mds.makeModes(
            Number_Of_Modes,
            Ls.wavelength,
            InitialBeamWaist,
            spotSeparation,
            "Square",
            "Spot",
        )
        BProp = Mds.makeModes(
            Number_Of_Modes, Ls.wavelength, Exit_InitialBeamWaist, 0, "Central", "HG"
        )

        # Propagate to end for Backwards propagation
        for m in range(Number_Of_Modes):
            BProp[m] = Ls.PropagateFreeSpace(
                [BProp[m][0]],
                np.sum(PlaneSetUp),
                showBeam=False,
                overridedz=True,
                newdz=np.sum(PlaneSetUp),
            )[
                -1
            ]  # propagates to the output so that I can work backwards from there
            if show_Initial_States:
                FProp2[m] = Ls.PropagateFreeSpace(
                    [FProp[m][0]],
                    np.sum(PlaneSetUp),
                    showBeam=False,
                    overridedz=True,
                    newdz=np.sum(PlaneSetUp),
                )[-1]
                BProp2[m] = Ls.PropagateFreeSpace(
                    [BProp[m][0]],
                    np.sum(PlaneSetUp),
                    Forwards=False,
                    showBeam=False,
                    overridedz=True,
                    newdz=np.sum(PlaneSetUp),
                )[-1]

        if show_Initial_States:
            cv2.imshow("FProp", np.abs(np.sum(FProp**2, axis=0))[0])
            cv2.imshow("FProp2", np.abs(np.sum(FProp2**2, axis=0))[0])

            cv2.imshow("BProp", np.sum(np.abs(BProp) ** 2, axis=0)[0])
            cv2.imshow("BProp2", np.sum(np.abs(BProp2) ** 2, axis=0)[0])
            cv2.waitKey(0)

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
