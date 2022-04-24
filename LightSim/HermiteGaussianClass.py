## Generate Hermite Gaussian Modes for Propagation
import numpy as np
from LightProp import LightSim


class HermiteGaussian(LightSim):
    def __init__(self, PlaneSetUp, modeNum):
        super().__init__(PlaneSetUp, modeNum)
        self.xDim = self.Nx * self.pixelSize
        self.yDim = self.Ny * self.pixelSize

    def hermiteH(self, n, x):
        if n == 0:
            return 1
        elif n == 1:
            return 2 * x
        else:
            return 2 * x * self.hermiteH(n - 1, x) - 2 * (n - 1) * self.hermiteH(
                n - 2, x
            )

    def makeHG(
        self, n, m, w0, xshift: float = 0, yshift: float = 0, z: float = 0
    ):
        k = 2 * np.pi / self.wavelength # Wavenumber of light

        zR = (k * w0**2.0) / 2
        if z != 0:
            Rz_inv = 1 / (z * (1 + (zR / z) ** 2))
        else:
            Rz_inv = 0
        wz = w0 * np.sqrt(1 + (z / zR) ** 2)
        # ! I think this should be self.xDim/2 and self.yDim/2, needs implementing
        [xx, yy] = np.meshgrid(
            np.linspace(-self.xDim/2, self.xDim/2, self.Nx),
            np.linspace(-self.yDim/2, self.yDim/2, self.Ny),
        )
        U00 = (
            1.0
            / (1 + 1j * z / zR)
            * np.exp(
                -((xx - xshift) ** 2 + (yy - yshift) ** 2) / wz**2 / (1 + 1j * z / zR)
            )
            * np.exp(-1j * k * ((xx - xshift) ** 2 + (yy - yshift) ** 2) * Rz_inv / 2)
        )
        Hn = self.hermiteH(n, (xx - xshift) / wz)
        Hm = self.hermiteH(m, (yy - yshift) / wz)
        return U00 * Hn * Hm * np.exp(-1j * (n + m) * np.arctan(z / zR))


# Demonstration of Hermite Gaussian
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    PlaneSetUp = [20e-3, 25e-3, 25e-3, 25e-3, 25e-3, 25e-3, 25e-3, 25e-3]
    Number_Of_Modes = 6
    InitialBeamWaist = 40e-6
    HG = HermiteGaussian(PlaneSetUp, Number_Of_Modes)
    U = HG.makeHG(1, 0, 700e-9, InitialBeamWaist, 0)
    plt.figure()
    plt.title("Intensity")
    plt.pcolor(abs(U) ** 2)
    plt.axis("equal")

    plt.figure()
    plt.title("Phase")
    plt.pcolor(np.angle(U) ** 2)
    plt.axis("equal")

    plt.show()
