## Generate Hermite Gaussian Modes for Propagation
import numpy as np
from LightProp import LightSim


class HermiteGaussian(LightSim):
    def __init__(self):
        super().__init__()

    def hermiteH(self, n:int, x:np.ndarray):
        """Returns a Hermite Polynomial of order n

        Args:
            n (int): The order of the polynomial
            x (np.ndarray): The space that the polynomial is broadcast to

        Returns:
            np.ndarray: The Hermite Polynomial
        """
        if n == 0:
            return 1
        elif n == 1:
            return 2 * x
        else:
            return 2 * x * self.hermiteH(n - 1, x) - 2 * (n - 1) * self.hermiteH(
                n - 2, x
            )

    def makeHG(
        self, n:int, m:int, w0:float, xshift: float = 0, yshift: float = 0, z: float = 0
    ):
        """Makes a Hermite Gaussian Mode with n Horizontal and m Vertical mode indices

        Args:
            n (int): Horizontal mode index
            m (int): Vertical mode index
            w0 (float): Beam waist at z = 0
            xshift (float, optional): X distance from center of simulated area. Defaults to 0.
            yshift (float, optional): Y distance from center of simulated area. Defaults to 0.
            z (float, optional): The distance in the axial direction to propagate. Defaults to 0.

        Returns:
            np.ndarray: A single HGnm mode
        """
        rayleigh_length = self.Rayleigh_Length(w0)
        if z != 0:
            Rz_inv = 1 / (z * (1 + (rayleigh_length / z) ** 2))
        else:
            Rz_inv = 0
        wz = w0 * np.sqrt(1 + (z / rayleigh_length) ** 2)
        U00 = (
            1.0
            / (1 + 1j * z / rayleigh_length)
            * np.exp(
                -((self.X - xshift) ** 2 + (self.Y - yshift) ** 2) / wz**2 / (1 + 1j * z / rayleigh_length)
            )
            * np.exp(-1j * self.k * ((self.X - xshift) ** 2 + (self.Y - yshift) ** 2) * Rz_inv / 2)
        )
        Hn = self.hermiteH(n, (self.X - xshift) / wz)
        Hm = self.hermiteH(m, (self.Y - yshift) / wz)
        return U00 * Hn * Hm * np.exp(-1j * (n + m) * np.arctan(z / rayleigh_length))


# Demonstration of Hermite Gaussian
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    InitialBeamWaist = 10*40e-6
    HG = HermiteGaussian()
    U = HG.makeHG(0, 5, InitialBeamWaist, 0)
    plt.figure()
    plt.title("Intensity")
    plt.pcolor(abs(U) ** 2)
    plt.axis("equal")

    plt.figure()
    plt.title("Phase")
    plt.pcolor(np.angle(U) ** 2)
    plt.axis("equal")

    plt.show()
