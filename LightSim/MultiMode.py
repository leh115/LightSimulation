from array import array
import numpy as np
import HermiteGaussianClass as HG
import cv2
from LightProp import LightSim
from Propagate import Propagate


class ModePosition(LightSim):
    def __init__(self, PlaneSetUp, modeNum, Amplitude):
        super().__init__(PlaneSetUp, modeNum)
        self.HGs = HG.HermiteGaussian(PlaneSetUp, modeNum)
        self.Amplitude = Amplitude

    def NormaliseInitialCrossSection(self, X):
        X *= self.Amplitude / np.sqrt(np.max(np.abs(X) ** 2))
        return X

    def combinationsUptoN(self, n:int, modesOnly:bool=False) -> list:
        """Some really crude code to generate the initial nth combinations of [-N:N , -K:K] for two unknowns N and K

        Args:
            n (int): Number of combinations to calculate
            modesOnly (bool): If True then only positive combinations (which can be used to make HG modes) will be returned. Defaults to False.

        Returns:
            list: A nested list of the nth initial combinations
        """
        coord = 0
        modulo = 1
        a = -1
        b = 0
        combos = []
        combos1 = []
        for m in range(n):
            if coord == 0:
                if modulo % 2 == 0:
                    a -= 1
                else:
                    a += 1
            else:
                if modulo % 2 == 0:
                    b -= 1
                else:
                    b += 1
            if m % modulo == 0:
                if coord == 0:
                    coord = 1
                else:
                    coord = 0
                    modulo += 1
            if modesOnly:
                combos1.append([np.abs(a), np.abs(b)])
                combos = []
                [combos.append(x) for x in combos1 if x not in combos]
            else:
                combos.append([a, b])
        return combos

    def makeModes(
        self,
        w0: float,
        separation: float,
        pattern: str,
        modeType: str,
    ) -> np.array:
        """Sets up an array of shape (modes, 1 , Beam shape x, Beam shape y)

        Args:
            w0 (float): Initial beam waist
            separation (float): Distance (approx.) between modes in x,y
            pattern (str): Choose from:"Central" -> all modes in centre, "Square" -> a square pattern, "Fib" -> a Fibonacci spiral
            modeType (str): Choose from: "HG" -> Hermite Gaussian modes, increasing in value, "Spot"-> simple Gaussian modes
        Returns:
            np.array: an array containing all of the modes
        """

        Modes = np.zeros((self.modeNum, 1, self.Nx, self.Ny), dtype=np.complex128)

        if pattern == "Fib":
            theta = np.pi * 137.5077 / 180  # the golden angle
            A = separation * 2
            if modeType == "Spot":
                for m in range(self.modeNum):
                    Modes[m] = self.NormaliseInitialCrossSection(
                        np.array(
                            [
                                self.HGs.makeHG(
                                    0,
                                    0,
                                    w0,
                                    xshift=m**1.5 * A * np.cos(theta * m) / (m + 1),
                                    yshift=m**1.5 * A * np.sin(theta * m) / (m + 1),
                                )
                            ]
                        )
                    )
            if modeType == "HG":
                m_n_mode_values = self.combinationsUptoN(
                    self.modeNum + self.modeNum * 10, modesOnly=True
                )  # requires a safety net of extra modes
                for m in range(self.modeNum):
                    Modes[m] = self.NormaliseInitialCrossSection(
                        np.array(
                            [
                                self.HGs.makeHG(
                                    m_n_mode_values[m][0],
                                    m_n_mode_values[m][1],
                                    w0,
                                    xshift=m**1.5 * A * np.cos(theta * m) / (m + 1),
                                    yshift=m**1.5 * A * np.sin(theta * m) / (m + 1),
                                )
                            ]
                        )
                    )

        if pattern == "Central":

            if modeType == "Spot":
                for m in range(self.modeNum):
                    Modes[m] = self.NormaliseInitialCrossSection(
                        np.array([self.HGs.makeHG(0, 0, w0)])
                    )
            if modeType == "HG":
                m_n_mode_values = self.combinationsUptoN(
                    self.modeNum + self.modeNum * 10, modesOnly=True
                )  # requires a safety net of extra modes
                for m in range(self.modeNum):
                    Modes[m] = self.NormaliseInitialCrossSection(
                        np.array(
                            [
                                self.HGs.makeHG(
                                    m_n_mode_values[m][0],
                                    m_n_mode_values[m][1],
                                    w0,
                                )
                            ]
                        )
                    )
        if pattern == "Square":

            if modeType == "Spot":
                m_n_mode_values = self.combinationsUptoN(self.modeNum)
                for m in range(self.modeNum):
                    Modes[m] = self.NormaliseInitialCrossSection(
                        np.array(
                            [
                                self.HGs.makeHG(
                                    0,
                                    0,
                                    w0,
                                    xshift=m_n_mode_values[m][0] * separation,
                                    yshift=m_n_mode_values[m][1] * separation,
                                )
                            ]
                        )
                    )
            if modeType == "HG":
                m_n_position_values = self.combinationsUptoN(self.modeNum)
                m_n_mode_values = self.combinationsUptoN(self.modeNum + self.modeNum * 10, modesOnly=True)
                for m in range(self.modeNum):
                    Modes[m] = self.NormaliseInitialCrossSection(
                        np.array(
                            [
                                self.HGs.makeHG(
                                    m_n_mode_values[m][0],
                                    m_n_mode_values[m][1],
                                    w0,
                                    xshift=m_n_position_values[m][0] * separation,
                                    yshift=m_n_position_values[m][1] * separation,
                                )
                            ]
                        )
                    )

        return Modes
    
    def make_input_output_modes(self, w0:float, w1:float, separation0:float, separation1:float, pattern_to_pattern:str, mode_type_to_mode_type:str) -> list:
        """Parses a string to generate input and output modes

        Args:
            w0 (float): Input beam waist
            w1 (float): Output beam waist
            separation0 (float): Separation of input modes
            separation1 (float): Separation of output modes
            pattern_to_pattern (str): Of form "pattern -> pattern"
            mode_type_to_mode_type (str): Of form "mode_type -> mode_type"

        Returns:
            list: Of form [input_modes, output_modes]
        """
        input_pattern, output_pattern = pattern_to_pattern.split(" -> ")
        input_mode_type, output_mode_type = mode_type_to_mode_type.split(" -> ")
        modes = self.makeModes(w0, separation0, input_pattern, input_mode_type)
        propagator = Propagate(self.PlaneSetUp, self.modeNum, override_dz=True)
        output_modes = self.makeModes(w1, separation1, output_pattern, output_mode_type)
        for i, mode in enumerate(modes):
            propagator.Beam_Cross_Sections = mode[0]
            propagator >> np.sum(self.PlaneSetUp)
            output_modes[i][0] = propagator.Beam_Cross_Sections[-1]

        return [modes, output_modes]


if __name__ == "__main__":
    PlaneSetUp = [20e-3, 25e-3, 25e-3, 25e-3, 25e-3, 25e-3, 25e-3, 25e-3]
    Number_Of_Modes = 6
    Mds = ModePosition(PlaneSetUp, Number_Of_Modes, 5)
    Mds.make_input_output_modes(1,1,1,1,"Fib -> Square","Spot -> HG")
    M = Mds.makeModes(60e-6, np.sqrt(3) * 60e-6, "Square", "Spot")
    cv2.imshow("Modes", np.sum(np.abs(M), axis=0)[0])
    # cv2.imwrite("C:/Users/Unimatrix Zero/Documents/Uni Masters/Project/Figures and Demos/Mode Positions.png",255*np.sum(np.abs(M),axis=0)[0])
    cv2.waitKey(0)
    print(Mds.combinationsUptoN(5, modesOnly=True))
