import numpy as np
import HermiteGaussianClass as HG
import cv2
from LightProp import LightSim
from Propagate import Propagate


class ModePosition(LightSim):
    def __init__(self, Amplitude):
        super().__init__()
        self.HGs = HG.HermiteGaussian()
        self.Amplitude = Amplitude

    def NormaliseInitialCrossSection(self, X):
        X *= self.Amplitude / np.sqrt(np.max(np.abs(X) ** 2))
        return X

    def combinationsUptoN(self, n: int, modesOnly: bool = False) -> list:
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
        start_mode: int = 0,
    ) -> np.array:
        """Sets up an array of shape (modes, 1 , Beam shape x, Beam shape y)

        Args:
            w0 (float): Initial beam waist
            separation (float): Distance (approx.) between modes in x,y
            pattern (str): Choose from:"central" -> all modes in centre, "square" -> a square pattern, "fib" -> a Fibonacci spiral
            modeType (str): Choose from: "hg" -> Hermite Gaussian modes, increasing in value, "spot"-> simple Gaussian modes
            start_mode (int): The first mode to start on (useful for choosing a subset of modes)
        Returns:
            np.array: an array containing all of the modes
        """
        pattern = pattern.upper()
        modeType = modeType.upper()
        assert (
            modeType == "SPOT" or modeType == "HG"
        ), f"{modeType} is not a valid type of mode"
        assert (
            pattern == "FIB"
            or pattern == "CENTRAL"
            or pattern == "SQUARE"
            or pattern == "INV_SQUARE"
            or pattern == "RIGHT_LEFT"
            or pattern == "LEFT_RIGHT"
            or pattern == "UP_DOWN"
            or pattern == "DOWN_UP"
        ), f"{pattern} is not a valid type of pattern"

        Modes = np.zeros(
            (self.number_of_modes, 1, self.Ny, self.Nx), dtype=np.complex128
        )
        if modeType == "SPOT":
            mode_values = np.zeros((self.number_of_modes, 2))

        elif modeType == "HG":
            mode_values = self.combinationsUptoN(
                self.number_of_modes + self.number_of_modes * 10, modesOnly=True
            )  # requires a safety net of extra modes

        position_values = np.zeros((self.number_of_modes, 2))
        line_vals = (
            np.arange(start=0, stop=self.number_of_modes) - self.number_of_modes / 2
        ) * separation
        if pattern == "FIB":
            theta = np.pi * 137.5077 / 180  # the golden angle
            A = separation * 2
            ints_to_n = np.arange(start=0, stop=self.number_of_modes)
            xs = ints_to_n**1.5 * A * np.cos(theta * ints_to_n) / (ints_to_n + 1)
            ys = ints_to_n**1.5 * A * np.sin(theta * ints_to_n) / (ints_to_n + 1)
            for i, x, y in zip(ints_to_n, xs, ys):
                position_values[i] = [x, y]

        elif pattern == "CENTRAL":
            pass

        elif pattern == "SQUARE":
            position_values = (
                np.array(self.combinationsUptoN(self.number_of_modes)) * separation
            )

        elif pattern == "INV_SQUARE":
            xs = np.array(self.combinationsUptoN(self.number_of_modes)) * separation
            for i, x in enumerate(xs):
                position_values[-(i + 1)] = x

        elif pattern == "LEFT_RIGHT":
            for i, x in enumerate(line_vals):
                position_values[i] = [x, 0]

        elif pattern == "RIGHT_LEFT":
            for i, x in enumerate(line_vals):
                position_values[-(i + 1)] = [x, 0]

        elif pattern == "UP_DOWN":
            for i, y in enumerate(line_vals):
                position_values[i] = [0, y]

        elif pattern == "DOWN_UP":
            for i, y in enumerate(line_vals):
                position_values[-(i + 1)] = [0, y]

        for m in range(start_mode, self.number_of_modes):
            Modes[m] = self.NormaliseInitialCrossSection(
                np.array(
                    [
                        self.HGs.makeHG(
                            mode_values[m][0],
                            mode_values[m][1],
                            w0,
                            xshift=position_values[m][0],
                            yshift=position_values[m][1],
                        )
                    ]
                )
            )
        LightSim.number_of_modes -= start_mode
        return Modes[start_mode:]

    def make_input_output_modes(
        self,
        w0: float,
        w1: float,
        separation0: float,
        separation1: float,
        pattern_to_pattern: str,
        mode_type_to_mode_type: str,
        start_mode: int = 0,
    ) -> list:
        """Parses a string to generate input and output modes

        Args:
            w0 (float): Input beam waist
            w1 (float): Output beam waist
            separation0 (float): Separation of input modes
            separation1 (float): Separation of output modes
            pattern_to_pattern (str): Of form "pattern -> pattern"
            mode_type_to_mode_type (str): Of form "mode_type -> mode_type"
            start_mode (int): The first mode to start on (useful for choosing a subset of modes)
        Returns:
            list: Of form [input_modes, output_modes]
        """
        input_pattern, output_pattern = pattern_to_pattern.split(" -> ")
        input_mode_type, output_mode_type = mode_type_to_mode_type.split(" -> ")
        modes = self.makeModes(
            w0, separation0, input_pattern, input_mode_type, start_mode=start_mode
        )
        propagator = Propagate(override_dz=True, show_beam=False)
        LightSim.number_of_modes = self.number_of_modes + start_mode
        output_modes = self.makeModes(
            w1, separation1, output_pattern, output_mode_type, start_mode=start_mode
        )
        for i, mode in enumerate(modes):
            propagator.Beam_Cross_Sections = mode[0]
            propagator >> np.sum(propagator.PlaneSetUp)
            output_modes[i][0] = propagator.Beam_Cross_Sections[-1]

        return [modes, output_modes]


if __name__ == "__main__":
    LightSim.number_of_modes = 30
    mode_maker = ModePosition(Amplitude=1)
    # mode_maker.make_input_output_modes(1,1,1,1,"Fib -> Square","Spot -> HG")
    mode_maker.make_input_output_modes(
        30e-6,
        30e-6,
        120e-6,
        120e-6,
        "left_right -> left_right",
        "spot -> spot",
        start_mode=1,
    )
    for i in range(mode_maker.number_of_modes):
        LightSim.number_of_modes = 30
        M = mode_maker.makeModes(
            20e-6, np.sqrt(3) * 60e-6, "inv_square", "HG", start_mode=i
        )

        cv2.imshow("Modes", np.sum(np.abs(M), axis=0)[0])
        # cv2.imwrite("C:/Users/Unimatrix Zero/Documents/Uni Masters/Project/Figures and Demos/Mode Positions.png",255*np.sum(np.abs(M),axis=0)[0])
        cv2.waitKey(100)
    print(mode_maker.combinationsUptoN(5, modesOnly=True))
