from turtle import st
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline
from LightProp import LightSim

class Beam_Analyser(LightSim):
    def __init__(self):
        super().__init__()
        self.CoMConvergence = []
        self.CouplingResults = []
        self.avgcoupler = []
        self.coupleMat = []
        self.Complex_Difference_Convergence = []
        self.average_Complex_Difference_Convergence = 1
        self.modeColours = [
            "r",
            "g",
            "b",
            "k",
            "m",
            "c",
            "y",
            "grey",
            "darksalmon",
            "olivedrab",
            "crimson",
            "olive",
            "pink",
            "darkmagenta",
            "chocolate",
            "sienna",
        ] * 50

    def beam_width(self, X:np.ndarray, style:str = "FWHM")->float:
        """Returns the beam width of an image (assumed to be a Gaussian Beam)

        Args:
            X (np.ndarray): A Cross Section of the beam at a position z. 
            style (str, optional): Can be fwhm or one_over_e_squared . Defaults to "fwhm".

        Returns:
            float: The beam waist
        """
        style = style.upper()
        if X.dtype == np.complex128:
            X = np.abs(X)
        if isinstance(X, np.ndarray):
            if len(X.shape)>2:
                X = X[0]
                if len(X.shape)>2:
                    X = X[0]
        X = np.real(X)
        s = X.shape
        lineIndex = int(s[0] / 2)
        y = X[lineIndex, :]
        y /= np.max(y)
        x = np.linspace(0, self.Nx * self.pixelSize, s[1])
        factor_difference = 1
        if style == "FWHM":
            spline = UnivariateSpline(x, y - np.max(y) / 2, s=0)
            factor_difference = 1.699 / 2
        if style == "ONE_OVER_E_SQUARED":
            spline = UnivariateSpline(x, y - np.max(y) * 0.13533528323, s=0)
            factor_difference = 1 / 2
        r1, r2 = spline.roots()
        return (r2 - r1)*factor_difference

    def show_beam_waist_calculation(self,x,y,r1,r2):
        plt.scatter(x, y)
        plt.scatter(
            np.linspace(r1, r2, 100),
            np.ones((100)) * np.max(y) / 2,
            s=3,
            marker=".",
            c="red",
        )
        plt.scatter(r1, np.max(y) / 2, marker="o", c="red", s=5)
        plt.scatter(r2, np.max(y) / 2, marker="o", c="red", s=5)
        plt.savefig(self.ROOTDIR + "Results/FWHM/%d.png" % save_number)
        plt.title(
            "Full Width Half Maxima: %.3fmm, z: %.3f" % (((r2 - r1) * 1000), z)
        )
        plt.xlabel("x position (mode)")
        plt.ylabel("Normalised Intensity")
        plt.show()
        # plt.show(block=False)
        plt.close()

    def CentreOfMass(self, Img):
        """Returns the centre of mass of an image (a bit dodgy but kind of works)"""
        CoMx = 0
        CoMy = 0
        xSum = 0
        ySum = 0
        X = Img.copy()
        X /= np.max(X)
        try:
            for i in range(X.shape[0]):
                CoMx += np.sum(np.abs(np.real(X[i, :] * list(range(X.shape[1])))))
                xSum += np.sum(np.abs(np.real(X[i, :])))
            for i in range(X.shape[1]):
                CoMy += np.sum(np.abs(np.real(X[:, i] * list(range(X.shape[0])))))
                ySum += np.sum(np.abs(np.real(X[:, i])))
            return [CoMx / xSum, CoMy / ySum]
        except:
            return [0, 0]
    
    def Centre_difference(self,Img1:np.ndarray,Img2:np.ndarray) -> float:
        """Compares the centre of mass of two images and returns their difference, e.g. two spots close together will return a smaller value than two spots far apart

        Args:
            Img1 (np.ndarray): The first image
            Img2 (np.ndarray): The second image

        Returns:
            float: The difference of centre of mass
        """
        [x1, y1] = self.CentreOfMass(
            np.abs(Img1)
        )  # these are the centre of masses of the desired output modes
        [x2, y2] = self.CentreOfMass(
            np.abs(Img2)
        )  # these are the simulated centre of masses of output modes
        X_diff = np.abs(x1 - y1)
        Y_diff = np.abs(x2 - y2)
        return X_diff + Y_diff
    
    def save_Centre_Of_Mass_Convergence(self, modeCentreDifferences):
        self.CoMConvergence.append(modeCentreDifferences)

        plt.close()
        for mode in range(self.number_of_modes):
            for x in range(len(self.CoMConvergence)):
                plt.scatter(x, self.CoMConvergence[x][mode], c=self.modeColours[mode])
        plt.title("Centre of Mass Convergence")
        plt.xlabel("Epoch")
        plt.ylabel("Pixel Distance")
        plt.savefig(
            self.ROOTDIR
            + "/Results/CoMConvergenceGraph"
            + str(self.VERSION)
            + " "
            + self.Variable_Name
            + ".png"
        )
    def complex_difference(self,F:np.ndarray,B:np.ndarray):
        return np.sum( np.abs(np.imag(F) + np.imag(B)) )

    def coupling_analysis(self,F:np.ndarray,B:np.ndarray, current_coupling_matrix, current_complex_difference):
        return [current_coupling_matrix + self.coupling_coefficient(F,B), current_complex_difference + self.complex_difference(F,B)]

    def coupling_coefficient(self,F:np.ndarray,B:np.ndarray) -> float:
        """Calculates the coupling coefficient of two Cross sections travelling in different directions (usually from the same axial location)

        Args:
            F (np.ndarray): Forward travelling cross section
            B (np.ndarray): Backward travelling cross section

        Returns:
            float: The coupling coefficient
        """
        return np.real(
                            np.sum(F * B) ** 2
                        )  # shouldn't have to take real because it is squared but python complains otherwise

    def save_multi_var_coupling(self):
        self.CouplingResults.append(self.avgcoupler)
        plt.close()
        plt.scatter(self.Variable_Tests[: len(self.CouplingResults)], self.CouplingResults)
        plt.title("Average Field Coupling")
        plt.xlabel(self.Variable_Name)
        plt.ylabel("Normalised Coupling")
        plt.savefig(
            self.ROOTDIR + "/Results/Average Coupling Graph " + self.Variable_Name + ".png"
        )

    def save_current_complex_convergence(self, Complex_Difference):
        self.Complex_Difference_Convergence.append(Complex_Difference)
        self.average_Complex_Difference_Convergence = np.sum(self.Complex_Difference_Convergence / np.max(self.Complex_Difference_Convergence)) / len(
             self.Complex_Difference_Convergence)
        plt.close()
        plt.scatter(
            list(range(len(self.Complex_Difference_Convergence))),
            self.Complex_Difference_Convergence / np.max(self.Complex_Difference_Convergence),
        )
        plt.title("Field Phase Convergence")
        plt.xlabel("Epoch")
        plt.ylabel("Normalised Phase difference")
        plt.savefig(
            self.ROOTDIR
            + "/Results/ConvergenceGraph"
            + str(self.VERSION)
            + " "
            + self.Variable_Name
            + ".png"
        )
    
    def w0_theory_vs_sim(self, X, z, initial_beam_waist):
        sim_wz = self.beam_width(X,style="one_over_e_squared")
        theory_wz = self.Theoretical_BeamWaist(initial_beam_waist,z)
        return [sim_wz, theory_wz]
    
if __name__ == "__main__":
    from Propagate import Propagate
    from MultiMode import ModePosition as mulmo
    LightSim.Nx += 1
    LightSim.number_of_modes = 1
    mode_maker = mulmo(Amplitude=1)
    propagator = Propagate(override_dz=True)
    analyser = Beam_Analyser()
    
    w0 = 30e-6
    single_mode = mode_maker.makeModes(w0,0,"central","spot")[0]
    
    differences = []
    factor_difference = []
    for i in range(900):
        z = 0.01+ 0.0001*i
        propagator.Beam_Cross_Sections = single_mode
        propagator >> z
        X = propagator.Beam_Cross_Sections[-1]
        sim_wz, theory_wz = analyser.w0_theory_vs_sim(X, z, w0)
        plt.scatter(z, sim_wz, color="red")
        plt.scatter(z, theory_wz, color="blue")
        print(f"Factor difference: {theory_wz / sim_wz}", f"Difference: {theory_wz - sim_wz}",sep=" ~~ ")
        differences.append(theory_wz - sim_wz)
        factor_difference.append(theory_wz / sim_wz)
        print(i)

    plt.title("Beam waist at distance z")
    plt.xlabel("Distance z (m)")
    plt.ylabel("Beam Waist (m)")
    plt.legend(["Simulation","Theory"])
    plt.show()
    plt.scatter(np.linspace(0.01,0.01+0.001*i,i+1), factor_difference, color="blue")
    import matplotlib
    y_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
    plt.gca().yaxis.set_major_formatter(y_formatter)
    plt.title("Theory/Simulation breakdown")
    plt.xlabel("Distance z (m)")
    plt.ylabel(r'$\omega_{z_{Theory}} / \omega_{z_{Simulation}}$')
    plt.show()
    plt.scatter(np.linspace(0.01,0.01+0.001*i,i+1), differences, color="blue")
    plt.title("Theory - Simulation breakdown")
    plt.xlabel("Distance z (m)")
    plt.ylabel(r'$\omega_{z_{Theory}} - \omega_{z_{Simulation}}$')
    plt.show()