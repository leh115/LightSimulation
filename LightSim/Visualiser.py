import numpy as np
import cv2
from LightProp import LightSim
from Propagate import Propagate

class Visualiser(LightSim):
    def __init__(self, PlaneSetUp, modeNum, show_all_modes = False, save_to_file = False, show_Propagation_live = True, show_Planes = False):
        super().__init__(PlaneSetUp, modeNum)
        self.show_all_modes = show_all_modes
        self.save_to_file = save_to_file
        self.show_Propagation_live = show_Propagation_live
        self.show_Planes = show_Planes

    def showProgress(
        self,
        Input,
        Output,
        showPlanes=False,
    ):
        propagator = Propagate(self.PlaneSetUp, self.modeNum, show_beam=False)

        animLength = int(np.sum(self.PlaneSetUp) / self.dz) + len(self.Planes) + 2
        F = None
        B = None
        for mode in range(Input.shape[0]):
            propagator.Beam_Cross_Sections = Input[mode]
            
            propagator.Propagate_FromPlane_ToPlane(
                0,
                len(self.PlaneSetUp),
            )
            if not isinstance(F,np.ndarray):
                F = np.zeros((self.modeNum, len(propagator.Beam_Cross_Sections), self.Nx, self.Ny, 3), dtype=np.complex128)
                B = np.zeros((self.modeNum, len(propagator.Beam_Cross_Sections), self.Nx, self.Ny, 3), dtype=np.complex128)

            F[mode] = self.BeamToRGB(propagator.Beam_Cross_Sections)

            propagator.Beam_Cross_Sections = Output[mode]
            propagator.Propagate_FromPlane_ToPlane(
                0,
                len(self.PlaneSetUp),
                Forwards=False,
            )

            B[mode] = self.BeamToRGB(propagator.Beam_Cross_Sections)

        if self.show_all_modes:
            for m in range(F.shape[0]):
                self.VisualiseBeam(
                    F[m],
                    "Beam Cross section A",
                    "Mode %d" % m,
                )
            for m in range(B.shape[0]):
                self.VisualiseBeam(
                    B[m],
                    "Beam Cross section B",
                    "Mode %d" % m,
                )

        self.VisualiseBeam(
            np.sum(np.abs(F) ** 2, axis=0)*255,
            "Beam Cross section A",
        )

        self.VisualiseBeam(
            np.sum(np.abs(B) ** 2, axis=0)*255,
            "Beam Cross section B",
        )

        cv2.destroyAllWindows()

    def VisualiseBeam(self, X, title, mode=""):
        for part in range(X.shape[0]):
            if self.show_Propagation_live:
                cv2.imshow(
                    title,
                    cv2.cvtColor(np.abs(X[part]).astype(np.float32), cv2.COLOR_RGB2BGR),
                )
            if self.save_to_file:
                if len(mode) > 0:
                    if "Cross section A" in title:
                        cv2.imwrite(
                            self.ROOTDIR
                            + "/Results/ModesA/"
                            + mode
                            + " part %d" % part
                            + ".png",
                            255
                            * cv2.cvtColor(
                                np.abs(X[part]).astype(np.float32), cv2.COLOR_RGB2BGR
                            ),
                        )
                    else:
                        cv2.imwrite(
                            self.ROOTDIR
                            + "/Results/ModesB/"
                            + mode
                            + " part %d" % part
                            + ".png",
                            255
                            * cv2.cvtColor(
                                np.abs(X[part]).astype(np.float32), cv2.COLOR_RGB2BGR
                            ),
                        )
                else:
                    cv2.imwrite(
                        self.ROOTDIR
                        + "/Results/"
                        + title
                        + "/Version%d part %d" % (self.VERSION, part)
                        + ".png",
                        255
                        * cv2.cvtColor(
                            np.abs(X[part]).astype(np.float32), cv2.COLOR_RGB2BGR
                        ),
                    )
            if self.show_Propagation_live:
                cv2.waitKey(30)
                cv2.setWindowProperty(title, cv2.WND_PROP_TOPMOST, 1)

    def visualise_Planes(self):
        for i, Plane in enumerate(self.Planes):
            X = np.angle(Plane)
            X = np.expand_dims(X, 2)
            X2 = X
            X = np.append(X, X2 * 0, axis=2)
            X = np.append(X, (1 - X2) * 0.3, axis=2)
            cv2.imshow("Plane %d" % i, X)
            cv2.setWindowProperty("Plane %d" % i, cv2.WND_PROP_TOPMOST, 1)
            cv2.imwrite(self.ROOTDIR + "/Results/Planes/Plane %d" % i + ".png", 255 * X)
            print("Sum of Plane %d:" % i, sum(sum(Plane)))
            print("Maximum angle of Plane %d" % i, np.max(np.angle(Plane)))

    def wavelength_to_rgb(self, wavelength, gamma=0.8):

        """This converts a given wavelength of light to an
        approximate RGB color value. The wavelength must be given
        in nanometers in the range from 380 nm through 750 nm
        (789 THz through 400 THz).

        Based on code by Dan Bruton
        http://www.physics.sfasu.edu/astro/color/spectra.html
        """

        wavelength = float(wavelength)
        if wavelength >= 380 and wavelength <= 440:
            attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
            R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
            G = 0.0
            B = (1.0 * attenuation) ** gamma
        elif wavelength >= 440 and wavelength <= 490:
            R = 0.0
            G = ((wavelength - 440) / (490 - 440)) ** gamma
            B = 1.0
        elif wavelength >= 490 and wavelength <= 510:
            R = 0.0
            G = 1.0
            B = (-(wavelength - 510) / (510 - 490)) ** gamma
        elif wavelength >= 510 and wavelength <= 580:
            R = ((wavelength - 510) / (580 - 510)) ** gamma
            G = 1.0
            B = 0.0
        elif wavelength >= 580 and wavelength <= 645:
            R = 1.0
            G = (-(wavelength - 645) / (645 - 580)) ** gamma
            B = 0.0
        elif wavelength >= 645 and wavelength <= 750:
            attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
            R = (1.0 * attenuation) ** gamma
            G = 0.0
            B = 0.0
        elif wavelength >= 750:
            R = 1.0 #/ (wavelength - 750)
            G = 0.0
            B = 0.0
        else:
            R = 0.0
            G = 0.0
            B = 0.0
        R *= 255
        G *= 255
        B *= 255
        return (int(R), int(G), int(B))

    def BeamToRGB(self, X:np.ndarray):
        """Turns a grayscale beam with a value for wavelength into a coloured beam

        Args:
            X (np.ndarray): The Input beam of shape (modes, sections, Nx, Ny)

        Returns:
            np.ndarray: A coloured output beam of shape (modes, sections, Nx, Ny 3)
        """
        pseudoColour = self.wavelength_to_rgb(self.wavelength * 1e9)
        X = np.expand_dims(X,axis=3)
        X = np.concatenate((X,X,X),axis=3)
        X[:, :, :, 0] = X[:, :, :, 0] * pseudoColour[0] / 255
        X[:, :, :, 1] = X[:, :, :, 1] * pseudoColour[1] / 255
        X[:, :, :, 2] = X[:, :, :, 2] * pseudoColour[2] / 255
        return X

    def show_Initial(self,In,Out,wait_value=0):
        cv2.imshow("Input Modes", np.sum(np.abs(In**2), axis=0)[0])
        cv2.imshow("Output Modes", np.sum(np.abs(Out**2) , axis=0)[0])
        cv2.waitKey(wait_value)