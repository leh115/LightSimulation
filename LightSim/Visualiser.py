import numpy as np
import cv2
from LightProp import LightSim
from Propagate import Propagate
import os
from matplotlib import pyplot as plt
from Beam_Analysis import Beam_Analyser
import matplotlib

class Visualiser(LightSim):
    number_of_progress_calls = 0

    def __init__(
        self,
        show_all_modes=False,
        save_to_file=False,
        show_Propagation_live=True,
        show_Planes=False,
        show_phase=False,
        save_last_only=False,
        show_loss = False,
    ):
        """A class for visualising beams of light.

        Args:
            show_all_modes (bool, optional): At every sample this will loop through every mode, can be very slow in a high mode number system. Defaults to False.
            save_to_file (bool, optional): Saves results or not. Defaults to False.
            show_Propagation_live (bool, optional): If False nothing will be shown, saves still happen though. Defaults to True.
            show_Planes (bool, optional): Shows every phase plane. Defaults to False.
            show_phase (bool, optional): Also shows the (coloured) phase and saves it to file. Defaults to False.
            save_last_only (bool, optional): Only saves the final result of an animation. Defaults to False.
        """
        super().__init__()
        self.show_all_modes = show_all_modes
        self.save_to_file = save_to_file
        self.show_Propagation_live = show_Propagation_live
        self.show_Planes = show_Planes
        self.show_phase = show_phase
        self.save_last_only = save_last_only
        self.show_loss = show_loss
        

    def showProgress(
        self,
        Input,
        Output,
        mask_offset,
    ):
        propagator = Propagate(show_beam=False)
        F = None
        B = None
        for mode in range(Input.shape[0]):
            propagator.Beam_Cross_Sections = Input[mode]

            propagator.Propagate_FromPlane_ToPlane(
                0,
                len(self.PlaneSetUp),
            )
            if not isinstance(F, np.ndarray):
                F = np.zeros(
                    (
                        self.number_of_modes,
                        len(propagator.Beam_Cross_Sections),
                        self.Nx,
                        self.Ny,
                    ),
                    dtype=np.complex128,
                )
                B = np.zeros(
                    (
                        self.number_of_modes,
                        len(propagator.Beam_Cross_Sections),
                        self.Nx,
                        self.Ny,
                    ),
                    dtype=np.complex128,
                )

            F[mode] = propagator.Beam_Cross_Sections

            propagator.Beam_Cross_Sections = Output[mode]
            propagator.Propagate_FromPlane_ToPlane(
                0,
                len(self.PlaneSetUp),
                Forwards=False,
            )

            B[mode] = propagator.Beam_Cross_Sections

        F_intensity = np.abs(F) ** 2
        B_intensity = np.abs(B) ** 2

        F_Phase = np.angle(F)
        B_Phase = np.angle(B)

        if self.show_all_modes:
            for m in range(F.shape[0]):
                self.VisualiseBeam(
                    F_intensity[m],
                    "Beam Cross section A",
                    "Mode %d" % m,
                    on_white="alpha",
                )
                self.VisualiseBeam(
                    B_intensity[m],
                    "Beam Cross section B",
                    "Mode %d" % m,
                    on_white="alpha",
                )
                if self.show_phase:
                    self.VisualiseBeam(
                        F_intensity[m],
                        "Beam Cross section A",
                        "Mode Phase %d" % m,
                        phase=F_Phase[m],
                        on_white="alpha",
                    )
                    self.VisualiseBeam(
                        B_intensity[m],
                        "Beam Cross section B",
                        "Mode Phase %d" % m,
                        phase=B_Phase[m],
                        on_white="alpha",
                    )

        self.VisualiseBeam(
            np.sum(F_intensity, axis=0) * 255,
            "Beam Cross section A",
            "Superposition",
            on_white="alpha",
        )

        self.VisualiseBeam(
            np.sum(B_intensity, axis=0) * 255,
            "Beam Cross section B",
            "Superposition",
            on_white="alpha",
        )

        if self.show_Planes:
            self.visualise_Planes()
        
        if self.show_loss:
            self.show_loss_graph(F_intensity)
        
        self.show_phasor(np.sum(B, axis=0),np.sum(F, axis=0),mask_offset,legend=["Backward propagation","Forward propagation","Mask Offset"])

        cv2.destroyAllWindows()

    def VisualiseBeam(
        self,
        X: np.ndarray,
        title: str,
        mode: str = "",
        on_white="",
        phase=None,
        anti_flicker=True,
    ):
        """Turns beam cross sections into true(ish) colour cross secitons and offers ability to save to file

        Args:
            X (np.ndarray): shape: (n,X,Y) all of the cross sections
            title (str): Name of the cross sections
            mode (str, optional): Name of the modes. Defaults to "".
            on_white (str, optional): Defines background colour as black or white. Defaults to black.
            phase
        """
        if not isinstance(phase, np.ndarray):
            X = np.float32(self.BeamToRGB(X))
        else:
            X = np.float32(self.PhaseToRGB(X, phase))
        animation_length = X.shape[0]
        for i, part in enumerate(range(animation_length)):

            coloured_image = (
                cv2.cvtColor(np.abs(X[part]), cv2.COLOR_RGB2BGR) / np.max(X[part]) * 255
            )
            if len(on_white) > 0:
                normed_colour_image = np.sum(coloured_image, axis=2) / np.max(
                    np.sum(coloured_image, axis=2)
                )
                coloured_image_temp = np.ones(X[part].shape, dtype=np.float32) * 255
                if on_white == "black":
                    coloured_image_temp[:, :, 0] -= 255 * (normed_colour_image)
                    coloured_image_temp[:, :, 1] -= 255 * (normed_colour_image)
                    coloured_image_temp[:, :, 2] -= 255 * (normed_colour_image)
                if on_white == "red":
                    coloured_image_temp[:, :, 0] -= 255 * (normed_colour_image)
                    coloured_image_temp[:, :, 1] -= 200 * (normed_colour_image)
                    coloured_image_temp[:, :, 2] -= 30 * (normed_colour_image)
                if on_white == "green":
                    coloured_image_temp[:, :, 0] -= 250 * (normed_colour_image)
                    coloured_image_temp[:, :, 1] -= 80 * (normed_colour_image)
                    coloured_image_temp[:, :, 2] -= 255 * (normed_colour_image)
                if on_white == "purple":
                    coloured_image_temp[:, :, 0] -= 100 * (normed_colour_image)
                    coloured_image_temp[:, :, 1] -= 255 * (normed_colour_image)
                    coloured_image_temp[:, :, 2] -= 100 * (normed_colour_image)
                if on_white == "blue":
                    coloured_image_temp[:, :, 0] -= 100 * (normed_colour_image)
                    coloured_image_temp[:, :, 1] -= 100 * (normed_colour_image)
                    coloured_image_temp[:, :, 2] -= 255 * (normed_colour_image)
                if on_white == "alpha":
                    coloured_image_temp = cv2.cvtColor(
                        coloured_image, cv2.COLOR_RGB2RGBA
                    )
                    coloured_image_temp[:, :, 3] = 255 * (normed_colour_image)

                coloured_image = coloured_image_temp

            if self.show_Propagation_live:
                if anti_flicker:
                    if i % 2 == 0:
                        cv2.imshow(
                            title,
                            coloured_image / 255,
                        )
                else:
                    cv2.imshow(
                        title,
                        coloured_image / 255,
                    )

            if self.save_to_file:
                if (
                    self.save_last_only and i == animation_length - 1
                ) or self.save_last_only == False:
                    path = os.path.join(self.ROOTDIR, "Results", title, mode)
                    isdir = os.path.isdir(path)
                    if not isdir:
                        os.makedirs(path)
                    filename = os.path.join(
                        path, f"{mode} Version {self.VERSION} part {part}.png"
                    )
                    cv2.imwrite(
                        filename,
                        coloured_image,
                    )

            if self.show_Propagation_live:
                cv2.waitKey(30)
                if self.make_windows_top:
                    cv2.setWindowProperty(title, cv2.WND_PROP_TOPMOST, 1)

    def visualise_Planes(self):
        for i, Plane in enumerate(LightSim.Planes):
            X = np.angle(Plane)
            X = np.expand_dims(X, 2)
            X = np.concatenate((X, X, X), axis=2)
            cv2.imshow(f"Plane {i}", X)
            if self.make_windows_top:
                cv2.setWindowProperty(f"Plane {i}", cv2.WND_PROP_TOPMOST, 1)
            cv2.imwrite(
                os.path.join(
                    LightSim.ROOTDIR,
                    f"Results/Planes/Plane {i} Version {LightSim.VERSION}.png",
                ),
                255 * X,
            )
            cv2.waitKey(100)
            print(f"Sum of Plane {i}:", sum(sum(Plane)))
            print(f"Maximum angle of Plane {i}", np.max(np.angle(Plane)))

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
            R = 1.0  # / (wavelength - 750)
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

    def BeamToRGB(self, X: np.ndarray):
        """Turns a grayscale beam with a value for wavelength into a coloured beam

        Args:
            X (np.ndarray): The Input beam of shape (modes, sections, Nx, Ny)

        Returns:
            np.ndarray: A coloured output beam of shape (modes, sections, Nx, Ny 3)
        """
        pseudoColour = self.wavelength_to_rgb(self.wavelength * 1e9)
        if len(X.shape) == 3:
            X = np.expand_dims(X, axis=3)
            X = np.concatenate((X, X, X), axis=3)
            X[:, :, :, 0] = X[:, :, :, 0] * pseudoColour[0] / 255
            X[:, :, :, 1] = X[:, :, :, 1] * pseudoColour[1] / 255
            X[:, :, :, 2] = X[:, :, :, 2] * pseudoColour[2] / 255
        elif len(X.shape) == 4:
            X = np.expand_dims(X, axis=4)
            X = np.concatenate((X, X, X), axis=4)
            X[:, :, :, :, 0] = X[:, :, :, :, 0] * pseudoColour[0] / 255
            X[:, :, :, :, 1] = X[:, :, :, :, 1] * pseudoColour[1] / 255
            X[:, :, :, :, 2] = X[:, :, :, :, 2] * pseudoColour[2] / 255
        return X

    def PhaseToRGB(self, X: np.ndarray, phase):
        if len(X.shape) == 3:
            X = np.expand_dims(X, axis=3)
            X = np.concatenate((X, X, X), axis=3)
            X[:, :, :, 0] = X[:, :, :, 0] * phase / 255
            X[:, :, :, 1] = X[:, :, :, 1] * (1 - phase) / 255
            X[:, :, :, 2] = X[:, :, :, 2] * 0 / 255
        return X

    def show_Initial(self, In, Out, wait_value=0):
        cv2.imshow("Input Modes", np.sum(np.abs(In**2), axis=0)[0])
        cv2.imshow("Output Modes", np.sum(np.abs(Out**2), axis=0)[0])
        cv2.waitKey(wait_value)
    
    def show_loss_graph(self,X):
        loss = np.sum(X, axis=(2,3))
        #plt.close()
        ax = self.threeD_loss_ax
        print(ax)
        Visualiser.threeD_loss_kfilter.append([self.kFilter]*loss.shape[1])
        Visualiser.threeD_loss_vals.append(np.sum(loss, axis=0) / np.max(np.sum(loss, axis=0)))
        ax.plot(np.array(range(loss.shape[1]))*self.dz, np.sum(loss, axis=0) / np.max(np.sum(loss, axis=0)), [self.kFilter]*loss.shape[1] )
        legend = ["Superposition loss"]
        for mode_number, mode in enumerate(loss):
            ax.plot(np.array(range(loss.shape[1]))*self.dz, mode/np.max(mode), [self.kFilter]*loss.shape[1])
            legend.append(f"Mode {mode_number} loss")
        Visualiser.threeD_loss_figure = ax
        plt.title(f"Mode losses")
        plt.xlabel("Axial position (m)")
        plt.ylabel("Normalised Loss")
        plt.legend(legend)
        y_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
        plt.gca().yaxis.set_major_formatter(y_formatter)
        plt.savefig(f"{self.ROOTDIR}Results/Loss Graphs/Version {self.VERSION} at stage {self.number_of_progress_calls}.png")
        plt.show()
        

    def show_phasor(self, *phases, legend:list = None, single_values = None, animate = False):

        def convert_polar_xticks_to_radians(ax):
            """This function, was slightly modified from an answer found on https://stackoverflow.com/questions/21172228/python-matplotlib-polar-plots-with-angular-labels-in-radians"""
            # Converts x-tick labels from degrees to radians

            # Get the x-tick positions (returns in radians)
            label_positions = ax.get_xticks()

            # Convert to a list since we want to change the type of the elements
            labels = list(label_positions)

            # Format each label (edit this function however you'd like)
            labels = [f"{label / (np.pi)}π" for label in labels]

            ax.set_xticklabels(labels)
        
        
        good_colours = Beam_Analyser.modeColours
        #if isinstance(single_values,list):
        fig = plt.figure()
        if animate == True:
            steps = len(phases[0])
        else:
            steps = 1
        for z in range(steps):
            #plt.close()
            
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
            convert_polar_xticks_to_radians(ax)
            plt.grid(True)
            overall_max = 0
            for i,phase in enumerate(phases):
                if isinstance(phase,list):
                    if np.max(phase[z])>overall_max:
                        overall_max = np.max(phase[z])
                else:
                    if np.max(phase)>overall_max:
                        overall_max = np.max(phase)

            for i,phase in enumerate(phases):
                if isinstance(phase,list):
                    this_phase = phase[z]
                else:
                    this_phase = phase
                plt.arrow(
                    #np.angle(np.sum(this_phase)),
                    np.angle(np.sum(this_phase)),
                    0,
                    0,
                    1,
                    #np.abs(np.max(phase)/overall_max),
                    #alpha=1,
                    width=0.03,
                    edgecolor=None,
                    facecolor=good_colours[i],
                    lw=0,
                    zorder=10000, # chooses when the arrow is drawn, (should be the very last)
                )
            
            if not legend == None:
                plt.legend(legend)
            plt.title("Phasor diagram")
            plt.show(block=False)
            plt.savefig(f"{self.ROOTDIR}Results/Phasor diagrams/Version {self.VERSION} at stage {self.number_of_progress_calls}.png")
            if animate:
                cv2.waitKey(100)
            else:
                cv2.waitKey(500)
            


if __name__ == "__main__":
    visual = Visualiser()
    visual.show_phasor(0.1 + 1j, -1 - 0.1j, [0.1+ 1j, -1 - 0.1j],legend=["B","F","MO"])
