from UpdatePlanes import Updater
from MultiMode import ModePosition as mulmo
import numpy as np

PlaneSetUp = [20e-3, 25e-3, 25e-3, 25e-3, 25e-3, 25e-3, 25e-3, 25e-3]
Number_Of_Modes = 10
Amplitude = 1

mode_maker = mulmo(PlaneSetUp, Number_Of_Modes, Amplitude)

Initial_Beam_Waist = 30e-6
Output_Beam_Waist = 2 * Initial_Beam_Waist
Input_Spot_Separation = np.sqrt(4) * Initial_Beam_Waist
Output_Spot_Separation = Input_Spot_Separation

input_modes, output_modes = mode_maker.make_input_output_modes(
    Initial_Beam_Waist,
    Output_Beam_Waist,
    Input_Spot_Separation,
    Output_Spot_Separation,
    "Square -> Central",
    "Spot -> HG",
)

updater = Updater(PlaneSetUp, Number_Of_Modes)

updater.maskOffset *= 30
updater.dz = 5e-3
updater.kFilter = 1
updater.show_modes_at_start = True

updater.GradientDescent(input_modes, output_modes, EpochNumber = 1000, samplingRate=10)
