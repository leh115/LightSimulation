from UpdatePlanes import Updater
from MultiMode import ModePosition as mulmo
from LightProp import LightSim
import numpy as np


for version_number, sep_factor in enumerate([1]):#np.linspace(1,20,10)):
    # * Class variables: LightSim is the underlying class for almost all other classes and so it should be used to update class variables.
    # * Do NOT call <otherclass.classvariable> e.g. updater.dz because the classes that Updater uses will not inherit from it.
    LightSim.dz = 5e-3
    LightSim.kFilter = 0.9#+version_number/10
    LightSim.number_of_modes = 9
    LightSim.PlaneSetUp = [20e-3, 25e-3, 25e-3, 25e-3, 25e-3, 25e-3, 25e-3, 25e-3]
    LightSim.reverse_time = False
    LightSim.make_windows_top = False
    LightSim.VERSION = 1007 + version_number
    mode_maker = mulmo(Amplitude=10)

    Initial_Beam_Waist = 60e-6
    Output_Beam_Waist = 200e-6
    Input_Spot_Separation = 4 * Initial_Beam_Waist
    Output_Spot_Separation = Output_Beam_Waist * 0

    input_modes, output_modes = mode_maker.make_input_output_modes(
        Initial_Beam_Waist,
        Output_Beam_Waist,
        Input_Spot_Separation,
        Output_Spot_Separation,
        "square -> central",
        "spot -> hg",
        propagate_to_output=False,
    )

    updater = Updater(save_to_file=False)
    updater.mask_offset = ( np.sqrt(137) / np.sqrt(0**2 + 1**2) ) * ( 0*np.sqrt(1e-3 / (updater.Nx * updater.Ny)) + 1*1j * np.sqrt(1e-3 / (updater.Nx * updater.Ny)) )
    updater.show_modes_at_start = False

    updater.GradientDescent(
        input_modes,
        output_modes,
        EpochNumber=350,
        samplingRate=350-1,
        showAllModes=False,
        show_phase=True,
        show_Propagation_live=False,
        save_last_only=False,
        show_loss=False,
    )
    # updater.mask_offset *= 4 / 3.5 
    # updater.GradientDescent(
    #     input_modes,
    #     output_modes,
    #     EpochNumber=20,
    #     samplingRate=100,
    #     showAllModes=False,
    #     show_phase=False,
    #     save_last_only=True,
    # )
    # updater.mask_offset *= 4.25 / 4
    # updater.GradientDescent(
    #     input_modes,
    #     output_modes,
    #     EpochNumber=30,
    #     samplingRate=30-1,
    #     showAllModes=True,
    #     show_phase=True,
    #     save_last_only=True,
    #     show_Propagation_live=True,
    # )
    # updater.mask_offset *= 4.5 / 4#.25
    # updater.GradientDescent(
    #     input_modes,
    #     output_modes,
    #     EpochNumber=20,
    #     samplingRate=20 - 1,
    #     showAllModes=True,
    #     show_phase=True,
    #     save_last_only=True,
    #     show_planes=True,
    # )


# Notes
# Joel Carpenters set-up:
# w0 = 30e-6m and an exit beam of wz = 200e-6m
# Spot separation of 89.8e-6m
# A set of seven planes with a distance of 20mm before the first plane and 12.5mm*2 between each mirror
# Pixel size of 8e-6m with 274pixels so plane width is 2.192e-3m

# I want the beam to interact with every part of the spot array, the array has a diameter of approx. one side length which is (separation dist) * sqrt(number of modes)
