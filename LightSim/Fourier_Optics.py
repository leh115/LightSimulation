from UpdatePlanes import Updater
from MultiMode import ModePosition as mulmo
from LightProp import LightSim
import numpy as np
import cv2

#* Class variables: LightSim is the underlying class for almost all other classes and so it should be used to update class variables.
#* Do NOT call <otherclass.classvariable> e.g. updater.dz because the classes that Updater uses will not inherit from it.
LightSim.dz = 5e-3
LightSim.kFilter = 0.9
LightSim.number_of_modes = 25
LightSim.PlaneSetUp = [20e-3, 25e-3, 25e-3, 25e-3, 25e-3, 25e-3, 25e-3, 25e-3]
mode_maker = mulmo(Amplitude = 5)

Initial_Beam_Waist = 60e-6
Output_Beam_Waist = 30e-6
Input_Spot_Separation = 0 * Initial_Beam_Waist
Output_Spot_Separation = Output_Beam_Waist * 2

input_modes, output_modes = mode_maker.make_input_output_modes(
    Initial_Beam_Waist,
    Output_Beam_Waist,
    Input_Spot_Separation,
    Output_Spot_Separation,
    "central -> fib",
    "hg -> spot",
)

#print(output_modes.shape)
#cv2.imshow("output", np.resize(np.sum(np.abs(output_modes)**2,axis=0),(270,270)).astype(np.float32))
#for i,mode in enumerate(np.abs(output_modes)**2):
#    cv2.imshow("mode"+str(i), np.resize( np.abs(output_modes)**2, (270,270)).astype(np.float32))

updater = Updater(save_to_file=True) # mask_offset = 0.001
updater.show_modes_at_start = True

updater.GradientDescent(input_modes, output_modes, EpochNumber = 1000, samplingRate=10, showAllModes=False)


# Notes
# Joel Carpenters set-up:
# w0 = 30e-6m and an exit beam of wz = 200e-6m
# Spot separation of 89.8e-6m
# A set of seven planes with a distance of 20mm before the first plane and 12.5mm*2 between each mirror
# Pixel size of 8e-6m with 274pixels so plane width is 2.192e-3m

# I want the beam to interact with every part of the spot array, the array has a diameter of approx. one side length which is (separation dist) * sqrt(number of modes)