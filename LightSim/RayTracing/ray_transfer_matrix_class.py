import numpy as np

class ray_transfer:
    def __init__(self, debug) -> None:
        self.debug = debug
        self.last_method_name = []

    def flat_mirror(self, beam_matrix, mirror_rotation=[0, 0], beam_rotation=[0, 0]):
        flat_mirror_matrix = np.array([[1, 0], [0, 1]])
        return np.round(np.matmul(beam_matrix, flat_mirror_matrix.transpose()), 2)

    def free_space(self, beam_matrix, distance):
        try:
            assert beam_matrix.shape == (2,1) or beam_matrix.shape == (2,), f"Input beam Matrix needs to be of shape (2, 1) or (2,), not {beam_matrix.shape}"
        except Exception as e:
            print(e)
        free_space_matrix = np.array([[1, distance], [0, 1]])
        self.debugger(
            f"Propagation with matrix:\n {np.round(free_space_matrix[0], 2)}\n {np.round(free_space_matrix[1], 2)}"
        ,"Free space",1)
        return np.round(np.matmul(beam_matrix, free_space_matrix.transpose()), 2)
    
    def loc_rot_2_mats(self, location= [0,0], rotation = 0):
        beam_matrix_x = np.array([location[0], np.sin(rotation)])
        beam_matrix_y = np.array([location[1], np.cos(rotation)])
        return beam_matrix_x, beam_matrix_y
    
    
    def debugger(self, debug_str:str, method_name:str = "", method_int = 0):
        if self.debug:
            if self.last_method_name is not method_name:
                print("")
            print(str(" "*method_int*4) +"~"+method_name +" ... "+ str(debug_str))
            self.last_method_name = method_name

if __name__ == "__main__":    
    rays = ray_transfer(True)
    r = np.pi/2
    d = 1
    beam_x_mat, beam_y_mat = rays.loc_rot_2_mats(location= [1, 0], rotation=r) # should be of format [x,y] [theta]
    # rays.flat_mirror()
    print(beam_x_mat)
    beam_x_mat_2 = rays.free_space(beam_x_mat, distance=d*np.sin(r))
    beam_y_mat_2 = rays.free_space(beam_y_mat, distance=d*np.cos(r))
    print(beam_x_mat_2)
    print(beam_y_mat_2)
    
    print(" ")
    
    beam_x_mat, beam_y_mat = rays.loc_rot_2_mats(location= [1, 0], rotation=r)
    print(f"rotation: {r}")
    print(f"Inx: {beam_x_mat}")
    print(f"Outx: {rays.flat_mirror(beam_x_mat)}")
    print(f"Iny: {beam_y_mat}")
    print(f"Outy: {rays.flat_mirror(beam_y_mat)}")
