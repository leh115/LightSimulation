import numpy as np

class ray_transfer:
    def __init__(self) -> None:
        pass

    def flat_mirror(self, mirror_rotation=[0, 0], beam_rotation=[0, 0]):
        flat_mirror_matrix = np.array([[1, 0], [0, 1]])
        

    def free_space(self, beam_matrix, distance):
        try:
            assert beam_matrix.shape == (2,1) or beam_matrix.shape == (2,), f"Input beam Matrix needs to be of shape (2, 1) or (2,), not {beam_matrix.shape}"
        except Exception as e:
            print(e)
        free_space_matrix = np.array([[1, distance], [0, 1]])
        print(
            f"\nFree space propagation with matrix:\n {np.round(free_space_matrix[0], 2)}\n {np.round(free_space_matrix[1], 2)}\n"
        )
        return np.round(np.matmul(beam_matrix, free_space_matrix.transpose()), 2)
    
    def loc_rot_2_mats(self, location= [0,0], rotation = 0):
        beam_matrix_x = np.array([location[0], np.sin(rotation)])
        beam_matrix_y = np.array([location[1], np.cos(rotation)])
        return beam_matrix_x, beam_matrix_y


if __name__ == "__main__":    
    rays = ray_transfer()
    r = -np.pi/2
    d = 1
    beam_x_mat, beam_y_mat = rays.loc_rot_2_mats(location= [1, 0], rotation=r) # should be of format [x,y] [theta]
    # rays.flat_mirror()
    print(beam_x_mat)
    beam_x_mat_2 = rays.free_space(beam_x_mat, distance=d*np.sin(r))
    beam_y_mat_2 = rays.free_space(beam_y_mat, distance=d*np.cos(r))
    print(beam_x_mat_2)
    print(beam_y_mat_2)
