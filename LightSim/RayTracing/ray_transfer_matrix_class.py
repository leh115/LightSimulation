import numpy as np


class ray_transfer:
    def __init__(self, debug) -> None:
        self.debug = debug
        self.last_method_name = []

    def flat_mirror(self, beam_matrix, mirror_rotation=[0, 0], beam_rotation=[0, 0]):
        flat_mirror_matrix = np.array([[1, 0], [0, -1]])
        self.matrix_debugger(flat_mirror_matrix, "Flat Mirror", "Bounce", 2)
        return np.round(np.matmul(beam_matrix, flat_mirror_matrix.transpose()), 2)

    def thin_lens(self, beam_matrix):
        f = 0.5
        thin_lens_matrix = np.array([[1,0],[-1/f,1]])
        self.matrix_debugger(thin_lens_matrix, "Thin Lens", "Lensing", 2)
        return self.free_space(np.round(np.matmul(beam_matrix, thin_lens_matrix.transpose()), 2),1)

    def free_space(self, beam_matrix, distance):
        assert type(beam_matrix) is np.ndarray, f"Needs to be a numpy array, not a {type(beam_matrix)}"
        try:
            assert beam_matrix.shape == (2, 1) or beam_matrix.shape == (
                2,
            ), f"Input beam Matrix needs to be of shape (2, 1) or (2,), not {beam_matrix.shape}"
        except Exception as e:
            print(e)
        free_space_matrix = np.array([[1, distance], [0, 1]])
        self.matrix_debugger(free_space_matrix, "Free Space", "Propagation", 1)
        return np.round(np.matmul(beam_matrix, free_space_matrix.transpose()), 2)

    def loc_rot_2_mats(self, location=[0, 0], rotation=0):
        beam_matrix_x = np.array([location[0], np.sin(rotation)])
        beam_matrix_y = np.array([location[1], np.cos(rotation)])
        return beam_matrix_x, beam_matrix_y

    def debugger(self, debug_str: str, method_name: str = "", method_int=0):
        if self.debug:
            if self.last_method_name is not method_name:
                print("")
            print(
                str(" " * method_int * 4) + "~" + method_name + " ... " + str(debug_str)
            )
            self.last_method_name = method_name

    def matrix_debugger(self, matrix, method_name, matrix_name, method_int):
        self.debugger(f"{matrix_name} with matrix: ", method_name, method_int)
        self.debugger(f"{np.round(matrix[0], 2)}", method_name, method_int)
        self.debugger(f"{np.round(matrix[1], 2)}", method_name, method_int)

if __name__ == "__main__":
    rays = ray_transfer(True)
    r = np.pi / 4
    d = 1
    # beam_x_mat, beam_y_mat = rays.loc_rot_2_mats(location= [1, 0], rotation=r) # should be of format [x,y] [theta]
    # rays.flat_mirror()
    # print(beam_x_mat)
    # beam_x_mat_2 = rays.free_space(beam_x_mat, distance=d*np.sin(r))
    # beam_y_mat_2 = rays.free_space(beam_y_mat, distance=d*np.cos(r))
    # print(beam_x_mat_2)
    # print(beam_y_mat_2)

    print(" ")
    beam = np.array([0, r])
    print(beam)
    beam = rays.free_space(beam, 1)
    beam = rays.flat_mirror(beam)
    beam = rays.free_space(beam, 1)
    print(beam)
    # beam_x_mat, beam_y_mat = rays.loc_rot_2_mats(location= [1, 0], rotation=r)
    # print(f"rotation: {r}")
    # print(f"Inx: {beam_x_mat}")
    # print(f"Outx: {rays.flat_mirror(beam_x_mat)}")
    # print(f"Iny: {beam_y_mat}")
    # print(f"Outy: {rays.flat_mirror(beam_y_mat)}")
