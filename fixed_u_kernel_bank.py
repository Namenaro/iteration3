import numpy as np
from kernel_sutuations_bank import *

class FixedUSituation:
    def __init__(self, kernel_situation, uactivation_matrix):
        self.kernel_situation = kernel_situation
        self.uactivation_matrix = uactivation_matrix

class FixedUKernelBank:
    def __init__(self, kernel_bank, ux, uy, hside):
        self.u_situations = []

        self.ux = ux
        self.uy = uy
        self.hside = hside

        for situation in kernel_bank.situations:
            uactivation_matrix = self._get_u_activation_matrix(situation)
            if uactivation_matrix is not None:
                u_situation = FixedUSituation(situation, uactivation_matrix)
                self.u_situations.append(u_situation)

    def _get_u_activation_matrix(self, kernel_situation):
        x_center = kernel_situation.x + self.ux
        y_center = kernel_situation.y + self.uy
        imlen = kernel_situation.image.shape[0]
        if not self._check_if_in_bounds(x_center, y_center, imlen):
            return None
        x_min = x_center - self.hside
        x_max = x_center + self.hside + 1
        y_min = y_center - self.hside
        y_max = y_center + self.hside + 1
        return kernel_situation.image[y_min:y_max, x_min:x_max]


    def _check_if_in_bounds(self, x_center, y_center, imlen):
        if x_center < self.hside or y_center < self.hside:
            return False
        if x_center > (imlen - self.hside -1) or \
            y_center >  (imlen - self.hside -1):
            return False
        return True


    def get_raw_activations_matrixes(self):
        matrixes = []
        for situation in self.u_situations:
            matrixes.append(situation.uactivation_matrix)
        return np.array(matrixes)

    def get_activations_of_kernel(self):
        activations = []
        for usituation in self.u_situations:
            activations.append(usituation.kernel_situation.activation)
        return np.array(activations)

if __name__ == "__main__":
    bank = BankCreator().load_bank()
    bank.show_first_n_situations()
    ux = -6
    uy = -2
    hside = 3
    ubank = FixedUKernelBank(bank, ux, uy, hside)
    print (ubank.get_raw_activations_matrixes().shape)
    from tsne_visualise import  *
    tnse_visialise_with_colors(ubank.get_raw_activations_matrixes(), ubank.get_activations_of_kernel())