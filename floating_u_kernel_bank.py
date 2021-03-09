import numpy as np
from kernel_sutuations_bank import *
from fixed_u_kernel_bank import FixedUKernelBank

class FloatingUSituation:
    def __init__(self, kernel_situation, B_activation_matrix, realdx, realdy, dist):
        self.kernel_situation = kernel_situation
        self.B_uactivation_matrix = B_activation_matrix
        self.realdx = realdx
        self.realdy =realdy
        self.dist = dist


class FloatingU:
    def __init__(self, x, y, hside, dx, dy):
        self.x = x
        self.y = y
        self.hside = hside
        self.dx = dx
        self.dy = dy


class FloatingUKernelBank:
    def __init__(self, floating_u, kernelA_bank, kernel_B):
        self.fl_situations = []
        self.u = floating_u
        self.kernel_B = kernel_B
        for situationA in kernelA_bank.situations:
            B_activation_matrix, dx, dy, dist = self._get_best_B_activation_matrix(situationA)
            if B_activation_matrix is not None:
                fl_situation = FloatingUSituation(situationA, B_activation_matrix, dx, dy, dist)
                self._add_situation(fl_situation)

    def _add_situation(self, fl_situation):
        len_before = len(self.fl_situations)
        if len_before == 0:
            self.fl_situations.append(fl_situation)
        else:
            for i in range(len(self.fl_situations)):
                if self.fl_situations[i].dist > fl_situation.dist:
                    self.fl_situations.insert(i-1, fl_situation)
                    break
        if len_before == len(self.fl_situations):
            self.fl_situations.append(fl_situation)


    def _get_best_B_activation_matrix(self, kernel_A_situation):
        best_dist = None
        best_dx = None
        best_dy = None
        imlen = kernel_A_situation.image.shape[0]

        for dx in range(-self.u.dx, self.u.dx):
            for dy in range(-self.u.dy, self.u.dy):
                x_center = kernel_A_situation.x + self.u.x + dx
                y_center = kernel_A_situation.y + self.u.y + dy
                if self._check_if_in_bounds(x_center, y_center, imlen):
                    dist = self._get_activationB(x_center, y_center, kernel_A_situation)
                    if best_dist is None:
                        best_dist = dist
                        best_dx = dx
                        best_dy = dy
                    else:
                        if dist < best_dist:
                            best_dist = dist
                            best_dx = dx
                            best_dy = dy
                        else:
                            pass
        if best_dist is not None:
            best_center_x = kernel_A_situation.x + self.u.x + best_dx
            best_center_y = kernel_A_situation.y + self.u.y + best_dy
            best_matrix = self._get_patch_of_image(best_center_x, best_center_y, kernel_A_situation)
            return best_matrix, best_dx, best_dy, best_dist
        else:

            return None, None, None, None

    def _get_patch_of_image(self, x_center, y_center, kernel_A_situation):
        x_min = x_center - self.u.hside
        x_max = x_center + self.u.hside + 1
        y_min = y_center - self.u.hside
        y_max = y_center + self.u.hside + 1
        return kernel_A_situation.image[y_min:y_max, x_min:x_max]

    def _get_activationB(self, x_center, y_center, kernel_A_situation):
        patch = self._get_patch_of_image( x_center, y_center, kernel_A_situation)
        dist = np.linalg.norm(patch - self.kernel_B)
        return dist

    def _check_if_in_bounds(self, x_center, y_center, imlen):
        if x_center < self.u.hside or y_center < self.u.hside:
            return False
        if x_center > (imlen - self.u.hside - 1) or \
            y_center >  (imlen - self.u.hside - 1):
            return False
        return True

    def get_raw_activations_matrixes(self):
        matrixes = []
        for situation in self.fl_situations:
            matrixes.append(situation.B_uactivation_matrix)
        return np.array(matrixes)

    def show_first_n_situations(self):
        plt.figure()
        rows = 20
        cols = 10
        num_of_images = rows * cols
        if num_of_images > len(self.fl_situations):
            num_of_images = len(self.fl_situations) -1
        for index in range(1, num_of_images + 1):
            plt.subplot(rows, cols, index)
            plt.axis('off')
            usituation = self.fl_situations[index]
            plt.imshow(usituation.B_uactivation_matrix, cmap='gray_r')
        plt.show()

    def show_first_n_situationsA(self):
        plt.figure()
        rows = 20
        cols = 10
        num_of_images = rows * cols
        if num_of_images > len(self.fl_situations):
            num_of_images = len(self.fl_situations) -1
        for index in range(1, num_of_images + 1):
            plt.subplot(rows, cols, index)
            plt.axis('off')
            usituation = self.fl_situations[index]
            plt.imshow(usituation.kernel_situation.image, cmap='gray_r')
        plt.show()



if __name__ == "__main__":
    bank = BankCreator().load_bank()
    bank.show_first_n_situations()
    ux = -5
    uy = -2
    hside = 5
    ubank = FixedUKernelBank(bank, ux, uy, hside)
    kernel_B = ubank.get_raw_activations_matrixes()[15]
    plt.imshow(kernel_B, cmap='gray_r')
    plt.show()

    dx = 5
    dy = 5
    floating_u = FloatingU(ux,uy,hside, dx, dy)
    fl_bank = FloatingUKernelBank(floating_u, bank, kernel_B)
    print(fl_bank.get_raw_activations_matrixes().shape)
    fl_bank.show_first_n_situations()
    fl_bank.show_first_n_situationsA()
    from tsne_visualise import *

    tnse_visialise(fl_bank.get_raw_activations_matrixes())
    tnse_visialise_with_contrast(fl_bank.get_raw_activations_matrixes())











