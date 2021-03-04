import numpy as np
import pickle
import matplotlib.pyplot as plt

class Situation:
    def __init__(self, image, x, y, activation):
        self.image = image
        self.x = x
        self.y = y
        self.activation = activation


class SituationsBank:
    def __init__(self, size, name="noname.bank"):
        self.name = name
        self.size = size
        self.situations = []

    def try_add_situation(self, image, x, y, activation):
        i = self._find_i(activation)
        if i > self.size:
            return
        new_situation = Situation(image, x, y, activation)
        self.situations.insert(i, new_situation)
        print("added new situation...")
        if len(self.situations) > self.size:
            del self.situations[-1]

    def _find_i(self, activation):
        result_i = self.size + 666
        for i in range(len(self.situations)-1, 0, -1):
            if self.situations[i].activation > activation:
                result_i = i
                continue
        if result_i == self.size + 666:
            if len(self.situations) < self.size:
                result_i = len(self.situations)
        return result_i

    def show_first_n_situations(self):
        pass

    def show_all_situations(self):
        pass

    def save(self, filename='data.pickle'):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def show_hist(self):
        activations = []
        for situation in self.situations:
            activations.append(situation.activation)
        plt.hist(activations)
        plt.ylabel('how many times')
        plt.xlabel('lavel of activation (lower better)')
        plt.title("histogram")

class BankCreator:
    def __init__(self):
        pass

    def load_bank(self, filename='data.pickle'):
        with open(filename, 'rb') as f:
            bank = pickle.load(f)
            return bank
        return None

    def create_bank(self, images, kernel, banksize, bankname="noname.bank"):
        bank = SituationsBank(banksize, bankname)
        hside = int(kernel.shape[0]/2)
        for image in images:
            ilen = image.shape[0]
            for x_center in range(hside, ilen - hside - 1):
                for y_center in range(hside, ilen - hside - 1):
                    x_min = x_center - hside
                    x_max = x_center + hside + 1
                    y_min = y_center - hside
                    y_max = y_center + hside + 1
                    patch = image[x_min:x_max, y_min:y_max]
                    activation = np.linalg.norm(patch - kernel)
                    bank.try_add_situation(image, x_center, y_center, activation)
        return bank

if __name__ == "__main__":
    ###################################
    ##### get some matrix  ############
    from utils import get_mnist_number, get_all_images_np
    from first_device import FirstDeviceCreator
    mnist_number = 2
    image = get_mnist_number(mnist_number)
    devcr = FirstDeviceCreator(3, image)
    matrix = devcr.create_device()

    ###################################
    ##### get train mnist  ############
    imgs = get_all_images_np() [:6]

    ###################################
    ### make a bank and save it to file ##
    banksize = 60
    bank = BankCreator().create_bank(imgs, matrix, banksize)
    bank.save()
    del bank

    ##################################
    # resstore bank from file #######
    bank = BankCreator().load_bank()
    bank.show_hist()
    plt.show()

