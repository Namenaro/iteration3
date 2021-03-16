from kernel_sutuations_bank import BankCreator, SituationsBank, Situation
from fixed_u_kernel_bank import FixedUKernelBank

def get_2_starts_np_dataset_from_bank():
    bank = BankCreator().load_bank("2_necks.bank")
    ux = 4
    uy = 0
    hside = 5
    ubank = FixedUKernelBank(bank, ux, uy, hside)
    print(ubank.get_raw_activations_matrixes().shape)
    ubank.show_first_n_situations()
    return ubank.get_raw_activations_matrixes()


if __name__ == "__main__":
    get_2_starts_np_dataset_from_bank()
