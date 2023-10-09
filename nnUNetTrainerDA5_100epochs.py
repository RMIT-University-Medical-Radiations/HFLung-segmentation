import torch
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation import nnUNetTrainerDA5


class nnUNetTrainerDA5_100epochs(nnUNetTrainerDA5):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 100
