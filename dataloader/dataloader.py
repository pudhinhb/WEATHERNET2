import torch
from torch.utils.data import DataLoader

class WADSLoader(DataLoader):
    def __init__(self, **hparams):
        self.hparams = hparams
        
    def train_dataloader(self):
        dataloader = DataLoader(self.hparams['train_ds'],
                                batch_size=self.hparams['train_batch_size'],
                                shuffle=self.hparams['train_shuffle'],
                               )
        return dataloader
    
    def validation_dataloader(self):
        dataloader = DataLoader(self.hparams['valid_ds'],
                                batch_size=self.hparams['valid_batch_size'],
                                shuffle=self.hparams['valid_shuffle'],
                               )
        return dataloader
    
    def test_dataloader(self):
        dataloader = DataLoader(self.hparams['test_ds'],
                                batch_size=self.hparams['test_batch_size'],
                                shuffle=self.hparams['test_shuffle'],
                               )
        return dataloader