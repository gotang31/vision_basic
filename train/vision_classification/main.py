import sys
import sys
import argparse
import json
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers
from model import SimRes50
from train import ResBck, DEVICE
from dataset import ResData, collate_fn

class TrainingApp:
    def __init__(self, sys_argv = None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]
        
        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
            help='Batch size to use for training',
            default=64,
            type=int,)
        
        parser.add_argument('--num-workers',
            help='Number of worker processes for background data loading',
            default=8,
            type=int,)
        
        parser.add_argument('--epochs',
            help='Number of epochs to train',
            default=10,
            type=int,)
        
        parser.add_argument('--learning-rate',
            help='Learning rate for training',
            default=0.001,
            type=float,)
        
        parser.add_argument('--num_classes',
            help="Number of classes which will be clssified.",
            default = 80,
            type = int,)
        
        parser.add_argument('--mode',
            help="whether an elder fc layer are replaced with a new fc layer or not.",
            default = False,
            type = bool,)
        
        parser.add_argument('--freeze-depth',
            help="To the extent which you want freeze from last to ascending layer",
            default = 1,
            type = int,)
        
        parser.add_argument('--dir',
            help="Directory where you save best model and logs",
            default = 'logs/',
            type = str,)

        parser.add_argument('--jsonfile-name',
            help="File name that you save parameter inputs by json format",
            default = 'default.json',
            type = str,)

        parser.add_argument('--train-datasets',
            help="Directory where you made the train datasets of dataframe format by default",
            default = 'train2017.csv',
            type = str,)

        parser.add_argument('--val-datasets',
            help="Directory where you made the validation datasets of dataframe format by default",
            default = 'val2017.csv',
            type = str,)        

        self.args = parser.parse_args(sys_argv)
        self.model = self.init_model()

        # save hyperparameter 
        with open(f'{self.args.dir}/{self.args.jsonfile_name}', 'w') as fp:
            json.dump(vars(self.args), fp)
    
    def init_model(self):
        criterion = nn.CrossEntropyLoss()

        model = ResBck(SimRes50, self.args.num_classes, criterion = criterion,lr = self.args.learning_rate, mode = self.args.mode, freeze_depth = self.args.freeze_depth)
        model.to(DEVICE)
        
        return model

    def train_loader(self):
        train_df = pd.read_csv(self.args.train_datasets)
        train_inds = list(range(len(train_df)))
        train_dl = torch.utils.data.DataLoader(ResData(train_df, train_inds, 'train'),
                                        batch_size = self.args.batch_size,
                                        shuffle = True,
                                        collate_fn = collate_fn,
                                        pin_memory= True if torch.cuda.is_available() else False,
                                        num_workers = self.args.num_workers)
        return train_dl
        
    def val_loader(self):
        val_df = pd.read_csv(self.args.val_datasets)
        val_inds = list(range(len(val_df)))
        val_dl = torch.utils.data.DataLoader(ResData(val_df, val_inds, 'val'),
                                        batch_size = self.args.batch_size,
                                        shuffle = False,
                                        collate_fn=collate_fn,
                                        pin_memory= True if torch.cuda.is_available() else False,
                                        num_workers = self.args.num_workers)
        return val_dl
        
    def main(self):
        checkpoint_callback = ModelCheckpoint(dirpath = self.args.dir, filename = 'resnet50-{epoch:02d}-{val_loss:.2f}', save_top_k = 3, save_last = True, monitor = 'val_loss')
        tb_logger = loggers.TensorBoardLogger(save_dir = self.args.dir)
        trainer = pl.Trainer(max_epochs = self.args.epochs, gradient_clip_val=0.1, logger = tb_logger, callbacks = checkpoint_callback, accelerator='gpu')

        trainer.fit(self.model, train_dataloaders = self.train_loader(), val_dataloaders = self.val_loader())

if __name__ == "__main__": 

    train = TrainingApp()
    train.main()