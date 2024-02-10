import sys
import argparse
import json
import torch
import pytorch_lightning as pl
from transformers import DetrForObjectDetection, DetrImageProcessor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers
from train import DETR, DEVICE
from dataset import DetrData

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
            default=1e-4,
            type=float,)
        
        parser.add_argument('--bck-learning-rate',
            help='Learning rate for training',
            default=1e-5,
            type=float,)
        
        parser.add_argument('--weight-decay',
            help='Learning rate for training',
            default=1e-4,
            type=float,)
        
        parser.add_argument('--num-classes',
            help="Number of classes which will be clssified.",
            default = 91,
            type = int,)
        
        parser.add_argument('--dir',
            help="Directory where you save best model and logs",
            default = 'logs/',
            type = str,)

        parser.add_argument('--jsonfile-name',
            help="File name that you save parameter inputs by json format",
            default = 'default.json',
            type = str,)

        parser.add_argument('--train-img-datasets',
            help="Directory where you made the train datasets of dataframe format by default",
            default = 'train2017',
            type = str,)
        
        parser.add_argument('--train-ann-datasets',
            help="Directory where you made the train datasets of dataframe format by default",
            default = 'annotations/instances_train2017.json',
            type = str,)

        parser.add_argument('--val-img-datasets',
            help="Directory where you made the train datasets of dataframe format by default",
            default = 'val2017',
            type = str,)
        
        parser.add_argument('--val-ann-datasets',
            help="Directory where you made the train datasets of dataframe format by default",
            default = 'annotations/instances_val2017.json',
            type = str,)      

        self.args = parser.parse_args(sys_argv)
        self.model = self.init_model()
        self.processor = self.assign_transform(DetrImageProcessor, "facebook/detr-resnet-50")

        # save hyperparameter 
        with open(f'{self.args.dir}/{self.args.jsonfile_name}', 'w') as fp:
            json.dump(vars(self.args), fp)
    
    def init_model(self):
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50",
                                                             revision="no_timm",
                                                             num_labels = self.args.num_classes,
                                                             ignore_mismatched_sizes=True)

        model = DETR(model, lr = self.args.learning_rate, lr_backbone = self.args.bck_learning_rate, weight_decay = self.args.weight_decay)
        model.to(DEVICE)
        
        return model
    
    def assign_transform(self, processor, hubpath):
        image_processor = processor.from_pretrained(hubpath)
        
        return image_processor

    def collate_fn(self, batch):
    # DETR 논문에서 훈련 중 다양한 이미지 크기를 사용했기 때문에 직접 배치를 구성할 수 없음. 
    # 가장 큰 해상도(min:800, max: 1333)로 이미지를 패딩하고 실제 픽셀과 패딩 픽셀을 구분하는 이진 pixel_mask를 생성.
 
        pixel_values = [item[0] for item in batch]
        encoding = self.processor.pad(pixel_values, return_tensors="pt")
        labels = [item[1] for item in batch]
        return {'pixel_values': encoding['pixel_values'],
                'pixel_mask': encoding['pixel_mask'],
                'labels': labels}

    def train_loader(self):
        train_dl = torch.utils.data.DataLoader(DetrData(self.args.train_img_datasets, self.args.train_ann_datasets, self.processor),
                                        batch_size = self.args.batch_size,
                                        shuffle = True,
                                        collate_fn = self.collate_fn,
                                        pin_memory= True if torch.cuda.is_available() else False,
                                        num_workers = self.args.num_workers)
        return train_dl
        
    def val_loader(self):
        val_dl = torch.utils.data.DataLoader(DetrData(self.args.val_img_datasets, self.args.val_ann_datasets, self.processor),
                                        batch_size = self.args.batch_size,
                                        shuffle = False,
                                        collate_fn = self.collate_fn,
                                        pin_memory= True if torch.cuda.is_available() else False,
                                        num_workers = self.args.num_workers)
        return val_dl
        
    def main(self):
        checkpoint_callback = ModelCheckpoint(dirpath = self.args.dir, filename = 'detr-{epoch:02d}-{val_loss:.2f}', save_top_k = 3, save_last = True, monitor = 'val_loss')
        tb_logger = loggers.TensorBoardLogger(save_dir = self.args.dir)
        trainer = pl.Trainer(max_epochs = self.args.epochs, gradient_clip_val=0.1, logger = tb_logger, callbacks = checkpoint_callback, accelerator='gpu')

        trainer.fit(self.model, train_dataloaders = self.train_loader(), val_dataloaders = self.val_loader())

if __name__ == "__main__": 

    train = TrainingApp()
    train.main()