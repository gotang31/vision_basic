import torch
import pytorch_lightning as pl
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ResBck(pl.LightningModule):
    def __init__(self, model, num_classes, criterion , lr, **kwargs):
        super().__init__()
        self.model = model(num_classes, **kwargs)
        self.model.to(DEVICE)

        # train hyper-parameter
        self.criterion = criterion
        self.lr = lr

        # metrics
        metrics = MetricCollection([
            MulticlassAccuracy(num_classes), MulticlassPrecision(num_classes), 
            MulticlassRecall(num_classes), MulticlassF1Score(num_classes)])
        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')
        
    def forward(self, imgs):
        outputs = self.model(imgs)
        return outputs
    
    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = self.model(imgs)
        _, preds = torch.max(outputs.data, 1)
        loss = self.criterion(outputs, labels)

        metrics = self.train_metrics(preds, labels)
        metrics['loss'] = loss

        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return metrics

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = self.model(imgs)
        _, preds = torch.max(outputs.data, 1)
        loss = self.criterion(outputs, labels)

        metrics = self.valid_metrics(preds, labels)
        metrics['loss'] = loss

        self.valid_metrics.update(outputs, labels)
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return metrics
    
    def on_validation_epoch_end(self):
        output = self.valid_metrics.compute()
        self.log_dict(output)
        self.valid_metrics.reset()

        return output

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr = self.lr, momentum = 0.9, weight_decay = 0.0005)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        return {'optimizer' : optimizer, 'lr_scheduler' : lr_scheduler}
    