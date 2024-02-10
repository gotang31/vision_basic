import torch
import pytorch_lightning as pl


DEVICE =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DETR(pl.LightningModule):
    def __init__(self, model, lr = 1e-4, lr_backbone = 1e-5, weight_decay = 1e-4):
        super().__init__()
        self.model = model

        # train hyper-parameter
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        
    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
    
    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict
    
    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)

        self.log("training_loss", loss, on_step=True, on_epoch=True, logger=True)
        for k,v in loss_dict.items():
            self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)

        self.log("validation_loss", loss,on_step=True, on_epoch=True, logger=True)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item())

        return loss

    def configure_optimizers(self):
        param_dicts = [
            {
                "params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},  # backbone 아닌경우 learning rate = lr
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],   # backbone일 경우 learning rate = lr_backbone
                "lr": self.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        return {'optimizer' : optimizer, 'lr_scheduler' : lr_scheduler}