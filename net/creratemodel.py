from net.model import SegModel
from monai.losses import DiceLoss
from torchmetrics import Accuracy,Dice
from torchmetrics.classification import BinaryJaccardIndex
import torch
import torch.nn as nn
import pytorch_lightning as pl
from copy import deepcopy
import pandas as pd
import sys
import numpy as np
import datetime

class CreateModel(pl.LightningModule):
    def __init__(self, args):
        super(CreateModel, self).__init__()
        self.model = SegModel(args.bert_type, args.vision_type, args.project_dim)
        self.lr = args.lr
        self.history = {}
        pos_weight = float(getattr(args, "bce_pos_weight", 1.0))
        self.dice_loss = DiceLoss(sigmoid=True)
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

        metrics_dict = {"acc":Accuracy(task='binary'),"dice":Dice(),"MIoU":BinaryJaccardIndex()}
        self.train_metrics = nn.ModuleDict(metrics_dict)
        self.val_metrics = deepcopy(self.train_metrics)
        self.test_metrics = deepcopy(self.train_metrics)
        self.save_hyperparameters()

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.model.parameters(),lr = self.hparams.lr)
        optimizer = torch.optim.AdamW(self.model.parameters(),lr = self.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max =200, eta_min=1e-6)
        return {"optimizer":optimizer,"lr_scheduler":lr_scheduler}
        
    def forward(self,x):
       return self.model.forward(x)


    def shared_step(self,batch,batch_idx):
        x, y = batch
        logits1, logits2 = self(x)
        y = y.float()
        loss1 = self.dice_loss(logits1, y) + self.bce_loss(logits1, y)
        loss2 = self.dice_loss(logits2, y) + self.bce_loss(logits2, y)
        probs1 = torch.sigmoid(logits1)
        probs2 = torch.sigmoid(logits2)
        probs = (probs1 + probs2) / 2.0
        loss = loss1 + loss2
        return {'loss': loss, 'preds': probs.detach(), 'y': y.detach()}
    
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch,batch_idx)
    
    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch,batch_idx)
    
    def test_step(self, batch, batch_idx):
        return self.shared_step(batch,batch_idx)
    
    def predict_step(self, batch, batch_idx):
        if isinstance(batch,list) and len(batch)==2:
            return self(batch[0])
        else:
            return self(batch)
        
    def shared_step_end(self,outputs,stage):
        metrics = self.train_metrics if stage=="train" else (
            self.val_metrics if stage=="val" else self.test_metrics)
        target = outputs['y'].int()
        for name in metrics:
            step_metric = metrics[name](outputs['preds'], target).item()
            if stage=="train":
                self.log(name,step_metric,prog_bar=True)
        return outputs["loss"].mean()
        
    def training_step_end(self, outputs):
        return {'loss':self.shared_step_end(outputs,"train")}
            
    def validation_step_end(self, outputs):
        return {'val_loss':self.shared_step_end(outputs,"val")}
            
    def test_step_end(self, outputs):
        return {'test_loss':self.shared_step_end(outputs,"test")}
            
    def shared_epoch_end(self,outputs,stage="train"):
        metrics = self.train_metrics if stage=="train" else (
            self.val_metrics if stage=="val" else self.test_metrics)
        
        epoch = self.trainer.current_epoch
        stage_loss = torch.mean(torch.tensor([t[(stage+"_loss").replace('train_','')] for t in outputs])).item()
        dic = {"epoch":epoch,stage+"_loss":stage_loss}
        
        for name in metrics:
            epoch_metric = metrics[name].compute().item() 
            metrics[name].reset()
            dic[stage+"_"+name] = epoch_metric 
        if stage!='test':
            self.history[epoch] = dict(self.history.get(epoch,{}),**dic)    
        return dic 
    
    def training_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs,stage="train")
        self.print(dic)
        dic.pop("epoch",None)
        for key, value in dic.items():
            self.log(key, value, logger=True, on_step=False, on_epoch=True)

    def validation_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs,stage="val")
        self.print_bar()
        self.print(dic)
        dic.pop("epoch",None)
        for key, value in dic.items():
            self.log(key, value, logger=True, prog_bar=(key in {"val_loss", "val_dice", "val_MIoU"}), on_step=False, on_epoch=True)
        
        #log when reach best score
        ckpt_cb = self.trainer.checkpoint_callback
        monitor = ckpt_cb.monitor 
        mode = ckpt_cb.mode 
        arr_scores = self.get_history()[monitor]
        best_score_idx = np.argmax(arr_scores) if mode=="max" else np.argmin(arr_scores)
        if best_score_idx==len(arr_scores)-1:   
            self.print("<<<<<< reach best {0} : {1} >>>>>>".format(
                monitor,arr_scores[best_score_idx]),file = sys.stderr)
    
    def test_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs,stage="test")
        dic.pop("epoch",None)
        self.print(dic)
        for key, value in dic.items():
            self.log(key, value, logger=True, on_step=False, on_epoch=True)
        
    def get_history(self):
        return pd.DataFrame(self.history.values()) 
    
    def print_bar(self): 
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.print("\n"+"="*80 + "%s"%nowtime)
