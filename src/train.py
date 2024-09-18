import torch
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

# basic libarys
import os
import glob
import numpy as np
from datetime import datetime
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from omegaconf import OmegaConf
from abc import ABC, abstractmethod
import hydra
import math

from utils import s2toRGB

class BaseTrainer(ABC):
    
    def __init__(self,config):
        
        # some variables we need
        self.config = config
        self.globalstep = 0
        self.loss = 0

        if config.gpu_idx != "cpu":
            self.cuda = True
        else:
            self.cuda = False

        # seeding
        np.random.seed(config.seed)
        if self.cuda:
            torch.cuda.manual_seed(config.seed)
        else:
            torch.manual_seed(config.seed)

        # make folders 
        date_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        self.savepath = os.path.join(config.outputpath, config.experimentname, date_time)
        self.checkpoint_dir = os.path.join(self.savepath,"model_checkpoints")
        os.makedirs(self.savepath, exist_ok = True)
        os.makedirs(self.checkpoint_dir, exist_ok = True)

        # store the config file that has been used
        # to make it more traceable
        with open(os.path.join(self.savepath,"used_parameters.json"), 'w') as f:
            json.dump(OmegaConf.to_container(config), f) 

        # sice we have constant imput size in each
        # iteration we can set up benchmark mode
        torch.backends.cudnn.benchmark = True

        # loading datasets 
        self.train_dataset = hydra.utils.instantiate(config.dataset, train_val_test_key="train")

        self.val_dataset = hydra.utils.instantiate(config.dataset, train_val_test_key="val")


        self.train_data_loader = hydra.utils.instantiate(config.dataloader,
                                                         shuffle=True,
                                                         dataset = self.train_dataset,
                                                         )
        
        self.val_data_loader = hydra.utils.instantiate(config.dataloader,
                                                       shuffle=False,
                                                       dataset = self.val_dataset,
                                                       )

        model: torch.nn.Module = hydra.utils.instantiate(config.model)

        if config.compile:
            self.model = torch.compile(model)
        else:
            self.model =  model

        if config.resume:
            checkpoint = torch.load(config.resumeCheckpoint)
            self.init_epoch = checkpoint["epoch"]
            self.init_global_step = checkpoint["global_step"]
            self.init_loss = checkpoint["loss"]
            try:
                model.load_state_dict(checkpoint["model_state_dict"])
            except:
                mod_state_dict = {key.replace("_orig_mod.",""): val for key, val in checkpoint["model_state_dict"].items()}
                model.load_state_dict(mod_state_dict)
            print(f"Model loaded from checkpoint: {config.resumeCheckpoint}")

        summary(self.model)

        if self.cuda:
            self.model = self.model.cuda()

        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=config.optimizer.base_learning_rate * config.dataloader.batch_size / 256,
                                           betas=(0.9, 0.95),
                                           weight_decay=config.optimizer.weight_decay)
        
        lr_func = lambda epoch: min((epoch + 1) / (config.scheduler.warmup_epoch + 1e-8),
                                    0.5 * (math.cos(epoch / config.nEpochs * math.pi) + 1))
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_func, verbose=True)

        # gradient scaler
        self.scaler = torch.cuda.amp.GradScaler()

        # setup tensorboard
        self.TB_writer = SummaryWriter(log_dir=os.path.join(self.savepath,"logs"))
        self.TB_writer.add_text("Parameters",str(config))

        # plot the learning rate
        f, ax1 = plt.subplots(1,1,figsize=(8,4))
        x = np.arange(0,config.nEpochs+1)
        ax1.plot(x,[lr_func(xx)*config.optimizer.base_learning_rate for xx in x])
        self.TB_writer.add_figure(f"LR", f, global_step=0)
        plt.close()

        self.config = config
        
        # when to validate:
        # to make it comparable we dont do it after x batches rather after x sampels
        self.nextValidationstep = self.config.validation_every_N_sampels
        self.nextPlottingstep = self.config.plotting_every_N_sampels

        self.best_metric_saveCrit = 10000
            
    def fit(self):
        
        self.current_epoch = 1

        if self.config.resume:
            self.current_epoch = self.init_epoch
            self.globalstep = self.init_global_step
            self.loss = self.init_loss
            if not self.config.nEpochs > self.current_epoch:
                raise ValueError(f"Model checkpoint has been trained for {self.current_epoch} and final number of epochs is {self.config.nEpochs}")
        
        self.model.train()

        for epoch in range(self.current_epoch, self.config.nEpochs + 1):

            self._train_one_epoch()

            if not self.config.validate_after_every_n_epoch == -1:
                if epoch % self.config.validate_after_every_n_epoch == 0:
                    self.model.eval() # todo: be carefull projection head
                    self._validate()
                    self.model.train()

            # if lr sceduler is active
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                    if self.current_epoch - 2 < self.config.lrscheduler.T_max:
                         self.scheduler.step()
                else:
                    self.scheduler.step()

            if self.current_epoch in self.config.special_save_nEpoch:
                self._save_checkpoint(nameoverwrite = f"special_save_E{self.current_epoch}")

            self.TB_writer.add_scalar(f"lr", self.optimizer.param_groups[0]["lr"], global_step=self.globalstep)
            self.TB_writer.add_scalar(f"lr/over_epoch", self.optimizer.param_groups[0]["lr"], global_step=self.current_epoch)
            
            self.current_epoch += 1

        self.TB_writer.flush()

        return None
   
    def _train_one_epoch(self):
         
        pbar_train = tqdm(total=len(self.train_data_loader), desc=f"EPOCH: {self.current_epoch}",leave=False)
        
        for batch_idx, batch in enumerate(self.train_data_loader, start=1): 
            
            self._train_one_batch(batch)
            
            if not self.config.plotting_every_N_sampels == -1:
                if (self.globalstep * self.config.dataloader.batch_size) >= self.nextPlottingstep:
                    print("Plotting Exampels",self.globalstep)
                    self.model.eval()
                    self._plot()
                    self._validate()                
                    self.nextPlottingstep += self.config.plotting_every_N_sampels
                    self.model.train()

            pbar_train.update()
        pbar_train.close()        
        
        return None

    @abstractmethod
    def _train_one_batch(self, batch):
        """
        the core of the training... this you have to specify for each
        trainer seperately
        """
        pass

    @abstractmethod
    def _validate(self, batch):
        """
        the core of the training... this you have to specify for each
        trainer seperately
        """
        pass
        
    @abstractmethod
    def _plot(self):
        """
        the core of the training... this you have to specify for each
        trainer seperately
        """
        pass

    def _save_checkpoint(self,nameoverwrite = ""):
        
        # only delete if nameoverwrite is "" so that
        # at last epoch we dont delte the inbetween checkpoint
        if nameoverwrite == "":
            all_models_there = glob.glob(os.path.join(self.checkpoint_dir,"checkpoint*.pt"))
            # there should be none or one model
            if not len(all_models_there) in [0,1]:
                raise ValueError(f"There is more then one model in the checkpoint dir ({len(self.checkpoint_dir)})... seems wrong")
            else:
                for model in all_models_there:
                    os.remove(model)
        
        if nameoverwrite == "":
            outputloc =  os.path.join(self.checkpoint_dir,f"checkpoint_{self.current_epoch}_{self.globalstep}_{self.best_metric_saveCrit}.pt")
        else:
            outputloc =  os.path.join(self.checkpoint_dir,f"{nameoverwrite}.pt")

        torch.save({
            'epoch': self.current_epoch,
            'global_step': self.globalstep,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss,
            },
           outputloc)
          
        return None
              
    def finalize(self):

        # save hyperparameters
        self.TB_writer.add_hparams(
                    {"lr": self.config.optimizer.lr,
                     "bsize": self.config.dataloader.batch_size,
                    },
                    {
                    "hparam/IoU": self.best_metric_saveCrit,
                    },
                    run_name="hparams"
                )

        self.TB_writer.close()
        
        self._save_checkpoint("state_at_finalize")
        
        return None


class SenPaMAE_Trainer(BaseTrainer):

    def __init__(self, config):
        super().__init__(config)        
        pass
    
    def _train_one_batch(self, batch):
        
        data = batch["data"]
        rf = batch["rfs"]
        gsd = batch["gsds"]

        if self.cuda:
            data = data.cuda()
            rf = rf.cuda()
            gsd = gsd.cuda()

        with torch.cuda.amp.autocast():
            predicted_img, mask, lenFeat = self.model(data,rf,gsd)
            self.loss = torch.mean(torch.abs((predicted_img - data)) * mask) #/ self.mask_ratio
        
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(self.loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.globalstep += 1

        # write current train loss to tensorboard at every step
        self.TB_writer.add_scalar("train/lenFeat", lenFeat, global_step=self.globalstep)
        self.TB_writer.add_scalar("train/loss", self.loss, global_step=self.globalstep)
        
        return None
    
    def _plot(self):

        with torch.no_grad():
            
            for batch_idx, batch in enumerate(self.val_data_loader):

                data = batch["data"]
                rf = batch["rfs"]
                gsd = batch["gsds"]

                if self.cuda:
                    data = data.cuda()
                    rf = rf.cuda()
                    gsd = gsd.cuda()

                predicted_img, mask, lenFeat = self.model(data,rf,gsd)

                # just grab the first img in the batch
                data = data[0]
                sensor = batch["sensor"][0]
                predicted_img = predicted_img[0]
                mask = mask[0]
                rf = rf[0]
                gsd = gsd[0]

                # turn everything into numpy
                data = data.cpu().detach().numpy()
                predicted_img = predicted_img.cpu().detach().numpy()
                mask = mask.cpu().detach().numpy().astype("uint8")

                reconstructed_img = predicted_img.copy()
                reconstructed_img[~mask.astype(bool)] = data[~mask.astype(bool)]

                num_channels = predicted_img.shape[0]
                width = 2.5
                f, loax = plt.subplots(num_channels,5,figsize=(5*width,num_channels*width))

                for ijk in range(num_channels):
                    
                    vmin = data[ijk,:,:].min()
                    vmax = data[ijk,:,:].max()

                    loax[ijk,0].imshow( mask[ijk,:,:] , vmin=0, vmax=1)
                    loax[ijk,1].imshow( predicted_img[ijk,:,:], vmin=vmin, vmax=vmax)
                    loax[ijk,2].imshow( reconstructed_img[ijk,:,:], vmin=vmin, vmax=vmax)
                    loax[ijk,3].imshow( data[ijk,:,:], vmin=vmin, vmax=vmax)
                    loax[ijk,4].plot( rf[ijk,:].cpu().detach().numpy())
                    loax[ijk,4].set_title(str(gsd[ijk].item()))

                    for ii in range(5):
                        loax[ijk,ii].axis("off")

                f.suptitle(sensor)

                plt.tight_layout()
                self.TB_writer.add_figure(f"example_output_{batch_idx}", f, global_step=self.globalstep)
                plt.close()
                
                if batch_idx == 30: break

    def _validate(self):
    
        with torch.no_grad():
            
            losses = []
            for batch_idx, batch in enumerate(self.val_data_loader):

                data = batch["data"]
                rf = batch["rfs"]
                gsd = batch["gsds"]

                if self.cuda:
                    data = data.cuda()
                    rf = rf.cuda()
                    gsd = gsd.cuda()

                predicted_img, mask, lenFeat = self.model(data,rf,gsd)
                losses.append(torch.mean(torch.abs((predicted_img - data)) * mask))

                if batch_idx == 200: break

        self.TB_writer.add_scalar("val/loss", torch.stack(losses).mean(), global_step=self.globalstep)

        return None
    
