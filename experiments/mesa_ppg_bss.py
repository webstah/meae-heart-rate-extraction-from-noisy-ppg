import os
import matplotlib
import matplotlib.pyplot as plt

import numpy as np

import torch
from torch import nn

import pytorch_lightning as pl

import numpy as np

from utils.importer import import_model_from_config

from models.separation_loss import WeightSeparationLoss
from utils.plots import plot_signal, z_norm
from utils.signal_processing import detect_peaks

def zero_shared_weights(model_item, num_splits):
    for name, w in model_item.named_parameters():
        if name.split('.')[-1] == 'weight' and len(w.shape) == 3:
            w_l = w.shape[0]//num_splits
            w_w = w.shape[1]//num_splits
            for i in range(num_splits-1):
                w[i*w_l:(i+1)*w_l, (i+1)*w_w:] = w[i*w_l:(i+1)*w_l, (i+1)*w_w:]*0.0
                w[(i+1)*w_l:(i+2)*w_l, :(i+1)*w_w] = w[(i+1)*w_l:(i+2)*w_l, :(i+1)*w_w]*0.0


class Experiment(pl.LightningModule):
    def __init__(self, config, x_plot=None):
        super().__init__()
        
        self.hidden = config.hidden
        self.num_encoders = config.num_encoders
        
        self.input_signal_type = config.input_signal_type
        
        self.lr = config.lr
        self.lr_step_size = config.lr_step_size
        self.weight_decay = config.weight_decay
        
        self.sep_loss = config.sep_loss
        self.sep_lr = config.sep_lr
        
        self.zero_loss = config.zero_loss
        self.zero_lr = config.zero_lr

        self.z_loss = config.z_loss
        self.z_lr = config.z_lr

        self.fs = config.fs
        self.input_length = config.signal_duration
        
        self.padding = config.padding

        self.get_ecg = config.get_ecg

        self.x_plot = x_plot

        self.save_plots = config.save_plots
        self.plot_dir = config.plot_dir
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)
        self.plot_step = config.plot_step
        
        self.prepare_data()

        model_import = import_model_from_config(config)
        self.model = model_import(input_channels=config.input_channels, input_length=self.input_length, channels=config.channels, 
                                  hidden=config.hidden, use_weight_norm=config.use_weight_norm,
                                  num_encoders=config.num_encoders, norm_type=config.norm_type,
                                  input_padding=config.input_padding)

        if config.zero_mixing_weights:
            with torch.no_grad():   
                zero_shared_weights(self.model.decoder, config.num_encoders)

        self.loss = nn.BCEWithLogitsLoss()
        self.separation_loss = WeightSeparationLoss(config.num_encoders, config.sep_norm)

    def forward(self, x): 
        pred, _ = self.model(x)

        return pred

    def training_step(self, batch, batch_idx):
        if self.get_ecg:
            ecg, x = batch
        else:
            x = batch

        if self.padding > 0:
            pad_size = int(self.padding/2)
            x = torch.nn.functional.pad(x, (pad_size, pad_size), mode='constant', value=0)

        x_pred, z = self.model(x)
        recon_loss = self.loss(x_pred, x)
        self.log(f'recon_loss/train', recon_loss, on_step=True, 
                                    on_epoch=True, prog_bar=True)
        loss = recon_loss
        
        if self.z_loss:
            for z_i in z:
                loss = loss + self.z_lr * torch.mean(z_i**2)
        
        if self.sep_loss:
            sep_loss = self.separation_loss(self.model.decoder)
            self.log(f'sep_loss/train', sep_loss, on_step=False, 
                                    on_epoch=True, prog_bar=True)
            loss = loss + sep_loss*self.sep_lr
        
        if self.zero_loss:
            z_zeros = [torch.zeros(1, self.hidden//self.num_encoders, z[0].shape[-1]).to(self.device) for _ in range(self.num_encoders)]
            x_pred_zeros = self.model.decode(z_zeros, True)
            zero_recon_loss = self.loss(x_pred_zeros, torch.zeros_like(x_pred_zeros))
            loss = loss + zero_recon_loss*self.zero_lr
            self.log(f'zero_recon_loss/train', zero_recon_loss, on_step=False, 
                                on_epoch=True, prog_bar=True)
            
        if batch_idx % 50 == 0:
            plot = plot_signal(x, torch.sigmoid(x_pred))
            self.logger.experiment.add_image('images/train_plot', plot, batch_idx)
        
        if batch_idx % self.plot_step == 0 and self.save_plots:
            self.save_inference_samples(self.x_plot)
            self.save_weight_visualizations()
            
        return loss
            
    def validation_step(self, batch, batch_idx):
        if self.get_ecg:
            ecg, x = batch
        else:
            x = batch
        if self.padding > 0:
            pad_size = int(self.padding/2)
            x = torch.nn.functional.pad(x, (pad_size, pad_size), mode='constant', value=0)

        x_pred, z = self.model(x)
        recon_loss = self.loss(x_pred, x)
        self.log(f'recon_loss/val', recon_loss, on_step=True, 
                                    on_epoch=True, prog_bar=True)
                
        return recon_loss
    
    def _local_normalize(self, x):
        x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
        
        return x
    
    def _numpy_normalize(self, x):
        # Normalize each column to the range [0, 1]
        min_vals = np.min(x, axis=1, keepdims=True)  # Shape (8, 1)
        max_vals = np.max(x, axis=1, keepdims=True)  # Shape (8, 1)

        range_vals = max_vals - min_vals
        # range_vals[range_vals == 0] = 1  # Handle constant rows

        x = (x - min_vals) / range_vals
        return x
    
    def save_inference_samples(self, patient):
        step = self.global_step
        
        save_name = os.path.join(self.plot_dir, f'images_{step}.png')
        
        ecg = [torch.tensor(x_i[:, 0]).unsqueeze_(0).to(self.device) for x_i in patient]
        mask = [torch.tensor(x_i[:, 1]).unsqueeze_(0).to(self.device) for x_i in patient]
        x = [torch.tensor(x_i[:, 2]).unsqueeze_(0).to(self.device) for x_i in patient]

        ecg = torch.stack(ecg, dim=0).type(torch.float32).to(self.device)
        ecg = self._local_normalize(ecg)
        mask = torch.stack(mask, dim=0).type(torch.float32).to(self.device)
        mask = self._local_normalize(mask)
        x = torch.stack(x, dim=0).type(torch.float32).to(self.device)
        x = self._local_normalize(x)

        if self.padding > 0:
            pad_size = int(self.padding/2)
            ecg = torch.nn.functional.pad(ecg, (pad_size, pad_size), mode='constant', value=0)
            mask = torch.nn.functional.pad(mask, (pad_size, pad_size), mode='constant', value=0)
            x = torch.nn.functional.pad(x, (pad_size, pad_size), mode='constant', value=0)

        with torch.no_grad():
            x_pred, _ = self.model(x)

        self.model.eval()
        with torch.no_grad():
            z = self.model.encode(x)
            zeros = torch.zeros_like(z[0])
            y = []
            for z_a in range(len(z)):
                z_pad = [zeros for _ in range(len(z))]
                z_pad[z_a] = z[z_a]
                y_i = self.model.decode(z_pad)
                y_i = torch.sigmoid_(y_i).squeeze_()
                y.append(y_i.cpu().numpy())

        clip = 10
        if self.padding > 0:
            clip = int(self.padding/2)
                
        title = f'Step: {step}'
            
        colors = ['#24e254', '#12e29f', '#00e1e9', '#05b5f4', '#0a89ff', '#535def', '#9c31df', '#c91976', '#f5000c',
                  '#24e254', '#12e29f', '#00e1e9', '#05b5f4', '#0a89ff', '#535def', '#9c31df', '#c91976', '#f5000c'] ##https://coolors.co/24e254-12e29f-00e1e9-05b5f4-0a89ff-535def-9c31df-c91976-f5000c
        
        line_width = 1.5
        idx = 0

        matplotlib.rcParams['axes.linewidth'] = line_width
        matplotlib.rcParams['ytick.major.width'] = line_width
        matplotlib.rcParams['xtick.major.width'] = line_width
        default_c = '#434343'
        matplotlib.rcParams.update({'text.color' : f'{default_c}',
                                    'axes.labelcolor' : f'{default_c}',
                                    'axes.edgecolor' : f'{default_c}',
                                    'xtick.color' : f'{default_c}', 
                                    'ytick.color' : f'{default_c}'})

        plt.figure(figsize=(8,13))
        plt.title(title, pad=125)
        plt.tick_params(axis='both', direction='in')
        plt.grid(True, alpha=0.35, linewidth=line_width)
        plt.margins(x=0)
        
        # plot
        ecg_cpu = ecg[idx].detach().squeeze().cpu().numpy()
        ecg_cpu = z_norm(ecg_cpu[clip:-clip])
        mask_cpu = mask[idx].detach().squeeze().cpu().numpy()
        mask_cpu = z_norm(mask_cpu[clip:-clip])
        x_pred_cpu = x_pred[idx].detach().squeeze().cpu().numpy()
        x_pred_cpu = z_norm(x_pred_cpu[clip:-clip])
        x_cpu_ = x[idx].detach().squeeze().cpu().numpy()
        x_cpu = z_norm(x_cpu_[clip:-clip])
        plt.plot(np.arange(len(ecg_cpu)), z_norm(ecg_cpu)+8, c='#787878', linewidth=line_width)
        plt.plot(np.arange(len(x_cpu)), x_cpu, label=f'{self.input_signal_type} (Input)', c='#787878', linewidth=line_width)

        for i, y_i in enumerate(y):
            y_i_plot = z_norm(y_i[idx][clip:-clip])
            plt.plot(np.arange(len(y_i_plot)), y_i_plot-8*(i+1), label=f'Enc. {i} Source Pred', alpha=0.9, c=colors[i+1], linewidth=line_width)

        plt.plot(np.arange(len(x_pred_cpu)), x_pred_cpu-8*(i+2), label=f'{self.input_signal_type} (Recon)', c='#787878', linewidth=line_width)

        dummy = detect_peaks(mask_cpu, self.fs, 0.66, 3)
        for dummy_peaks in dummy:
            plt.axvline(x=dummy_peaks, color='purple', linestyle=':')

        plt.subplots_adjust(top=0.8, bottom=0.05, left=0.05, right=0.95)
        plt.yticks([])
        plt.legend(loc='lower center', frameon=False, fancybox=True, ncol=2, 
           bbox_to_anchor=(0.5, 1.015), fontsize='medium', markerfirst=False)
        plt.savefig(save_name)
        plt.close()

        self.model.train()
    
    def save_weight_visualizations(self):
        step = self.global_step
        title = f'Step: {step}'
        save_name = os.path.join(self.plot_dir, f'weights_{step}.png')

        weights = []
        shapes = []
        for name, w in self.model.decoder.named_parameters():
            if name.split('.')[-1] == 'weight' and len(w.shape) == 3:
                weights.append(torch.sum(torch.abs(w.detach()), dim=-1).cpu().numpy())
                shapes.append(tuple(w.shape))
        
        fig, axs = plt.subplots(1, len(weights), figsize=(14,4.25))
        fig.suptitle(title)
        fig.tight_layout()
        for ax, weights, title in zip(axs, weights, shapes):
            ax.set_title(title)
            ax.axis('off')
            ax.imshow(weights, aspect='auto')
            
        plt.savefig(save_name)
        plt.close()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_step_size, gamma=0.1, last_epoch=-1),
            'name': 'step_lr_scheduler',
         }
   
        return [optimizer], [scheduler]
    