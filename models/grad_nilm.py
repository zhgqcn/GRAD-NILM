import torch
from abc import *
import numpy as np
from torch import nn
import os, json, math
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F
from .gat import GraphAttentionNetwork
from utils.metrics import mean_metrics
from torch.utils.tensorboard import SummaryWriter


def get_first(list):
    pos = 0
    for i in range(len(list)):
        if list[i] == 1:
            pos = i
            break
    return pos
    
def get_last(list):
    pos = 0
    for i in range(len(list)):
        if list[i] == 1:
            pos = i
    return pos

#################################################
# generate co-occurrence probability graph label
#################################################
def get_adj_batch_dir(batch_status):
    graph = None
    for i in range(batch_status.size()[0]):
        if i == 0:
            graph = get_adj(batch_status[i]).unsqueeze(0) + get_adj_distance(batch_status[i]).unsqueeze(0)
        else:
            temp = get_adj(batch_status[i]).unsqueeze(0) + get_adj_distance(batch_status[i]).unsqueeze(0)
            graph = torch.cat([graph, temp], dim=0)        
    return graph

########################################
# soft dependency from temporal context
########################################
def get_adj_distance(state):
    state = state.T
    nodes  = state.size()[0]
    length = state.size()[1]
    graph = torch.zeros((5, 5))
    for i in range(nodes):
        if state[i].sum() == 0:
            continue
        else:
            graph[i][i] = 0                  #  no self-loop

        end = get_last(state[i])
        for j in range(nodes):
            if i != j and state[j].sum() != 0:
                start = get_first(state[j])
                graph[i][j] = round(abs(end - start) / length, 2)
    return graph

######################################################
# Hard dependency is used to solve the sparse problem
######################################################
def get_adj(status):
    status = status.T
    graph = torch.zeros((5, 5))
    for i in range(status.size()[0]):
        a = status[i].sum()
        if i == 0 and a >= 2.0:
            graph[i][0] = graph[i][1] = graph[i][2] = graph[i][3] = graph[i][4] = 1  
        elif i == 1 and a >= 5.0:
            graph[i][0] = graph[i][1] = graph[i][2] = graph[i][3] = graph[i][4] = 1
        elif i == 2 and a >= 3.0:
            graph[i][0] = graph[i][1] = graph[i][2] = graph[i][3] = graph[i][4] = 1
        elif i == 3 and a >= 5.0:
            graph[i][0] = graph[i][1] = graph[i][2] = graph[i][3] = graph[i][4] = 1
        elif i == 4 and a >= 2.0:
            graph[i][0] = graph[i][1] = graph[i][2] = graph[i][3] = graph[i][4] = 1
        else:
            pass
        graph[0][0] = graph[1][1] = graph[2][2] = graph[3][3] = graph[4][4]  = 0       # no self-loop
    return graph


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Encoder, self).__init__()
        self.sequential = nn.Sequential(
            Conv1D(in_channels, out_channels, kernel_size, stride=stride),
            Conv1D(out_channels, out_channels, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.PReLU(),
            Conv1D(out_channels, out_channels, kernel_size=3, stride=1, dilation=2, padding=2),
            nn.PReLU(),
            Conv1D(out_channels, out_channels, kernel_size=3, stride=1, dilation=4, padding=4),
            nn.PReLU(),
            Conv1D(out_channels, out_channels, kernel_size=3, stride=1, dilation=8, padding=8),
            nn.PReLU()
        )
        
    def forward(self, x):
        '''
           x: [B, T]
           out: [B, N, T]
        '''
        x = self.sequential(x)
        return x


class Decoder(nn.Module):
    '''
        Decoder
        This module can be seen as the gradient of Conv1d with respect to its input. 
        It is also known as a fractionally-strided convolution 
        or a deconvolution (although it is not an actual deconvolution operation).
    '''

    def __init__(self, N, kernel_size=16, stride=16 // 2):
        super(Decoder, self).__init__()
        self.sequential = nn.Sequential(
            nn.ConvTranspose1d(N, N, kernel_size=3, stride=1, dilation=8, padding=8),
            nn.PReLU(),
            nn.ConvTranspose1d(N, N, kernel_size=3, stride=1, dilation=4, padding=4),
            nn.PReLU(),
            nn.ConvTranspose1d(N, N, kernel_size=3, stride=1, dilation=2, padding=2),
            nn.PReLU(),
            nn.ConvTranspose1d(N, N, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.PReLU(),
            nn.ConvTranspose1d(N, 1, kernel_size=kernel_size, stride=stride, bias=True)
        )

    def forward(self, x):
        """
        x: N x L or N x C x L
        """
        x = self.sequential(x)
        if torch.squeeze(x).dim() == 1:
            x = torch.squeeze(x, dim=1)
        else:
            x = torch.squeeze(x)

        return x


class Conv1D(nn.Conv1d):
    '''
       Applies a 1D convolution over an input signal composed of several input planes.
    '''

    def __init__(self, *args, **kwargs):
        super(Conv1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        # x: N x C x L
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        if squeeze:
            x = torch.squeeze(x)
        return x


class GRADNILM(nn.Module):
    def __init__(self,
                 N=128,
                 L=16,
                 B=128,
                 num_appliances=5,
                 activate="relu"):
        super(GRADNILM, self).__init__()
        
        self.encoder = Encoder(1, N, L, stride=L // 2)
        self.BottleN_S = Conv1D(N, 5, 1)
        self.gat = GraphAttentionNetwork(15, 512, 128 * 3, 0.5, 0.2, 2).cuda()
        self.gen_masks = Conv1D(B, num_appliances * N, 1)
        self.decoder_y = Decoder(N, L, stride=L//2)
        self.decoder_s = Decoder(N, L, stride=L//2)
        # activation function
        active_f = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'softmax': nn.Softmax(dim=0)
        }
        self.activation_type = activate
        self.activation = active_f[activate]
        self.num_apps = num_appliances
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if x.dim() >= 3:
            raise RuntimeError(
                "{} accept 1/2D tensor as input, but got {:d}".format(
                    self.__name__, x.dim()))
        if x.dim() == 1:
            x = torch.unsqueeze(x, 0)
        
        w = self.encoder(x)                                 # 32 128 15
        e = self.BottleN_S(w)                               # 32 5 15
        G_power_logits, G_mid = self.gat(e)                 # 32 5 128 * 3
        G_power_logits = G_power_logits.view(-1, 128, 15)
        m = self.gen_masks(G_power_logits)                  # 32 2560 15
        m = torch.chunk(m, chunks=self.num_apps, dim=1)
        m = self.activation(torch.stack(m, dim=0))          # 5 32 128 15
        d = [w * m[i] for i in range(self.num_apps)]
        y = [self.decoder_y(d[i]) for i in range(self.num_apps)]
        y = torch.cat([y[0].unsqueeze(-1), y[1].unsqueeze(-1), y[2].unsqueeze(-1), y[3].unsqueeze(-1), y[4].unsqueeze(-1)], dim=-1)
        s = [self.decoder_s(d[i]) for i in range(self.num_apps)]
        s = torch.cat([s[0].unsqueeze(-1), s[1].unsqueeze(-1), s[2].unsqueeze(-1), s[3].unsqueeze(-1), s[4].unsqueeze(-1)], dim=-1)
        # G_mid = self.sigmoid(G_mid)
        return y, s, G_mid

def check_parameters(net):
    '''
        Returns module parameters. Mb
    '''
    parameters = sum(param.numel() for param in net.parameters())
    return parameters / 10**6


class GRADNILM_Trainer(metaclass=ABCMeta):
    def __init__(self, args, model, train_loader = None, val_loader = None):
        self.args = args
        self.device = 'cuda'
        self.model = model.to(self.device)
        self.num_epochs = self.args.num_epochs
        self.patience = self.args.patience
        self.export_root = Path(self.args.export_root)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.means = self.args.train_means
        self.stds  = self.args.train_stds
        self.alpha = 0.1
        self.beta  = 0.1
        self.optimizer = self._create_optimizer()
        if self.args.enable_lr_schedule:
            scheduler_kwargs = {'factor': 0.1, 'patience': 5, 'mode': 'max'}
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **scheduler_kwargs)

        if self.train_loader != None:
            self.log = os.path.join('./logs', self.args.export_root.split('/')[-3] + '_' + self.args.export_root.split('/')[-2] + '/' + self.args.export_root.split('/')[-1])
            if not self.log:
                os.mkdirs(self.log)
            self.writer = SummaryWriter(self.log)

        self.mae = nn.L1Loss(reduction = 'mean')

    def train(self):
        # val_mae_loss = []
        patience_cnt = 0
        best_val_mae_loss = self.validate(epoch=0)

        for epoch in range(self.num_epochs):
            if patience_cnt == self.patience:
                print("Valid-MAE-LOSS does not improve after {} Epochs, thus Earlystopping is calling".format(self.patience))
                break

            self.train_one_epoch(epoch + 1)

            val_mae_loss = self.validate(epoch + 1)
            # val_mae_loss.append(val_mae_loss.tolist())

            if val_mae_loss < best_val_mae_loss:
                best_val_mae_loss = val_mae_loss
                self._save_state_dict()
                patience_cnt = 0
            else:
                patience_cnt = patience_cnt + 1

    def train_one_epoch(self, epoch):
        train_mse_losses, train_ce_losses, train_l1_losses, train_losses, train_running_metrics = [], [], [], [], []
        y_hats, ys, s_hats, ss = [], [], [], []
        self.model.train()
        tqdm_dataloader = tqdm(self.train_loader)
        for batch_idx, batch in enumerate(tqdm_dataloader):
            self.optimizer.zero_grad()

            X, y, s = batch
            g = get_adj_batch_dir(s)
            X, y, s, g = X.to(self.device), y.to(self.device), s.to(self.device), g.to(self.device)
            X = X.squeeze(-1)
            
            y_hat, s_hat, g_hat = self.model(X)
            pred_G       = g_hat.view(-1, g_hat.shape[2] * g_hat.shape[1])
            true_label_G = g.view(-1, g.shape[2] * g.shape[1])
            mse_loss  = F.mse_loss(y_hat, y)
            ce_loss   = F.binary_cross_entropy(torch.sigmoid(s_hat), s)
            l1_loss   = F.binary_cross_entropy(pred_G, torch.sigmoid(true_label_G - 1.)) # TODO
            # l1_loss   = F.binary_cross_entropy(pred_G, torch.tanh(true_label_G))
            loss = mse_loss + self.alpha * ce_loss + self.beta * l1_loss
            train_mse_losses.append(mse_loss)
            train_ce_losses.append(ce_loss)
            train_l1_losses.append(l1_loss)
            train_losses.append(loss)
            loss.backward()
            self.optimizer.step()

            y_hat = y_hat.detach()
            s_hat = s_hat.detach()
            y_hats.append(y_hat.cpu())
            ys.append(y.cpu())
            s_hats.append(s_hat.cpu())
            ss.append(s.cpu())

        y_hat, y, s_hat, s = torch.vstack(y_hats), torch.vstack(ys), torch.vstack(s_hats), torch.vstack(ss)
        
        y_hat, y = self.undo_normalize(y_hat, self.means.to(y_hat.device), self.stds.to(y_hat.device)), \
                   self.undo_normalize(y,     self.means.to(y.device),     self.stds.to(y.device))

        train_running_metrics = mean_metrics(y_hat, y, s_hat, s)

        self.writer.add_scalar('train/loss/mse_loss', torch.mean(torch.as_tensor(train_mse_losses)), global_step=epoch)
        self.writer.add_scalar('train/loss/cross_entropy', torch.mean(torch.as_tensor(train_ce_losses)), global_step=epoch)
        self.writer.add_scalar('train/loss/l1_loss', torch.mean(torch.as_tensor(train_l1_losses)), global_step=epoch)
        self.writer.add_scalar('train/loss/loss', torch.mean(torch.as_tensor(train_losses)), global_step=epoch)

        self.writer.add_scalar('train/metrics/train_f1', train_running_metrics[0], global_step=epoch)
        self.writer.add_scalar('train/metrics/train_nde', train_running_metrics[1], global_step=epoch)
        self.writer.add_scalar('train/metrics/train_eac', train_running_metrics[2], global_step=epoch)
        self.writer.add_scalar('train/metrics/train_mae', train_running_metrics[3], global_step=epoch)

    def validate(self, epoch):
        self.model.eval()
        val_mse_losses, val_ce_losses, val_losses, val_l1_losses = [], [], [], []
        val_mae_WashingMachine, val_mae_Dishwasher, val_mae_Kettle, val_mae_Fridge, val_mae_Microwave = [],[],[],[],[]
        val_running_metrics = 0
        y_hats, ys, s_hats, ss = [], [], [], []

        with torch.no_grad():
            tqdm_dataloader = tqdm(self.val_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                X, y, s = batch
                g = get_adj_batch_dir(s)
                X, y, s, g = X.to(self.device), y.to(self.device), s.to(self.device), g.to(self.device)
                X = X.squeeze(-1)

                y_hat, s_hat, g_hat = self.model(X)
                pred_G       = g_hat.view(-1, g_hat.shape[2] * g_hat.shape[1])
                true_label_G = g.view(-1, g.shape[2] * g.shape[1])
                mse_loss  = F.mse_loss(y_hat, y)
                ce_loss   = F.binary_cross_entropy(torch.sigmoid(s_hat), s)
                l1_loss   = F.binary_cross_entropy(pred_G, torch.sigmoid(true_label_G - 1.))  # TODO
                # l1_loss   = F.binary_cross_entropy(pred_G, torch.tanh(true_label_G))
                loss = mse_loss + self.alpha * ce_loss + self.beta * l1_loss
                val_mse_losses.append(mse_loss)
                val_ce_losses.append(ce_loss)
                val_l1_losses.append(l1_loss)
                val_losses.append(loss)
                y_hat = y_hat.detach()
                s_hat = s_hat.detach()

                y_hats.append(y_hat.cpu())
                ys.append(y.cpu())
                s_hats.append(s_hat.cpu())
                ss.append(s.cpu())

                val_mae_WashingMachine.append(self.mae(y_hat[:,:,0], y[:,:,0]).item())
                val_mae_Dishwasher.append(self.mae(y_hat[:,:,1], y[:,:,1]).item())
                val_mae_Kettle.append(self.mae(y_hat[:,:,2], y[:,:,2]).item())
                val_mae_Fridge.append(self.mae(y_hat[:,:,3], y[:,:,3]).item())
                val_mae_Microwave.append(self.mae(y_hat[:,:,4], y[:,:,4]).item())

        val_loss = torch.mean(torch.as_tensor(val_losses))
        
        y_hat, y, s_hat, s = torch.vstack(y_hats), torch.vstack(ys), torch.vstack(s_hats), torch.vstack(ss)
        y_hat, y = self.undo_normalize(y_hat, self.means.to(y_hat.device), self.stds.to(y_hat.device)), \
                   self.undo_normalize(y,     self.means.to(y.device),     self.stds.to(y.device))

        val_running_metrics = mean_metrics(y_hat, y, s_hat, s)

        self.writer.add_scalar('val/loss/mse_loss', torch.mean(torch.as_tensor(val_mse_losses)), global_step=epoch)
        self.writer.add_scalar('val/loss/cross_entropy', torch.mean(torch.as_tensor(val_ce_losses)), global_step=epoch)
        self.writer.add_scalar('val/loss/l1_loss', torch.mean(torch.as_tensor(val_l1_losses)), global_step=epoch)
        self.writer.add_scalar('val/loss/loss', val_loss, global_step=epoch)

        self.writer.add_scalar('val/metrics/f1', val_running_metrics[0], global_step=epoch)
        self.writer.add_scalar('val/metrics/nde', val_running_metrics[1], global_step=epoch)
        self.writer.add_scalar('val/metrics/eac', val_running_metrics[2], global_step=epoch)
        self.writer.add_scalar('val/metrics/mae', val_running_metrics[3], global_step=epoch)
        self.writer.add_scalar('hyperpara/learning_rate', self.optimizer.param_groups[0]['lr'], global_step=epoch)
   
        self.writer.add_scalar('val/single_mae/WashingMachine', np.mean(np.array(val_mae_WashingMachine)), global_step=epoch)
        self.writer.add_scalar('val/single_mae/Dishwasher',     np.mean(np.array(val_mae_Dishwasher)), global_step=epoch)
        self.writer.add_scalar('val/single_mae/Kettle',         np.mean(np.array(val_mae_Kettle)), global_step=epoch)
        self.writer.add_scalar('val/single_mae/Fridge',         np.mean(np.array(val_mae_Fridge)), global_step=epoch)
        self.writer.add_scalar('val/single_mae/Microwave',      np.mean(np.array(val_mae_Microwave)), global_step=epoch)

        self.lr_scheduler.step(val_running_metrics[0])
        
        return val_running_metrics[3]

    def test(self, test_loader):
        self._load_best_model()
        self.model.eval()
        with torch.no_grad():
            tqdm_dataloader = tqdm(test_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                input, label, status = batch
                input, label, status = input.to(self.device), label.to(self.device), status.to(self.device)
                input = input.squeeze(-1)
                # if torch.sum(status) != 0:
                #     print()
                # if torch.sum(get_adj_batch_dir(status)) > 4:
                #     print('4')
                # if torch.sum(get_adj_batch_dir(status)) > 8:
                #     print('8')
                # if torch.sum(get_adj_batch_dir(status)) > 12:
                #     print('12')
                # if torch.sum(get_adj_batch_dir(status)) > 16:
                #     print('16')
                # if torch.sum(get_adj_batch_dir(status)) > 20:
                #     print('20')

                y_hats, s_hats, _ = self.model(input)
                y_hats = self.undo_normalize(y_hats, self.means.to(y_hats.device), self.stds.to(y_hats.device))
                label  = self.undo_normalize(label,  self.means.to(label.device),  self.stds.to(label.device))
                if batch_idx == 0:
                    return_yh = y_hats
                    return_sh = s_hats
                    return_y  = label
                    return_x  = input.unsqueeze(-1)
                    return_s  = status
                else:
                    return_yh = torch.cat((return_yh, y_hats), dim=0)
                    return_sh = torch.cat((return_sh, s_hats), dim=0)
                    return_y = torch.cat((return_y, label),   dim=0)
                    return_x = torch.cat((return_x, input.unsqueeze(-1)),   dim=0)
                    return_s = torch.cat((return_s, status),   dim=0)
        return return_yh.cpu().numpy(), return_sh.cpu().numpy(), return_x.cpu().numpy(), return_y.cpu().numpy(), return_s.cpu().numpy()


    def _create_optimizer(self):
        args = self.args
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'layer_norm']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay,
            },
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]
        if args.optimizer.lower() == 'rmsprop':
            return torch.optim.RMSprop(optimizer_grouped_parameters, lr=args.lr)
            # return torch.optim.RMSprop(self.model.parameters(), lr=args.lr)
        elif args.optimizer.lower() == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=args.lr, betas=(0.9, 0.98))
        elif args.optimizer.lower() == 'sgd':
            return torch.optim.SGD(optimizer_grouped_parameters, lr=args.lr, momentum=args.momentum)
        else:
            raise ValueError

    def _load_best_model(self):
        try:
            self.model.load_state_dict(torch.load(self.export_root.joinpath('best_acc_model.pth')))
            self.model.to(self.device)
        except:
            print('Failed to load best model, continue testing with current model...')

    def _save_state_dict(self):
        if not os.path.exists(self.export_root):
            os.makedirs(self.export_root)
        print('Saving best model...')
        torch.save(self.model.state_dict(),
                   self.export_root.joinpath('best_acc_model.pth'))

    def undo_normalize(self, signal, mean, std):
        """
        Takes Signal and undoes the normalization with its mean and std

        Args:
            signal (np.array): signal to create quantiles
            mean (float): mean value used for normalizing
            std (float): standard deviation used for normalizing
        Returns:
            normalized_signal (np.array): normalized signal
        """
        return signal * std + mean

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    target_appliances = ['washing_machine', 'dishwasher', 'kettle', 'fridge', 'microwave']
    x = torch.randn(32, 128).cuda()
    nnet = GRADNILM().cuda()
    s = nnet(x)
    print(str(check_parameters(nnet))+' Mb')
    print(nnet)