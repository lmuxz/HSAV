import copy
import torch
import numpy as np
import progressbar as pb

from torch import nn
from torch.utils import data

class adaptation(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, input_dim, bias=True),
            # nn.ReLU(),
            # nn.Linear(input_dim, input_dim, bias=True)
        )
        self.input_layer[0].weight.data.copy_(torch.eye(input_dim))

    def forward(self, inputs):
        return self.input_layer(inputs)

class mmd_dist_alignment():
    """
    DAN without the classification part
    """
    def __init__(self, model, device=torch.device("cuda")):
        self.device = device
        self.trans = model.to(device)

    def fit(self, xs, xt, xvs, xvt,
            epoch=20, batch_size=16, lr=0.001, early_stop=True, verbose=True):

        train_tensor = data.TensorDataset(
            torch.from_numpy(xs).float(),
            torch.from_numpy(xt).float(),
        )

        valid_tensor = data.TensorDataset(
            torch.from_numpy(xvs).float(),
            torch.from_numpy(xvt).float(),
        )

        train_loader = data.DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
        valid_loader = data.DataLoader(valid_tensor, batch_size=batch_size, shuffle=True)


        optim_g = torch.optim.Adam(self.trans.parameters(), lr=lr, betas=(0.9, 0.999))

        widget = [pb.DynamicMessage("epoch"), ' ', 
                    pb.Percentage(), ' ',
                    pb.ETA(), ' ',
                    pb.Timer(), ' ', 
                    pb.DynamicMessage("disT"), ' ',
                    pb.DynamicMessage("disV")]


        best_state = copy.deepcopy(self.trans.state_dict())
        opt_cum_dist = 0
        for b, (source_num, target_num) in enumerate(train_loader):
            source_num = source_num.to(self.device)
            target_num = target_num.to(self.device)

            target_trans = self.trans(target_num)

            dis = self.mmd_rbf_noaccelerate(source_num, target_trans, 
                                            kernel_mul=2.0, kernel_num=4, fix_sigma=None)
            opt_cum_dist += dis.detach().cpu().data
        opt_cum_dist = float(opt_cum_dist)/(b+1)
        if verbose:
            print("Init Discrepancy:", opt_cum_dist, flush=True)
        
        for e in range(epoch):
            iteration_per_epoch = int(xs.shape[0] / batch_size) + 1

            if verbose:
                timer = pb.ProgressBar(widgets=widget, maxval=iteration_per_epoch).start()

            self.trans.train(True)
            cum_dis = 0
            for b, (source_num, target_num) in enumerate(train_loader):

                percentage = (e * iteration_per_epoch + b) / (epoch * iteration_per_epoch)
                optim_g = self.optimizer_regularization(optim_g, [lr], 10, percentage, 0.75)

                source_num = source_num.to(self.device)
                target_num = target_num.to(self.device)

                target_trans = self.trans(target_num)

                dis = self.mmd_rbf_noaccelerate(source_num, target_trans, 
                                                kernel_mul=2.0, kernel_num=4, fix_sigma=None)
                dis.backward()
                cum_dis += dis.detach().cpu().data

                optim_g.step()

                if verbose:
                    timer.update(b+1, epoch=e+1, disT=float(cum_dis)/(b+1))
            
            self.trans.train(False)
            cum_dis = 0
            for b, (source_num, target_num) in enumerate(valid_loader):
                source_num = source_num.to(self.device)
                target_num = target_num.to(self.device)

                target_trans = self.trans(target_num)

                dis = self.mmd_rbf_noaccelerate(source_num, target_trans, 
                                                kernel_mul=2.0, kernel_num=4, fix_sigma=None)
                dis.backward()
                cum_dis += dis.detach().cpu().data

                if verbose:
                    timer.update(b+1, epoch=e+1, disV=float(cum_dis)/(b+1))
            if verbose:
                timer.finish()

            if cum_dis/(b+1) < opt_cum_dist:
                opt_cum_dist = cum_dis/(b+1)
                # best_state = self.trans.state_dict()
                best_state = copy.deepcopy(self.trans.state_dict())
            else:
                if early_stop or np.isnan(cum_dis):
                    if verbose:
                        print("Early Stop", flush=True)
                    break
        if verbose:
            print("Opt Discrepancy: {}".format(opt_cum_dist), flush=True)
        self.trans.load_state_dict(best_state)
        self.trans.eval()


    def transform(self, target, batch_size=1024, num_workers=5):
        data_tensor = data.TensorDataset(
            torch.from_numpy(target).float()
        )
        dataloader = data.DataLoader(data_tensor, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        res = []
        for num_batch, (target_num,) in enumerate(dataloader):
            target_num = target_num.to(self.device)
            target_trans = self.trans(target_num)
            res.append(target_trans.detach().cpu().numpy())
        return np.vstack(res)

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)#/len(kernel_val)


    def mmd_rbf_noaccelerate(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target,
                                  kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss

    def optimizer_regularization(self, optimizer, init_lr, alpha, percentage, beta):
        for i in range(len(init_lr)):
            optimizer.param_groups[i]["lr"] = init_lr[i] * (1 + alpha * percentage)**(-beta)

        return optimizer


class cmmd_dist_alignment():
    """
    DAN without the classification part
    """
    def __init__(self, model, device=torch.device("cuda")):
        self.device = device
        self.trans = model.to(device)

    def fit(self, xs, xt, xvs, xvt,
            epoch=20, batch_size=16, lr=0.001, early_stop=True, verbose=True):

        train_tensor = data.TensorDataset(
            torch.from_numpy(xs).float(),
            torch.from_numpy(xt).float(),
        )

        valid_tensor = data.TensorDataset(
            torch.from_numpy(xvs).float(),
            torch.from_numpy(xvt).float(),
        )

        train_loader = data.DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
        valid_loader = data.DataLoader(valid_tensor, batch_size=batch_size, shuffle=True)

        optim_g = torch.optim.Adam(self.trans.parameters(), lr=lr, betas=(0.9, 0.999))

        widget = [pb.DynamicMessage("epoch"), ' ', 
                    pb.Percentage(), ' ',
                    pb.ETA(), ' ',
                    pb.Timer(), ' ', 
                    pb.DynamicMessage("disT"), ' ',
                    pb.DynamicMessage("disV")]


        best_state = copy.deepcopy(self.trans.state_dict())
        opt_cum_dist = 0
        for b, (source_num, target_num) in enumerate(train_loader):
            source_num = source_num.to(self.device)
            target_num = target_num.to(self.device)

            target_trans = self.trans(target_num)

            dis = self.mmd_rbf_noaccelerate(source_num, target_trans, 
                                            kernel_mul=2.0, kernel_num=4, fix_sigma=None)
            opt_cum_dist += dis.detach().cpu().data
        opt_cum_dist = float(opt_cum_dist)/(b+1)
        if verbose:
            print("Init Discrepancy:", opt_cum_dist, flush=True)
        
        for e in range(epoch):
            iteration_per_epoch = int(xs.shape[0] / batch_size) + 1

            if verbose:
                timer = pb.ProgressBar(widgets=widget, maxval=iteration_per_epoch).start()

            self.trans.train(True)
            cum_dis = 0
            for b, (source_num, target_num) in enumerate(train_loader):

                percentage = (e * iteration_per_epoch + b) / (epoch * iteration_per_epoch)
                optim_g = self.optimizer_regularization(optim_g, [lr], 10, percentage, 0.75)

                source_num = source_num.to(self.device)
                target_num = target_num.to(self.device)

                target_trans = self.trans(target_num)

                dis = self.mmd_rbf_noaccelerate(source_num, target_trans, 
                                                kernel_mul=2.0, kernel_num=4, fix_sigma=None)
                dis.backward()
                cum_dis += dis.detach().cpu().data

                optim_g.step()

                if verbose:
                    timer.update(b+1, epoch=e+1, disT=float(cum_dis)/(b+1))
            
            self.trans.train(False)
            cum_dis = 0
            for b, (source_num, target_num) in enumerate(valid_loader):
                source_num = source_num.to(self.device)
                target_num = target_num.to(self.device)

                target_trans = self.trans(target_num)

                dis = self.mmd_rbf_noaccelerate(source_num, target_trans, 
                                                kernel_mul=2.0, kernel_num=4, fix_sigma=None)
                dis.backward()
                cum_dis += dis.detach().cpu().data

                if verbose:
                    timer.update(b+1, epoch=e+1, disV=float(cum_dis)/(b+1))
            if verbose:
                timer.finish()

            if cum_dis/(b+1) < opt_cum_dist:
                opt_cum_dist = cum_dis/(b+1)
                # best_state = self.trans.state_dict()
                best_state = copy.deepcopy(self.trans.state_dict())
            else:
                if early_stop or np.isnan(cum_dis):
                    if verbose:
                        print("Early Stop", flush=True)
                    break
        if verbose:
            print("Opt Discrepancy: {}".format(opt_cum_dist), flush=True)
        self.trans.load_state_dict(best_state)
        self.trans.eval()


    def transform(self, target, batch_size=1024, num_workers=5):
        data_tensor = data.TensorDataset(
            torch.from_numpy(target).float()
        )
        dataloader = data.DataLoader(data_tensor, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        res = []
        for num_batch, (target_num,) in enumerate(dataloader):
            target_num = target_num.to(self.device)
            target_trans = self.trans(target_num)
            res.append(target_trans.detach().cpu().numpy())
        return np.vstack(res)

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)#/len(kernel_val)


    def mmd_rbf_noaccelerate(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target,
                                  kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss

    def optimizer_regularization(self, optimizer, init_lr, alpha, percentage, beta):
        for i in range(len(init_lr)):
            optimizer.param_groups[i]["lr"] = init_lr[i] * (1 + alpha * percentage)**(-beta)

        return optimizer