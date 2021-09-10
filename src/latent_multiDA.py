import copy
import torch
import pickle
import numpy as np
import progressbar as pb

from torch import nn
from torch.utils import data

from itertools import chain


class _WeightedBatchNorm1d(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(_WeightedBatchNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input, weights):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            total_weights = weights.sum()
            mean = (input * weights[:, None]).sum(axis=0) / total_weights
            # use biased var in train
            var = ((input - mean)**2 * weights[:, None]).sum(axis=0) / total_weights
            # var = input.var([0, 2, 3], unbiased=False)
            # n = input.numel() / input.size(1)
            n = input.size(0)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        # input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
        input = (input - mean) / (torch.sqrt(var + self.eps))
        if self.affine:
            # input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]
            input = input * self.weight + self.bias

        return input


class WeightedBatchNorm1d(nn.Module):
    def __init__(self, num_features, num_domains):
        super().__init__()

        self.wbn_layers = nn.ModuleList()
        for _ in range(num_domains):
            self.wbn_layers.append(_WeightedBatchNorm1d(num_features))
    
    def forward(self, input, weights):
        res = None
        for i in range(weights.size(1)):
            res_norm = self.wbn_layers[i](input, weights[:,i]) * weights[:,[i]]
            if i == 0:
                res = res_norm
            else:
                res += res_norm
        return res


class embed_nn(nn.Module):
    # Feature extractor
    def __init__(self, embedding_input, embedding_dim, num_dim, embedding_dict=[], embedding_file=None):
        super().__init__()

        self.embed = nn.ModuleList()
        for i in range(len(embedding_input)):
            self.embed.append(nn.Embedding(embedding_input[i], embedding_dim[i]))

        if embedding_file is not None:
            with open(embedding_file, "rb") as file:
                embedding_dict = pickle.load(file)

        if len(embedding_dict) > 0:
            for i in range(len(self.embed)):
                self.embed[i].load_state_dict(embedding_dict[i])
                for param in self.embed[i].parameters():
                    param.requires_grad = False

        input_dim = np.sum(embedding_dim) + num_dim
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )


    def forward(self, cate_inputs, num_inputs):
        embeddings = []
        for i in range(len(self.embed)):
            embeddings.append(self.embed[i](cate_inputs[:, i]))
        embedding = torch.cat(embeddings, 1)
        inputs = torch.cat((embedding, num_inputs), 1)
        return self.input_layer(inputs)


class feature_classifier(nn.Module):
    def __init__(self, input_dim=32):
        super().__init__()

        self.output_layer = nn.Sequential(
            nn.Linear(input_dim, 2),
            # nn.Sigmoid()
            nn.LogSoftmax(dim=1)
        )
    
    def forward(self, inputs):
        return self.output_layer(inputs)


class domain_discriminator(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=16, num_domains=2):
        super().__init__()

        self.output_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_domains),
            nn.LogSoftmax(dim=1)
        )
    
    def forward(self, inputs):
        return self.output_layer(inputs)


def get_dataloader(batch_size, shuffle=True, *tensors):
    data_tensor = data.TensorDataset(*tensors)
    return data.DataLoader(data_tensor, batch_size=batch_size, shuffle=shuffle)


def set_to_cuda(*tensors):
    cuda_tensors = []
    for tensor in tensors:
        cuda_tensors.append(tensor.to(torch.device("cuda")))
    
    return cuda_tensors


def optimizer_regularization(optimizer, init_lr, alpha, percentage, beta):
    for i in range(len(init_lr)):
        optimizer.param_groups[i]["lr"] = init_lr[i] * (1 + alpha * percentage)**(-beta)

    return optimizer


class latent_multiDA_model():
    def __init__(self, model, cate_index, device=torch.device("cuda")):
        self.device = device
        self.model = model.to(device)
        self.cate_index = cate_index


    def fit(self, xs, ys,
            xt, 
            xtt=None, ytt=None, lmbda=None,
            epoch=15, batch_size=512, lr=0.05, tol=1e-3,
            verbose=True):
        
        # init data to torch tensor
        xs_cate = torch.from_numpy(xs[:, :self.cate_index]).long()
        xs_num = torch.from_numpy(xs[:, self.cate_index:]).float()

        xt_cate = torch.from_numpy(xt[:, :self.cate_index]).long()
        xt_num = torch.from_numpy(xt[:, self.cate_index:]).float()


        ys = torch.from_numpy(ys).long()

        # Weakly Labeled Data
        if xtt is not None:
            xtt_cate = torch.from_numpy(xtt[:, :self.cate_index]).long()
            xtt_num = torch.from_numpy(xtt[:, self.cate_index:]).float()
            ytt = torch.from_numpy(ytt).long()


        # init classifier and domain discriminator
        self.cls = feature_classifier().to(self.device)
        self.source_dis = domain_discriminator().to(self.device)
        self.source_norm = WeightedBatchNorm1d(32, 2).to(self.device)
        self.target_norm = nn.BatchNorm1d(32).to(self.device)


        optim_f = torch.optim.Adam(
            chain(
                self.model.parameters(), 
                self.cls.parameters(), 
                self.source_dis.parameters(), 
                self.source_norm.parameters(), 
                self.target_norm.parameters()),
            lr=lr, betas=(0.9, 0.999))

        train_loader = get_dataloader(batch_size, True, xs_cate, xs_num, xt_cate, xt_num, ys)

        widget = [pb.DynamicMessage("epoch"), ' ', 
                    pb.Percentage(), ' ',
                    pb.ETA(), ' ',
                    pb.Timer(), ' ', 
                    pb.DynamicMessage("cls"), ' ',
                    pb.DynamicMessage("dom"), ' ',
                    pb.DynamicMessage("loss")]

        opt_cum_loss = float('inf')
        f_state = copy.deepcopy(self.model.state_dict())
        cls_state = copy.deepcopy(self.cls.state_dict())
        source_dis_state = copy.deepcopy(self.source_dis.state_dict())
        source_norm_state = copy.deepcopy(self.source_norm.state_dict())
        target_norm_state = copy.deepcopy(self.target_norm.state_dict())
        for e in range(epoch):
            iteration_per_epoch = int(xs_cate.shape[0] / batch_size) + 1

            if verbose:
                timer = pb.ProgressBar(widgets=widget, maxval=iteration_per_epoch).start()

            self.model.train(True) 
            self.cls.train(True) 
            self.source_dis.train(True) 
            self.source_norm.train(True) 
            self.target_norm.train(True)
            cum_loss_cls = 0
            cum_loss_dom = 0
            cum_loss = 0


            for b, (s_cate, s_num, t_cate, t_num, s_label) in enumerate(train_loader):
                s_cate, s_num, t_cate, t_num, s_label = set_to_cuda(s_cate, s_num, t_cate, t_num, s_label)

                s_hidden = self.model(s_cate, s_num)
                t_hidden = self.model(t_cate, t_num)

                s_domains = self.source_dis(s_hidden)

                s_pred = self.cls(self.source_norm(s_hidden, torch.exp(s_domains)))
                t_pred = self.cls(self.target_norm(t_hidden))


                if xtt is not None:
                    ind = np.random.choice(xtt.shape[0], xtt.shape[0], replace=True)
                    tt_cate = xtt_cate[ind].to(self.device)
                    tt_num = xtt_num[ind].to(self.device)
                    tt_label = ytt[ind].to(self.device)

                    tt_hidden = self.model(tt_cate, tt_num)
                    tt_pred = self.cls(self.target_norm(tt_hidden))

                    l_cls_loss = nn.NLLLoss()(s_pred, s_label) + nn.NLLLoss()(tt_pred, tt_label) * lmbda
                else:
                    l_cls_loss = nn.NLLLoss()(s_pred, s_label)
                
                l_cls = l_cls_loss
                cum_loss_cls += l_cls.detach().cpu().data


                s_domains_avg = torch.exp(s_domains).mean(axis=0)
                l_dom_entropy = -(s_domains[:,0] * torch.exp(s_domains[:,0]) + s_domains[:,1] * torch.exp(s_domains[:,1])).mean()
                l_dom_avg = s_domains_avg[0] * torch.log(s_domains_avg[0]) + s_domains_avg[1] * torch.log(s_domains_avg[1])
                l_dom = l_dom_avg * 3 + l_dom_entropy 
                cum_loss_dom += l_dom.detach().cpu().data


                l = l_dom + l_cls
                cum_loss += l.detach().cpu().data

                self.model.zero_grad()
                self.cls.zero_grad()
                self.source_dis.zero_grad()
                self.source_norm.zero_grad()
                self.target_norm.zero_grad()
                l.backward()

                optim_f.step()

                if verbose:
                    timer.update(b+1, epoch=e+1, cls=float(cum_loss_cls)/(b+1), dom=float(cum_loss_dom)/(b+1), loss=float(cum_loss)/(b+1))
                    

            if verbose:
                timer.finish()

            if opt_cum_loss - cum_loss/(b+1) < tol:
                break


            opt_cum_loss = cum_loss/(b+1)

            f_state = copy.deepcopy(self.model.state_dict())
            cls_state = copy.deepcopy(self.cls.state_dict())
            source_dis_state = copy.deepcopy(self.source_dis.state_dict())
            source_norm_state = copy.deepcopy(self.source_norm.state_dict())
            target_norm_state = copy.deepcopy(self.target_norm.state_dict())

        if verbose:
            print("Opt Loss: {:.4f}".format(opt_cum_loss), flush=True)


        self.model.load_state_dict(f_state)
        self.cls.load_state_dict(cls_state)
        self.source_dis.load_state_dict(source_dis_state)
        self.source_norm.load_state_dict(source_norm_state)
        self.target_norm.load_state_dict(target_norm_state)

        self.model.eval()
        self.cls.eval()
        self.source_dis.eval()
        self.source_norm.eval()
        self.target_norm.eval()


    def predict(self, target, batch_size=1024, num_workers=5):
        target_cate = target[:, :self.cate_index]
        target_num = target[:, self.cate_index:]
        
        data_tensor = data.TensorDataset(
            torch.from_numpy(target_cate).long(),
            torch.from_numpy(target_num).float()
        )
        dataloader = data.DataLoader(data_tensor, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        res = []
        for dataset_cate, dataset_num in dataloader:
            dataset_cate = dataset_cate.to(self.device)
            dataset_num = dataset_num.to(self.device)

            t_hidden = self.model(dataset_cate, dataset_num)

            t_pred = torch.exp(self.cls(self.target_norm(t_hidden)))

            res.append(t_pred.detach().cpu().numpy()[:,1])
        res = np.hstack(res)
        return res
