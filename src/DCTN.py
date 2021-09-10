import copy
import torch
import pickle
import numpy as np
import progressbar as pb

from torch import nn
from torch.utils import data

import pdb

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


class GradientReverse(torch.autograd.Function):
    scale = 1.0
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # print(GradientReverse.scale * grad_output.neg())
        return GradientReverse.scale * grad_output.neg()
    
def grad_reverse(x, scale=1.0):
    GradientReverse.scale = scale
    return GradientReverse.apply(x)


class feature_classifier(nn.Module):
    def __init__(self, input_dim=32):
        super().__init__()

        self.output_layer = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, inputs):
        return self.output_layer(inputs)


class domain_discriminator(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=16):
        super().__init__()

        self.output_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, inputs, scale=1.0):
        return self.output_layer(grad_reverse(inputs, scale))
        # return self.output_layer(inputs)


def get_dataloader(batch_size, shuffle=True, *tensors):
    data_tensor = data.TensorDataset(*tensors)
    return data.DataLoader(data_tensor, batch_size=batch_size, shuffle=shuffle)


def set_to_cuda(*tensors):
    cuda_tensors = []
    for tensor in tensors:
        cuda_tensors.append(tensor.to(torch.device("cuda")))
    
    return cuda_tensors


class dctn_model():
    def __init__(self, model, cate_index, device=torch.device("cuda")):
        self.device = device
        self.model = model.to(device)
        self.cate_index = cate_index


    def fit(self, xs, ys, xss, yss, 
            xt, 
            xv, yv, xvv, yvv,
            xtt=None, ytt=None,
            pre_train_params = {"epoch": 15, "batch_size":512, "lr": 0.05},
            multi_adapt_params = {"max_iter":15, "max_beta":200, "batch_size":512, "pos_lmbda":0.99, "neg_lmbda": 0.001, "lmbda":0.01,
                "tol":1e-3, "lr": 0.05},
            early_stop=True, verbose=True):
        
        # init data to torch tensor
        xs_cate = torch.from_numpy(xs[:, :self.cate_index]).long()
        xs_num = torch.from_numpy(xs[:, self.cate_index:]).float()
        xss_cate = torch.from_numpy(xss[:, :self.cate_index]).long()
        xss_num = torch.from_numpy(xss[:, self.cate_index:]).float()

        xt_cate = torch.from_numpy(xt[:, :self.cate_index]).long()
        xt_num = torch.from_numpy(xt[:, self.cate_index:]).float()

        xv_cate = torch.from_numpy(xv[:, :self.cate_index]).long()
        xv_num = torch.from_numpy(xv[:, self.cate_index:]).float()
        xvv_cate = torch.from_numpy(xvv[:, :self.cate_index]).long()
        xvv_num = torch.from_numpy(xvv[:, self.cate_index:]).float()

        ys = torch.from_numpy(ys).float().reshape(-1, 1)
        yss = torch.from_numpy(yss).float().reshape(-1, 1)
        yv = torch.from_numpy(yv).float().reshape(-1, 1)
        yvv = torch.from_numpy(yvv).float().reshape(-1, 1)

        # Weakly Labeled Data
        if xtt is not None:
            xtt_cate = torch.from_numpy(xtt[:, :self.cate_index]).long()
            xtt_num = torch.from_numpy(xtt[:, self.cate_index:]).float()
            ytt = torch.from_numpy(ytt).float().reshape(-1, 1)


        # init classifier and domain discriminator
        self.cls1 = feature_classifier().to(self.device)
        self.cls2 = feature_classifier().to(self.device)
        self.dis1 = domain_discriminator().to(self.device)
        self.dis2 = domain_discriminator().to(self.device)

        self.model.eval()
        self.cls1.eval()
        self.cls2.eval()
        self.dis1.eval()
        self.dis2.eval()

        # Pre-train
        if verbose:
            print("## Pre-train", flush=True)

        epoch = pre_train_params["epoch"]
        batch_size = pre_train_params["batch_size"]
        lr = pre_train_params["lr"]

        optim_f = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999))
        optim_c1 = torch.optim.Adam(self.cls1.parameters(), lr=lr, betas=(0.9, 0.999))
        optim_c2 = torch.optim.Adam(self.cls2.parameters(), lr=lr, betas=(0.9, 0.999))

        train_loader = get_dataloader(batch_size, True, xs_cate, xs_num, xss_cate, xss_num, ys, yss)
        valid_loader = get_dataloader(batch_size, True, xv_cate, xv_num, xvv_cate, xvv_num, yv, yvv)

        widget = [pb.DynamicMessage("epoch"), ' ', 
                    pb.Percentage(), ' ',
                    pb.ETA(), ' ',
                    pb.Timer(), ' ', 
                    pb.DynamicMessage("lossT"), ' ',
                    pb.DynamicMessage("lossV")]

        opt_cum_loss = float('inf')
        f_state = copy.deepcopy(self.model.state_dict())
        cls1_state = copy.deepcopy(self.cls1.state_dict())
        cls2_state = copy.deepcopy(self.cls2.state_dict())
        for e in range(epoch):
            iteration_per_epoch = int(xs_cate.shape[0] / batch_size) + 1

            if verbose:
                timer = pb.ProgressBar(widgets=widget, maxval=iteration_per_epoch).start()

            self.model.train(True)
            self.cls1.train(True)
            self.cls2.train(True)
            cum_loss = 0
            for b, (s_cate, s_num, ss_cate, ss_num, s_label, ss_label) in enumerate(train_loader):
                s_cate, s_num, ss_cate, ss_num, s_label, ss_label = set_to_cuda(s_cate, s_num, ss_cate, ss_num, s_label, ss_label)

                s_hidden = self.model(s_cate, s_num)
                ss_hidden = self.model(ss_cate, ss_num)

                s_pred = self.cls1(s_hidden)
                ss_pred = self.cls2(ss_hidden)

                # prediction loss
                l = nn.BCELoss()(s_pred, s_label) + nn.BCELoss()(ss_pred, ss_label)
                cum_loss += l.detach().cpu().data

                self.model.zero_grad()
                self.cls1.zero_grad()
                self.cls2.zero_grad()
                l.backward()

                optim_f.step()
                optim_c1.step()
                optim_c2.step()

                if verbose:
                    timer.update(b+1, epoch=e+1, lossT=float(cum_loss)/(b+1))
            
            self.model.train(False)
            self.cls1.train(False)
            self.cls2.train(False)
            cum_loss = 0
            for b, (s_cate, s_num, ss_cate, ss_num, s_label, ss_label) in enumerate(valid_loader):
                s_cate, s_num, ss_cate, ss_num, s_label, ss_label = set_to_cuda(s_cate, s_num, ss_cate, ss_num, s_label, ss_label)

                s_hidden = self.model(s_cate, s_num)
                ss_hidden = self.model(ss_cate, ss_num)

                s_pred = self.cls1(s_hidden)
                ss_pred = self.cls2(ss_hidden)

                # prediction loss
                l = nn.BCELoss()(s_pred, s_label) + nn.BCELoss()(ss_pred, ss_label)
                cum_loss += l.detach().cpu().data

                if verbose:
                    timer.update(iteration_per_epoch, epoch=e+1, lossV=float(cum_loss)/(b+1))

            if verbose:
                timer.finish()

            if cum_loss/(b+1) < opt_cum_loss:
                opt_cum_loss = cum_loss/(b+1) 
                f_state = copy.deepcopy(self.model.state_dict())
                cls1_state = copy.deepcopy(self.cls1.state_dict())
                cls2_state = copy.deepcopy(self.cls2.state_dict())
            else:
                if early_stop or np.isnan(cum_loss):
                    if verbose:
                        print("Early Stop", flush=True)
                    break
        if verbose:
            print("Opt Loss: {:.4f}".format(opt_cum_loss), flush=True)
        self.model.load_state_dict(f_state)
        self.cls1.load_state_dict(cls1_state)
        self.cls2.load_state_dict(cls2_state)


        max_iter = multi_adapt_params["max_iter"]
        max_beta = multi_adapt_params["max_beta"]
        batch_size = multi_adapt_params["batch_size"]
        pos_lmbda = multi_adapt_params["pos_lmbda"]
        neg_lmbda = multi_adapt_params["neg_lmbda"]
        lmbda = multi_adapt_params["lmbda"]
        lr = multi_adapt_params["lr"]
        tol = multi_adapt_params["tol"]


        target_loader = get_dataloader(batch_size, True, xt_cate, xt_num)
        source_loader = get_dataloader(batch_size, True, xs_cate, xs_num, xss_cate, xss_num)
        source_label_loader = get_dataloader(batch_size, True, xs_cate, xs_num, xss_cate, xss_num, ys, yss)

        optim_f = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999))
        optim_d1 = torch.optim.Adam(self.dis1.parameters(), lr=lr, betas=(0.9, 0.999))
        optim_d2 = torch.optim.Adam(self.dis2.parameters(), lr=lr, betas=(0.9, 0.999))
        optim_c1 = torch.optim.Adam(self.cls1.parameters(), lr=lr, betas=(0.9, 0.999))
        optim_c2 = torch.optim.Adam(self.cls2.parameters(), lr=lr, betas=(0.9, 0.999))


        if verbose:
            print("## Multi-way Adv Adaptation, Target Dis Adaptation", flush=True)
        self.model.train(True)
        self.dis1.train(True)
        self.dis2.train(True)
        prev_loss = float('inf')

        self.alpha_1 = 0
        self.alpha_2 = 0

        f_state = copy.deepcopy(self.model.state_dict())
        dis1_state = copy.deepcopy(self.dis1.state_dict())
        dis2_state = copy.deepcopy(self.dis2.state_dict())
        cls1_state = copy.deepcopy(self.cls1.state_dict())
        cls2_state = copy.deepcopy(self.cls2.state_dict())

        for e in range(max_iter):
            
            if verbose:
                print("#### Iteration:", e, flush=True)

            
            self.cls1.eval()
            self.cls2.eval()
            cum_loss = 0
            domain_loss = 0
            for b in range(max_beta):
                
                ind = np.random.choice(xs_cate.shape[0], batch_size, replace=False)
                s_cate = xs_cate[ind].to(self.device)
                s_num = xs_num[ind].to(self.device)
                s_label = ys[ind].to(self.device)
                
                ind = np.random.choice(xss_cate.shape[0], batch_size, replace=False)
                ss_cate = xss_cate[ind].to(self.device)
                ss_num = xss_num[ind].to(self.device)
                ss_label = yss[ind].to(self.device)

                ind = np.random.choice(xt_cate.shape[0], batch_size, replace=False)
                t_cate = xt_cate[ind].to(self.device)
                t_num = xt_num[ind].to(self.device)


                ones = torch.ones(s_cate.size()[0], device=self.device)
                zeros = torch.zeros(t_cate.size()[0], device=self.device)
                domain_label = torch.cat((ones, zeros)).float().reshape(-1, 1)

                s_hidden = self.model(s_cate, s_num)
                ss_hidden = self.model(ss_cate, ss_num)
                t_hidden = self.model(t_cate, t_num)

                # pdb.set_trace()

                domain_pred1 = self.dis1(torch.cat((s_hidden, t_hidden)))
                domain_pred2 = self.dis2(torch.cat((ss_hidden, t_hidden)))

                # domain discrepancy loss
                l_dis1 = nn.BCELoss()(domain_pred1, domain_label)
                l_dis2 = nn.BCELoss()(domain_pred2, domain_label)
                l_dis = (l_dis1 + l_dis2) / 2
                domain_loss += l_dis.detach().cpu().data


                self.model.zero_grad()
                self.dis1.zero_grad()
                self.dis2.zero_grad()
                l_dis.backward(retain_graph=True)
                optim_d1.step()
                optim_d2.step()

                s_pred = self.cls1(s_hidden)
                ss_pred = self.cls2(ss_hidden)

                # prediction loss
                l = nn.BCELoss()(s_pred, s_label) + nn.BCELoss()(ss_pred, ss_label)
                cum_loss += l.detach().cpu().data

                # if l_dis1 > l_dis2:
                #     s_domain_pred = self.dis1(s_hidden)
                #     t_domain_pred = self.dis1(t_hidden)
                #     l_adv = ((torch.log(s_domain_pred) + torch.log(1-s_domain_pred)).mean() + (torch.log(t_domain_pred) + torch.log(1-t_domain_pred)).mean()) / 2
                # else:
                #     ss_domain_pred = self.dis2(ss_hidden)
                #     t_domain_pred = self.dis2(t_hidden)
                #     l_adv = ((torch.log(ss_domain_pred) + torch.log(1-ss_domain_pred)).mean() + (torch.log(t_domain_pred) + torch.log(1-t_domain_pred)).mean()) / 2
                
                # self.model.zero_grad()
                # (l + l_adv).backward()
                l.backward()
                
                optim_f.step()

            if verbose:
                print("Prediction Loss:", cum_loss/(b+1), "Domain Dis Loss", domain_loss/(b+1), flush=True)

            # get alpha constant 
            self.alpha_1 = 0
            self.alpha_2 = 0
            for b, (s_cate, s_num, ss_cate, ss_num) in enumerate(source_loader):
                s_cate, s_num, ss_cate, ss_num = set_to_cuda(s_cate, s_num, ss_cate, ss_num)
                ones = torch.ones(s_cate.size()[0], device=self.device).float().reshape(-1, 1)

                s_hidden = self.model(s_cate, s_num)
                ss_hidden = self.model(ss_cate, ss_num)

                domain_pred1 = self.dis1(s_hidden)
                domain_pred2 = self.dis2(ss_hidden)

                self.alpha_1 += nn.BCELoss()(domain_pred1, ones).detach().cpu().data
                self.alpha_2 += nn.BCELoss()(domain_pred2, ones).detach().cpu().data
            self.alpha_1 /= (b+1)
            self.alpha_2 /= (b+1)

            if verbose:
                print("Alpha Constant:", self.alpha_1, self.alpha_2, flush=True)

            # get confident target prediction
            t_cate_conf = []
            t_num_conf = []
            t_label_conf = []
            for b, (t_cate, t_num) in enumerate(target_loader):
                t_cate, t_num = set_to_cuda(t_cate, t_num)

                t_hidden = self.model(t_cate, t_num)

                domain_pred1 = self.dis1(t_hidden)
                domain_pred2 = self.dis2(t_hidden)

                scf1 = -torch.log(1-domain_pred1).detach() + self.alpha_1
                scf2 = -torch.log(1-domain_pred2).detach() + self.alpha_2

                t_pred1 = self.cls1(t_hidden)
                t_pred2 = self.cls2(t_hidden)

                scf12 = scf1 + scf2
                t_pred = t_pred1 * scf1 / scf12 + t_pred2 * scf2 / scf12

                pos_ind = (t_pred > pos_lmbda).reshape(-1)
                n_pos = pos_ind.sum()
                neg_ind = (t_pred < neg_lmbda).reshape(-1)
                n_neg = neg_ind.sum()
                
                if n_pos > 0:
                    t_cate_conf.append(t_cate[pos_ind])
                    t_num_conf.append(t_num[pos_ind])
                    t_label_conf.append(torch.ones(n_pos, device=self.device).float().reshape(-1, 1))
                if n_neg > 0:
                    t_cate_conf.append(t_cate[neg_ind])
                    t_num_conf.append(t_num[neg_ind])
                    t_label_conf.append(torch.zeros(n_neg, device=self.device).float().reshape(-1, 1))

            t_cate_conf = torch.cat(t_cate_conf)
            t_num_conf = torch.cat(t_num_conf)
            t_label_conf = torch.cat(t_label_conf)

            if verbose:
                print("Pos Confident Percentage:", pos_ind.float().mean(), "Neg Confident Percentage", neg_ind.float().mean(), flush=True)


            # update F and C
            widget = [  pb.Percentage(), ' ',
                        pb.ETA(), ' ',
                        pb.Timer(), ' ', 
                        pb.DynamicMessage("lossT")]
            
            iteration_per_epoch = int(xs_cate.shape[0] / batch_size) + 1
            if verbose:
                timer = pb.ProgressBar(widgets=widget, maxval=iteration_per_epoch).start()


            self.cls1.train(True)
            self.cls2.train(True)
            cum_loss = 0
            for b, (s_cate, s_num, ss_cate, ss_num, s_label, ss_label) in enumerate(source_label_loader):
                s_cate, s_num, ss_cate, ss_num, s_label, ss_label = set_to_cuda(s_cate, s_num, ss_cate, ss_num, s_label, ss_label)

                ind = np.random.choice(t_cate_conf.shape[0], batch_size, replace=True)
                t_cate = t_cate_conf[ind]
                t_num = t_num_conf[ind]
                t_label = t_label_conf[ind]

                s_hidden = self.model(s_cate, s_num)
                ss_hidden = self.model(ss_cate, ss_num)
                t_hidden = self.model(t_cate, t_num)

                t_pred1 = self.cls1(t_hidden)
                t_pred2 = self.cls2(t_hidden)
                s_pred = self.cls1(s_hidden)
                ss_pred = self.cls2(ss_hidden)

                # prediction loss
                l = nn.BCELoss()(t_pred1, t_label) + nn.BCELoss()(t_pred2, t_label) + nn.BCELoss()(s_pred, s_label) + nn.BCELoss()(ss_pred, ss_label)

                if xtt is not None:
                    ind = np.random.choice(xtt.shape[0], xtt.shape[0], replace=True)
                    tt_cate = xtt_cate[ind].to(self.device)
                    tt_num = xtt_num[ind].to(self.device)
                    tt_label = ytt[ind].to(self.device)

                    tt_hidden = self.model(tt_cate, tt_num)
                    tt_pred1 = self.cls1(tt_hidden)
                    tt_pred2 = self.cls2(tt_hidden)

                    l_weakly = nn.BCELoss()(tt_pred1, tt_label) + nn.BCELoss()(tt_pred2, tt_label)
                    l += l_weakly * lmbda

                cum_loss += l.detach().cpu().data

                self.model.zero_grad()
                self.cls1.zero_grad()
                self.cls2.zero_grad()
                l.backward()
                optim_f.step()
                optim_c1.step()
                optim_c2.step()

                if verbose:
                    timer.update(b+1, lossT=float(cum_loss)/(b+1))

            if verbose:
                timer.finish()
                print("Previous Loss:", prev_loss, "Current Loss", cum_loss, flush=True)

            if prev_loss - cum_loss < tol:
                print("Early Stop")
                break

            prev_loss = cum_loss
            f_state = copy.deepcopy(self.model.state_dict())
            dis1_state = copy.deepcopy(self.dis1.state_dict())
            dis2_state = copy.deepcopy(self.dis2.state_dict())
            cls1_state = copy.deepcopy(self.cls1.state_dict())
            cls2_state = copy.deepcopy(self.cls2.state_dict())
            

        self.model.load_state_dict(f_state)
        self.cls1.load_state_dict(cls1_state)
        self.cls2.load_state_dict(cls2_state)
        self.dis1.load_state_dict(dis1_state)
        self.dis2.load_state_dict(dis2_state)

        self.model.eval()
        self.cls1.eval()
        self.cls2.eval()
        self.dis1.eval()
        self.dis2.eval()


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

            domain_pred1 = self.dis1(t_hidden)
            domain_pred2 = self.dis2(t_hidden)

            scf1 = -torch.log(1-domain_pred1).detach() + self.alpha_1
            scf2 = -torch.log(1-domain_pred2).detach() + self.alpha_2

            t_pred1 = self.cls1(t_hidden)
            t_pred2 = self.cls2(t_hidden)

            scf12 = scf1 + scf2
            t_pred = t_pred1 * scf1 / scf12 + t_pred2 * scf2 / scf12

            res.append(t_pred.detach().cpu().numpy().reshape(-1))
        res = np.hstack(res)
        return res

    def predict_cls1(self, target, batch_size=1024, num_workers=5):
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
            t_pred1 = self.cls1(t_hidden)

            res.append(t_pred1.detach().cpu().numpy().reshape(-1))
        res = np.hstack(res)
        return res
    

    def predict_cls2(self, target, batch_size=1024, num_workers=5):
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
            t_pred2 = self.cls2(t_hidden)

            res.append(t_pred2.detach().cpu().numpy().reshape(-1))
        res = np.hstack(res)
        return res
