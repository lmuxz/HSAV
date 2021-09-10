import copy
import torch
import pickle
import numpy as np
import progressbar as pb

from torch import nn
from torch.utils import data


class fully_connected_embed():
    """
    Training wrapper of a fully connected network with embedding layer
    """
    def __init__(self, model, cate_index, device=torch.device("cuda")):
        self.device = device
        self.model = model.to(device)
        self.cate_index = cate_index

    def fit(self, xs, ys, xt, xv, yv, 
            xtt=None, ytt=None, lmbda=0.01,
            epoch=20, batch_size=16, lr=0.001, beta=0, early_stop=True, verbose=True):
        xs_cate = xs[:, :self.cate_index]
        xs_num = xs[:, self.cate_index:]
        xt_cate = xt[:, :self.cate_index]
        xt_num = xt[:, self.cate_index:]
        xv_cate = xv[:, :self.cate_index]
        xv_num = xv[:, self.cate_index:]

        if xtt is not None:
            xtt_cate = torch.from_numpy(xtt[:, :self.cate_index]).long()
            xtt_num = torch.from_numpy(xtt[:, self.cate_index:]).float()
            ytt = torch.from_numpy(ytt).float().reshape(-1, 1)

        train_tensor = data.TensorDataset(
            torch.from_numpy(xs_cate).long(),
            torch.from_numpy(xs_num).float(),
            torch.from_numpy(ys).reshape(-1,1).float(),
            torch.from_numpy(xt_cate).long(),
            torch.from_numpy(xt_num).float()
        )

        valid_tensor = data.TensorDataset(
            torch.from_numpy(xv_cate).long(),
            torch.from_numpy(xv_num).float(),
            torch.from_numpy(yv).reshape(-1,1).float()
        )

        train_loader = data.DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
        valid_loader = data.DataLoader(valid_tensor, batch_size=batch_size, shuffle=True)

        optim = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999))

        widget = [pb.DynamicMessage("epoch"), ' ', 
                    pb.Percentage(), ' ',
                    pb.ETA(), ' ',
                    pb.Timer(), ' ', 
                    pb.DynamicMessage("lossT"), ' ',
                    pb.DynamicMessage("lossV"), ' ',
                    pb.DynamicMessage("mmd")]

        best_risk = float('inf')
        best_state = copy.deepcopy(self.model.state_dict())
        for e in range(epoch):
            iteration_per_epoch = int(xs_cate.shape[0] / batch_size) + 1

            if verbose:
                timer = pb.ProgressBar(widgets=widget, maxval=iteration_per_epoch).start()

            self.model.train(True)
            cum_loss = 0
            cum_mmd = 0
            for b, (source_cate, source_num, source_label, target_cate, target_num) in enumerate(train_loader):
                source_cate = source_cate.to(self.device)
                source_num = source_num.to(self.device)
                source_label = source_label.to(self.device)
                target_cate = target_cate.to(self.device)
                target_num = target_num.to(self.device)
                
                self.model.zero_grad()

                output = self.model(source_cate, source_num)
                source_hidden_rep = self.model.hidden_rep
                l = nn.BCEWithLogitsLoss()(output, source_label)

                if xtt is not None:
                    ind = np.random.choice(xtt.shape[0], xtt.shape[0], replace=True)
                    tt_cate = xtt_cate[ind].to(self.device)
                    tt_num = xtt_num[ind].to(self.device)
                    tt_label = ytt[ind].to(self.device)

                    tt_pred = self.model(tt_cate, tt_num)

                    l_weakly = nn.BCEWithLogitsLoss()(tt_pred, tt_label)
                    l += l_weakly * lmbda

                l.backward(retain_graph=True)
                cum_loss += l.detach().cpu().data

                if beta==0:
                    mmd = 0
                else:
                    _ = self.model(target_cate, target_num)
                    target_hidden_rep = self.model.hidden_rep

                    mmd = beta * self.mmd_rbf_noaccelerate(source_hidden_rep, target_hidden_rep, 
                                                            kernel_mul=2.0, kernel_num=4, fix_sigma=None)
                    mmd.backward()
                    cum_mmd += mmd.detach().cpu().data

                percentage = (e * iteration_per_epoch + b) / (epoch * iteration_per_epoch)
                optim = self.optimizer_regularization(optim, [lr], 10, percentage, 0.75)
                optim.step()
                if verbose:
                    timer.update(b+1, epoch=e+1, lossT=float(cum_loss)/(b+1), mmd=float(cum_mmd)/(b+1))
            
            self.model.train(False)
            cum_loss = 0
            for b, (source_cate, source_num, source_label) in enumerate(valid_loader):
                source_cate = source_cate.to(self.device)
                source_num = source_num.to(self.device)
                source_label = source_label.to(self.device)

                output = self.model(source_cate, source_num)
                l = nn.BCEWithLogitsLoss()(output, source_label)
                cum_loss += l.detach().cpu().data

                if verbose:
                    timer.update(iteration_per_epoch, epoch=e+1, lossV=float(cum_loss)/(b+1))
            if verbose:
                timer.finish()

            if cum_loss/(b+1) < best_risk:
                best_risk = cum_loss/(b+1)
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                if early_stop or np.isnan(cum_loss) or np.isnan(cum_mmd):
                    if verbose:
                        print("Early Stop", flush=True)
                    break
        if verbose:
            print("Best valid risk: {:.4f}".format(best_risk), flush=True)
        self.model.load_state_dict(best_state)
        self.model.eval()


    def predict_cuda(self, target_train, batch_size=128):
        """
        target_train is a pytorch tensor in gpu
        return results in gpu
        """

        i = 0
        batch = target_train[i*batch_size:(i+1)*batch_size, :]
        res = torch.tensor([], device=self.device)
        while len(batch) > 0:
            batch_cate = batch[:, :self.cate_index].long()
            batch_num = batch[:, self.cate_index:].float()
            prediction = self.model(batch_cate, batch_num)
            res = torch.cat((res, prediction.detach().reshape(-1)), 0)

            i += 1
            batch = target_train[i*batch_size:(i+1)*batch_size, :]
        return torch.sigmoid(res)


    def predict(self, target, batch_size=1024, num_workers=5):
        target_cate = target[:, :self.cate_index]
        target_num = target[:, self.cate_index:]
        
        data_tensor = data.TensorDataset(
            torch.from_numpy(target_cate).long(),
            torch.from_numpy(target_num).float()
        )
        dataloader = data.DataLoader(data_tensor, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        res = torch.tensor([], device=self.device)
        for num_batch, (dataset_cate, dataset_num) in enumerate(dataloader):
            dataset_cate = dataset_cate.to(self.device)
            dataset_num = dataset_num.to(self.device)
            prediction = self.model(dataset_cate, dataset_num)
            res = torch.cat((res, prediction.detach()), 0)
        # probs = 1 / (1 + np.exp(-res.cpu().numpy().reshape(-1)))
        # return probs
        return torch.sigmoid(res).cpu().numpy().reshape(-1)


    def save_embedding_dict(self, path):
        embedding_dict = []
        for embedding in self.model.embed:
            embedding_dict.append(embedding.state_dict())
        with open(path, "wb") as file:
            pickle.dump(embedding_dict, file)


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



class fully_connected():
    """
    Training wrapper of a fully connected network
    """
    def __init__(self, model, device=torch.device("cuda")):
        self.device = device
        self.model = model.to(device)
    

    def fit(self, xs, ys, xt, xv, yv, epoch=20, batch_size=16, lr=0.001, beta=0, 
        xtt=None, ytt=None,
        early_stop=True, verbose=True):
        train_tensor = data.TensorDataset(
            torch.from_numpy(xs).float(),
            torch.from_numpy(ys).reshape(-1,1).float(),
            torch.from_numpy(xt).float()
        )

        valid_tensor = data.TensorDataset(
            torch.from_numpy(xv).float(),
            torch.from_numpy(yv).reshape(-1,1).float()
        )

        train_loader = data.DataLoader(train_tensor, batch_size=batch_size)
        valid_loader = data.DataLoader(valid_tensor, batch_size=batch_size)

        optim = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999))

        widget = [pb.DynamicMessage("epoch"), ' ', 
                    pb.Percentage(), ' ',
                    pb.ETA(), ' ',
                    pb.Timer(), ' ', 
                    pb.DynamicMessage("lossT"), ' ',
                    pb.DynamicMessage("lossV"), ' ',
                    pb.DynamicMessage("mmd")]

        best_risk = float('inf')
        best_state = copy.deepcopy(self.model.state_dict())
        for e in range(epoch):
            iteration_per_epoch = int(xs.shape[0] / batch_size) + 1

            if verbose:
                timer = pb.ProgressBar(widgets=widget, maxval=iteration_per_epoch).start()

            self.model.train(True)
            cum_loss = 0
            cum_mmd = 0
            for b, (source_dataset, source_label, target_dataset) in enumerate(train_loader):
                source_dataset = source_dataset.to(self.device)
                source_label = source_label.to(self.device)
                target_dataset = target_dataset.to(self.device)
                
                self.model.zero_grad()

                output = self.model(source_dataset)
                l = nn.BCEWithLogitsLoss()(output, source_label)
                l.backward(retain_graph=True)
                cum_loss += l.detach().cpu().data

                if beta==0:
                    mmd = 0
                else:
                    source_hidden_rep = self.model.hidden_rep
                    _ = self.model(target_dataset)
                    target_hidden_rep = self.model.hidden_rep

                    mmd = beta * self.mmd_rbf_noaccelerate(source_hidden_rep, target_hidden_rep, 
                                                            kernel_mul=2.0, kernel_num=4, fix_sigma=None)
                    mmd.backward()
                    cum_mmd += mmd.detach().cpu().data

                percentage = (e * iteration_per_epoch + b) / (epoch * iteration_per_epoch)
                optim = self.optimizer_regularization(optim, [lr], 10, percentage, 0.75)
                optim.step()
                if verbose:
                    timer.update(b+1, epoch=e+1, lossT=float(cum_loss)/(b+1), mmd=float(cum_mmd)/(b+1))
            
            self.model.train(False)
            cum_loss = 0
            for b, (source_dataset, source_label) in enumerate(valid_loader):
                source_dataset = source_dataset.to(self.device)
                source_label = source_label.to(self.device)

                output = self.model(source_dataset)
                l = nn.BCEWithLogitsLoss()(output, source_label)
                cum_loss += l.detach().cpu().data

                if verbose:
                    timer.update(iteration_per_epoch, epoch=e+1, lossV=float(cum_loss)/(b+1))
            if verbose:
                timer.finish()

            if cum_loss/(b+1) < best_risk:
                best_risk = cum_loss/(b+1)
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                if early_stop or np.isnan(cum_loss) or np.isnan(cum_mmd):
                    if verbose:
                        print("Early Stop", flush=True)
                    break
        if verbose:
            print("Best valid risk: {:.4f}".format(best_risk), flush=True)
        self.model.load_state_dict(best_state)
        self.model.eval()


    def predict_cuda(self, target_train, batch_size=128):
        """
        target_train is a pytorch tensor in gpu
        return results in gpu
        """

        i = 0
        batch = target_train[i*batch_size:(i+1)*batch_size, :]
        res = torch.tensor([], device=self.device)
        while len(batch) > 0:
            prediction = self.model(batch)
            res = torch.cat((res, prediction.detach().reshape(-1)), 0)

            i += 1
            batch = target_train[i*batch_size:(i+1)*batch_size, :]
        return torch.sigmoid(res)


    def predict(self, target_train, batch_size=128, num_workers=5):
        data_tensor = data.TensorDataset(
            torch.from_numpy(target_train).float()
        )
        dataloader = data.DataLoader(data_tensor, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        res = torch.tensor([], device=self.device)
        for num_batch, (dataset,) in enumerate(dataloader):
            dataset = dataset.to(self.device)
            prediction = self.model(dataset)
            res = torch.cat((res, prediction.detach()), 0)
        # probs = 1 / (1 + np.exp(-res.cpu().numpy().reshape(-1)))
        return torch.sigmoid(res).cpu().numpy().reshape(-1)
        # return probs


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


class embed_nn(nn.Module):
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

        self.output_layer = nn.Sequential(
            nn.Linear(32, 1)
        )

    def forward(self, cate_inputs, num_inputs):
        embeddings = []
        for i in range(len(self.embed)):
            embeddings.append(self.embed[i](cate_inputs[:, i]))
        embedding = torch.cat(embeddings, 1)
        inputs = torch.cat((embedding, num_inputs), 1)
        self.hidden_rep = self.input_layer(inputs)
        return self.output_layer(self.hidden_rep)


class fully_connected_nn(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.output_layer = nn.Sequential(
            nn.Linear(32, 1)
        )
    
    def forward(self, inputs):
        self.hidden_rep = self.input_layer(inputs)
        return self.output_layer(self.hidden_rep)
