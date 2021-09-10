import copy
import math
import torch
import pickle
import numpy as np
import progressbar as pb

from torch import nn
from torch.utils import data

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

        # self.output_layer = nn.Sequential(
        #     nn.Linear(32, 1)
        # )

    def forward(self, cate_inputs, num_inputs):
        embeddings = []
        for i in range(len(self.embed)):
            embeddings.append(self.embed[i](cate_inputs[:, i]))
        embedding = torch.cat(embeddings, 1)
        inputs = torch.cat((embedding, num_inputs), 1)
        # self.hidden_rep = self.input_layer(inputs)
        # return self.output_layer(self.hidden_rep)
        return self.input_layer(inputs)


class fully_connected_nn(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # self.output_layer = nn.Sequential(
        #     nn.Linear(32, 1)
        # )
    
    def forward(self, inputs):
        # self.hidden_rep = self.input_layer(inputs)
        # return self.output_layer(self.hidden_rep)
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
        )
    
    def forward(self, inputs):
        return self.output_layer(inputs)


class domain_discriminator(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=16):
        super().__init__()

        self.output_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, inputs, scale=1.0):
        return self.output_layer(grad_reverse(inputs, scale))


class dann_model():
    """
    Training wrapper of a fully connected network with embedding layer
    """
    def __init__(self, model, cate_index, device=torch.device("cuda")):
        self.device = device
        self.model = model.to(device)
        self.cate_index = cate_index

    def fit(self, xs, ys, xt, xv, yv,
            xtt=None, ytt=None, lmbda=0.01,
            epoch=20, batch_size=16, lr=0.001, scale=1.0, early_stop=True, verbose=True):
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
            torch.from_numpy(xt_num).float(),
        )

        valid_tensor = data.TensorDataset(
            torch.from_numpy(xv_cate).long(),
            torch.from_numpy(xv_num).float(),
            torch.from_numpy(yv).reshape(-1,1).float(),
        )

        train_loader = data.DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
        valid_loader = data.DataLoader(valid_tensor, batch_size=batch_size, shuffle=True)


        self.classifier = feature_classifier().to(self.device)
        self.discriminator = domain_discriminator().to(self.device)

        optim_g = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999))
        optim_f = torch.optim.Adam(self.classifier.parameters(), lr=lr, betas=(0.9, 0.999))
        optim_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.9, 0.999))

        widget = [pb.DynamicMessage("epoch"), ' ', 
                    pb.Percentage(), ' ',
                    pb.ETA(), ' ',
                    pb.Timer(), ' ', 
                    pb.DynamicMessage("lossT"), ' ',
                    pb.DynamicMessage("disT"), ' ',
                    pb.DynamicMessage("lossV")]

        opt_cum_loss = float('inf')
        best_state = copy.deepcopy(self.model.state_dict())
        best_state_classifier = copy.deepcopy(self.classifier.state_dict())
        best_state_discriminator = copy.deepcopy(self.discriminator.state_dict())
        for e in range(epoch):
            iteration_per_epoch = int(xs_cate.shape[0] / batch_size) + 1

            if verbose:
                timer = pb.ProgressBar(widgets=widget, maxval=iteration_per_epoch).start()

            self.model.train(True)
            self.classifier.train(True)
            self.discriminator.train(True)
            cum_loss = 0
            cum_dis = 0
            for b, (source_cate, source_num, source_label, target_cate, target_num) in enumerate(train_loader):

                percentage = (e * iteration_per_epoch + b) / (epoch * iteration_per_epoch)
                optim_g = self.optimizer_regularization(optim_g, [lr], 10, percentage, 0.75)
                optim_f = self.optimizer_regularization(optim_f, [lr], 10, percentage, 0.75)
                optim_d = self.optimizer_regularization(optim_d, [lr], 10, percentage, 0.75)

                source_cate = source_cate.to(self.device)
                source_num = source_num.to(self.device)
                source_label = source_label.to(self.device)
                target_cate = target_cate.to(self.device)
                target_num = target_num.to(self.device)

                ones = torch.ones(source_cate.size()[0], device=self.device)
                zeros = torch.zeros(target_cate.size()[0], device=self.device)
                domain_label = torch.cat((ones, zeros)).reshape(-1, 1).float()
                
                self.model.zero_grad()
                self.classifier.zero_grad()
                self.discriminator.zero_grad()

                source_middle = self.model(source_cate, source_num)
                target_middle = self.model(target_cate, target_num)
                pred = self.classifier(source_middle)
                domain_pred = self.discriminator(torch.cat((source_middle, target_middle)), scale=scale)

                # prediction loss
                l = nn.BCEWithLogitsLoss()(pred, source_label)

                if xtt is not None:
                    ind = np.random.choice(xtt.shape[0], xtt.shape[0], replace=True)
                    tt_cate = xtt_cate[ind].to(self.device)
                    tt_num = xtt_num[ind].to(self.device)
                    tt_label = ytt[ind].to(self.device)

                    target_middle = self.model(tt_cate, tt_num)
                    tt_pred = self.classifier(target_middle)

                    l_weakly = nn.BCEWithLogitsLoss()(tt_pred, tt_label)
                    l += l_weakly * lmbda

                l.backward(retain_graph=True)
                cum_loss += l.detach().cpu().data

                # domain discrepancy
                dis = nn.BCEWithLogitsLoss()(domain_pred, domain_label)
                dis.backward()
                cum_dis += dis.detach().cpu().data

                optim_g.step()
                optim_f.step()
                optim_d.step()

                if verbose:
                    timer.update(b+1, epoch=e+1, lossT=float(cum_loss)/(b+1), disT=float(cum_dis)/(b+1))
            
            self.model.train(False)
            self.classifier.train(False)
            self.discriminator.train(False)
            cum_loss = 0
            for b, (source_cate, source_num, source_label) in enumerate(valid_loader):
                source_cate = source_cate.to(self.device)
                source_num = source_num.to(self.device)
                source_label = source_label.to(self.device)

                middle = self.model(source_cate, source_num)
                pred = self.classifier(middle)

                l = nn.BCEWithLogitsLoss()(pred, source_label)
                l.backward(retain_graph=True)
                cum_loss += l.detach().cpu().data

                if verbose:
                    timer.update(iteration_per_epoch, epoch=e+1, lossV=float(cum_loss)/(b+1))
            if verbose:
                timer.finish()

            if cum_loss/(b+1) < opt_cum_loss:
                opt_cum_loss = cum_loss/(b+1) 
                best_state = copy.deepcopy(self.model.state_dict())
                best_state_classifier = copy.deepcopy(self.classifier.state_dict())
                best_state_discriminator = copy.deepcopy(self.discriminator.state_dict())
            else:
                if early_stop or np.isnan(cum_loss) or np.isnan(cum_dis):
                    if verbose:
                        print("Early Stop", flush=True)
                    break
        if verbose:
            print("Opt Loss: {:.4f}".format(opt_cum_loss), flush=True)
        self.model.load_state_dict(best_state)
        self.classifier.load_state_dict(best_state_classifier)
        self.discriminator.load_state_dict(best_state_discriminator)
        self.model.eval()
        self.classifier.eval()
        self.discriminator.eval()


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

            middle = self.model(dataset_cate, dataset_num)
            prediction = self.classifier(middle)
            res = torch.cat((res, prediction.detach()), 0)
        probs = 1 / (1 + np.exp(-res.cpu().numpy().reshape(-1)))
        # probs = res.cpu().numpy().reshape(-1)
        return probs


    def optimizer_regularization(self, optimizer, init_lr, alpha, percentage, beta):
        for i in range(len(init_lr)):
            optimizer.param_groups[i]["lr"] = init_lr[i] * (1 + alpha * percentage)**(-beta)

        return optimizer


class dann_model_num():
    """
    Training wrapper of a fully connected network with embedding layer
    """
    def __init__(self, model, device=torch.device("cuda")):
        self.device = device
        self.model = model.to(device)

    def fit(self, xs, ys, xt, xv, yv,
            epoch=20, batch_size=16, lr=0.001, scale=1.0, early_stop=True, verbose=True):

        train_tensor = data.TensorDataset(
            torch.from_numpy(xs).float(),
            torch.from_numpy(ys).reshape(-1,1).float(),
            torch.from_numpy(xt).float(),
        )

        valid_tensor = data.TensorDataset(
            torch.from_numpy(xv).float(),
            torch.from_numpy(yv).reshape(-1,1).float(),
        )

        train_loader = data.DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
        valid_loader = data.DataLoader(valid_tensor, batch_size=batch_size, shuffle=True)


        self.classifier = feature_classifier().to(self.device)
        self.discriminator = domain_discriminator().to(self.device)

        optim_g = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999))
        optim_f = torch.optim.Adam(self.classifier.parameters(), lr=lr, betas=(0.9, 0.999))
        optim_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.9, 0.999))

        widget = [pb.DynamicMessage("epoch"), ' ', 
                    pb.Percentage(), ' ',
                    pb.ETA(), ' ',
                    pb.Timer(), ' ', 
                    pb.DynamicMessage("lossT"), ' ',
                    pb.DynamicMessage("disT"), ' ',
                    pb.DynamicMessage("lossV")]

        opt_cum_loss = float('inf')
        best_state = copy.deepcopy(self.model.state_dict())
        best_state_classifier = self.classifier.state_dict()
        best_state_discriminator = self.discriminator.state_dict()
        for e in range(epoch):
            iteration_per_epoch = int(xs.shape[0] / batch_size) + 1

            if verbose:
                timer = pb.ProgressBar(widgets=widget, maxval=iteration_per_epoch).start()

            self.model.train(True)
            self.classifier.train(True)
            self.discriminator.train(True)
            cum_loss = 0
            cum_dis = 0
            for b, (source_num, source_label, target_num) in enumerate(train_loader):

                percentage = (e * iteration_per_epoch + b) / (epoch * iteration_per_epoch)
                optim_g = self.optimizer_regularization(optim_g, [lr], 10, percentage, 0.75)
                optim_f = self.optimizer_regularization(optim_f, [lr], 10, percentage, 0.75)
                optim_d = self.optimizer_regularization(optim_d, [lr], 10, percentage, 0.75)

                source_num = source_num.to(self.device)
                source_label = source_label.to(self.device)
                target_num = target_num.to(self.device)

                ones = torch.ones(source_num.size()[0], device=self.device)
                zeros = torch.zeros(target_num.size()[0], device=self.device)
                domain_label = torch.cat((ones, zeros)).reshape(-1, 1).float()
                
                self.model.zero_grad()
                self.classifier.zero_grad()
                self.discriminator.zero_grad()

                source_middle = self.model(source_num)
                target_middle = self.model(target_num)
                pred = self.classifier(source_middle)
                domain_pred = self.discriminator(torch.cat((source_middle, target_middle)), scale=scale)

                # prediction loss
                l = nn.BCEWithLogitsLoss()(pred, source_label)
                l.backward(retain_graph=True)
                cum_loss += l.detach().cpu().data

                # domain discrepancy
                dis = nn.BCEWithLogitsLoss()(domain_pred, domain_label)
                dis.backward()
                cum_dis += dis.detach().cpu().data

                optim_g.step()
                optim_f.step()
                optim_d.step()

                if verbose:
                    timer.update(b+1, epoch=e+1, lossT=float(cum_loss)/(b+1), disT=float(cum_dis)/(b+1))
            
            self.model.train(False)
            self.classifier.train(False)
            self.discriminator.train(False)
            cum_loss = 0
            for b, (source_num, source_label) in enumerate(valid_loader):
                source_num = source_num.to(self.device)
                source_label = source_label.to(self.device)

                middle = self.model(source_num)
                pred = self.classifier(middle)

                l = nn.BCEWithLogitsLoss()(pred, source_label)
                l.backward(retain_graph=True)
                cum_loss += l.detach().cpu().data

                if verbose:
                    timer.update(iteration_per_epoch, epoch=e+1, lossV=float(cum_loss)/(b+1))
            if verbose:
                timer.finish()

            if cum_loss/(b+1) < opt_cum_loss:
                opt_cum_loss = cum_loss/(b+1) 
                best_state = copy.deepcopy(self.model.state_dict())
                best_state_classifier = self.classifier.state_dict()
                best_state_discriminator = self.discriminator.state_dict()
            else:
                if early_stop or np.isnan(cum_loss) or np.isnan(cum_dis):
                    if verbose:
                        print("Early Stop", flush=True)
                    break
        if verbose:
            print("Opt Loss: {:.4f}".format(opt_cum_loss), flush=True)
        self.model.load_state_dict(best_state)
        self.classifier.load_state_dict(best_state_classifier)
        self.discriminator.load_state_dict(best_state_discriminator)
        self.model.eval()
        self.classifier.eval()
        self.discriminator.eval()


    def predict(self, target, batch_size=1024, num_workers=5):
        data_tensor = data.TensorDataset(
            torch.from_numpy(target).float()
        )
        dataloader = data.DataLoader(data_tensor, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        res = torch.tensor([], device=self.device)
        for num_batch, (dataset_num,) in enumerate(dataloader):
            dataset_num = dataset_num.to(self.device)

            middle = self.model(dataset_num)
            prediction = self.classifier(middle)
            res = torch.cat((res, prediction.detach()), 0)
        probs = 1 / (1 + np.exp(-res.cpu().numpy().reshape(-1)))
        # probs = res.cpu().numpy().reshape(-1)
        return probs


    def optimizer_regularization(self, optimizer, init_lr, alpha, percentage, beta):
        for i in range(len(init_lr)):
            optimizer.param_groups[i]["lr"] = init_lr[i] * (1 + alpha * percentage)**(-beta)

        return optimizer


class adaptation(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, input_dim, bias=False)
        )
        self.input_layer[0].weight.data.copy_(torch.eye(input_dim))


    def forward(self, inputs):
        return self.input_layer(inputs)


class adv_dist_alignment():
    """
    DANN without the classification part
    """
    def __init__(self, model, device=torch.device("cuda")):
        self.device = device
        self.trans = model.to(device)

    def fit(self, xs, xt, xvs, xvt,
            epoch=20, batch_size=16, lr=0.001, lr_adv=0.001, scale=1.0, early_stop=True, verbose=True):

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

        input_dim = xs.shape[-1]
        hidden_dim = int(math.ceil(float(input_dim) / 2))
        self.discriminator = domain_discriminator(input_dim, hidden_dim).to(self.device)

        # optim = torch.optim.Adam([{"params": self.trans.parameters()}, {"params": self.discriminator.parameters()}], lr=lr, betas=(0.9, 0.999))
        optim_g = torch.optim.Adam(self.trans.parameters(), lr=lr, betas=(0.9, 0.999))
        optim_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr_adv, betas=(0.9, 0.999))

        widget = [pb.DynamicMessage("epoch"), ' ', 
                    pb.Percentage(), ' ',
                    pb.ETA(), ' ',
                    pb.Timer(), ' ', 
                    pb.DynamicMessage("disT"), ' ',
                    pb.DynamicMessage("disV")]

        
        best_state = copy.deepcopy(self.trans.state_dict())
        best_state_discriminator = copy.deepcopy(self.discriminator.state_dict())
        opt_cum_dist = -float('inf')
        for e in range(epoch):
            iteration_per_epoch = int(xs.shape[0] / batch_size) + 1

            if verbose:
                timer = pb.ProgressBar(widgets=widget, maxval=iteration_per_epoch).start()

            self.trans.train(True)
            self.discriminator.train(True)
            cum_dis = 0
            for b, (source_num, target_num) in enumerate(train_loader):

                percentage = (e * iteration_per_epoch + b) / (epoch * iteration_per_epoch)
                optim_g = self.optimizer_regularization(optim_g, [lr], 10, percentage, 0.75)
                optim_d = self.optimizer_regularization(optim_d, [lr_adv], 10, percentage, 0.75)

                source_num = source_num.to(self.device)
                target_num = target_num.to(self.device)

                ones = torch.ones(source_num.size()[0], device=self.device)
                zeros = torch.zeros(target_num.size()[0], device=self.device)
                domain_label = torch.cat((ones, zeros)).reshape(-1, 1).float()
                
                self.trans.zero_grad()
                self.discriminator.zero_grad()

                target_trans = self.trans(target_num)
                domain_pred = self.discriminator(torch.cat((source_num, target_trans)), scale=scale)

                # domain discrepancy
                dis = nn.BCEWithLogitsLoss()(domain_pred, domain_label)
                dis.backward()
                cum_dis += dis.detach().cpu().data

                optim_g.step()
                optim_d.step()

                if verbose:
                    timer.update(b+1, epoch=e+1, disT=float(cum_dis)/(b+1))
            
            self.trans.train(False)
            self.discriminator.train(False)
            cum_dis = 0
            for b, (source_num, target_num) in enumerate(valid_loader):
                source_num = source_num.to(self.device)
                target_num = target_num.to(self.device)

                ones = torch.ones(source_num.size()[0], device=self.device)
                zeros = torch.zeros(target_num.size()[0], device=self.device)
                domain_label = torch.cat((ones, zeros)).reshape(-1, 1).float()

                target_trans = self.trans(target_num)
                domain_pred = self.discriminator(torch.cat((source_num, target_trans)), scale=scale)

                dis = nn.BCEWithLogitsLoss()(domain_pred, domain_label)
                cum_dis += dis.detach().cpu().data

                if verbose:
                    timer.update(b+1, epoch=e+1, disV=float(cum_dis)/(b+1))
            if verbose:
                timer.finish()

            if cum_dis/(b+1) > opt_cum_dist:
                opt_cum_dist = cum_dis/(b+1) 
                best_state = copy.deepcopy(self.trans.state_dict())
                best_state_discriminator = copy.deepcopy(self.discriminator.state_dict())
            else:
                if early_stop or np.isnan(cum_dis):
                    if verbose:
                        print("Early Stop", flush=True)
                    break
        if verbose:
            print("Opt Discrepancy: {}".format(opt_cum_dist), flush=True)
        self.trans.load_state_dict(best_state)
        self.discriminator.load_state_dict(best_state_discriminator)
        self.trans.eval()
        self.discriminator.eval()


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


    def optimizer_regularization(self, optimizer, init_lr, alpha, percentage, beta):
        for i in range(len(init_lr)):
            optimizer.param_groups[i]["lr"] = init_lr[i] * (1 + alpha * percentage)**(-beta)

        return optimizer