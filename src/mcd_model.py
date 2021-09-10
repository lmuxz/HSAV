import copy
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


class feature_classifier(nn.Module):
    def __init__(self, input_dim=32):
        super().__init__()

        self.output_layer = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, inputs):
        return self.output_layer(inputs)


class mcd_model():
    """
    Training wrapper of a fully connected network with embedding layer
    """
    def __init__(self, model, cate_index, device=torch.device("cuda")):
        self.device = device
        self.model = model.to(device)
        self.cate_index = cate_index

    def fit(self, xs, ys, xt, xv, yv,
            xtt=None, ytt=None, lmbda=0.01,
            epoch=20, batch_size=16, lr=0.001, early_stop=True, verbose=True):
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


        self.classifier_one = feature_classifier().to(self.device)
        self.classifier_two = feature_classifier().to(self.device)

        optim_g = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999))
        optim_f1 = torch.optim.Adam(self.classifier_one.parameters(), lr=lr, betas=(0.9, 0.999))
        optim_f2 = torch.optim.Adam(self.classifier_two.parameters(), lr=lr, betas=(0.9, 0.999))

        widget = [pb.DynamicMessage("epoch"), ' ', 
                    pb.Percentage(), ' ',
                    pb.ETA(), ' ',
                    pb.Timer(), ' ', 
                    pb.DynamicMessage("lossT"), ' ',
                    pb.DynamicMessage("disT"), ' ',
                    pb.DynamicMessage("lossV")]

        opt_cum_loss = float('inf')
        best_state = copy.deepcopy(self.model.state_dict())
        best_state_classifier_one = copy.deepcopy(self.classifier_one.state_dict())
        best_state_classifier_two = copy.deepcopy(self.classifier_two.state_dict())
        for e in range(epoch):
            iteration_per_epoch = int(xs_cate.shape[0] / batch_size) + 1

            if verbose:
                timer = pb.ProgressBar(widgets=widget, maxval=iteration_per_epoch).start()

            self.model.train(True)
            self.classifier_one.train(True)
            self.classifier_two.train(True)
            cum_loss = 0
            cum_dis = 0
            for b, (source_cate, source_num, source_label, target_cate, target_num) in enumerate(train_loader):

                percentage = (e * iteration_per_epoch + b) / (epoch * iteration_per_epoch)
                optim_g = self.optimizer_regularization(optim_g, [lr], 10, percentage, 0.75)
                optim_f1 = self.optimizer_regularization(optim_f1, [lr], 10, percentage, 0.75)
                optim_f2 = self.optimizer_regularization(optim_f2, [lr], 10, percentage, 0.75)

                source_cate = source_cate.to(self.device)
                source_num = source_num.to(self.device)
                source_label = source_label.to(self.device)
                target_cate = target_cate.to(self.device)
                target_num = target_num.to(self.device)
                
                # update generator on source labeled dataset
                self.model.zero_grad()
                self.classifier_one.zero_grad()
                self.classifier_two.zero_grad()

                middle = self.model(source_cate, source_num)
                pred_one = self.classifier_one(middle)
                pred_two = self.classifier_two(middle)
                l = nn.BCELoss()(pred_one, source_label) + nn.BCELoss()(pred_two, source_label)

                if xtt is not None:
                    ind = np.random.choice(xtt.shape[0], xtt.shape[0], replace=True)
                    tt_cate = xtt_cate[ind].to(self.device)
                    tt_num = xtt_num[ind].to(self.device)
                    tt_label = ytt[ind].to(self.device)

                    target_middle = self.model(tt_cate, tt_num)
                    tt_pred_one = self.classifier_one(target_middle)
                    tt_pred_two = self.classifier_two(target_middle)

                    l_weakly = nn.BCELoss()(tt_pred_one, tt_label) + nn.BCELoss()(tt_pred_two, tt_label)
                    l += l_weakly * lmbda

                l.backward(retain_graph=True)
                cum_loss += l.detach().cpu().data

                optim_g.step()
                optim_f1.step()
                optim_f2.step()

                # maximum discrepancy on classifier
                self.classifier_one.zero_grad()
                self.classifier_two.zero_grad()

                middle = self.model(source_cate, source_num)
                source_pred_one = self.classifier_one(middle)
                source_pred_two = self.classifier_two(middle)

                middle = self.model(target_cate, target_num)
                target_pred_one = self.classifier_one(middle)
                target_pred_two = self.classifier_two(middle)

                dis = torch.mean(torch.abs(target_pred_one - target_pred_two))*2
                l = nn.BCELoss()(source_pred_one, source_label) + nn.BCELoss()(source_pred_two, source_label)
                (l - dis).backward(retain_graph=True)

                optim_f1.step()
                optim_f2.step()

                # minimum discrepancy on classifier
                for i in range(4):
                    self.classifier_one.zero_grad()
                    self.classifier_two.zero_grad()

                    middle = self.model(target_cate, target_num)
                    target_pred_one = self.classifier_one(middle)
                    target_pred_two = self.classifier_two(middle)
                    dis = torch.mean(torch.abs(target_pred_one - target_pred_two))*2
                    dis.backward()
                    cum_dis += dis.detach().cpu().data / 4

                    optim_f1.step()
                    optim_f2.step()

                if verbose:
                    timer.update(b+1, epoch=e+1, lossT=float(cum_loss)/(b+1), disT=float(cum_dis)/(b+1))
            
            self.model.train(False)
            self.classifier_one.train(False)
            self.classifier_two.train(False)
            cum_loss = 0
            for b, (source_cate, source_num, source_label) in enumerate(valid_loader):
                source_cate = source_cate.to(self.device)
                source_num = source_num.to(self.device)
                source_label = source_label.to(self.device)

                middle = self.model(source_cate, source_num)
                source_pred_one = self.classifier_one(middle)
                source_pred_two = self.classifier_two(middle)

                l = nn.BCELoss()(source_pred_one, source_label) + nn.BCELoss()(source_pred_two, source_label)
                cum_loss += l.detach().cpu().data

                if verbose:
                    timer.update(iteration_per_epoch, epoch=e+1, lossV=float(cum_loss)/(b+1))
            if verbose:
                timer.finish()

            # if (cum_loss+cum_dis)/(b+1) < opt_cum_loss + opt_dis:
            #     opt_cum_loss = cum_loss/(b+1) 
            if cum_loss/(b+1) < opt_cum_loss:
                opt_cum_loss = cum_loss/(b+1)
                best_state = copy.deepcopy(self.model.state_dict())
                best_state_classifier_one = copy.deepcopy(self.classifier_one.state_dict())
                best_state_classifier_two = copy.deepcopy(self.classifier_two.state_dict())
            else:
                if early_stop or np.isnan(cum_loss) or np.isnan(cum_dis):
                    if verbose:
                        print("Early Stop", flush=True)
                    break
        if verbose:
            print("Opt Loss: {:.4f}".format(opt_cum_loss), flush=True)
        self.model.load_state_dict(best_state)
        self.classifier_one.load_state_dict(best_state_classifier_one)
        self.classifier_two.load_state_dict(best_state_classifier_two)
        self.model.eval()
        self.classifier_one.eval()
        self.classifier_two.eval()


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
            prediction = (self.classifier_one(middle) + self.classifier_two(middle)) / 2
            res = torch.cat((res, prediction.detach()), 0)
        # probs = 1 / (1 + np.exp(-res.cpu().numpy().reshape(-1)))
        probs = res.cpu().numpy().reshape(-1)
        return probs

    def optimizer_regularization(self, optimizer, init_lr, alpha, percentage, beta):
        for i in range(len(init_lr)):
            optimizer.param_groups[i]["lr"] = init_lr[i] * (1 + alpha * percentage)**(-beta)

        return optimizer


class mcd_model_num():
    """
    Training wrapper of a fully connected network with embedding layer
    """
    def __init__(self, model, device=torch.device("cuda")):
        self.device = device
        self.model = model.to(device)

    def fit(self, xs, ys, xt, xv, yv,
            epoch=20, batch_size=16, lr=0.001, early_stop=True, verbose=True):


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

        self.classifier_one = feature_classifier().to(self.device)
        self.classifier_two = feature_classifier().to(self.device)

        optim_g = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999))
        optim_f1 = torch.optim.Adam(self.classifier_one.parameters(), lr=lr, betas=(0.9, 0.999))
        optim_f2 = torch.optim.Adam(self.classifier_two.parameters(), lr=lr, betas=(0.9, 0.999))

        widget = [pb.DynamicMessage("epoch"), ' ', 
                    pb.Percentage(), ' ',
                    pb.ETA(), ' ',
                    pb.Timer(), ' ', 
                    pb.DynamicMessage("lossT"), ' ',
                    pb.DynamicMessage("disT"), ' ',
                    pb.DynamicMessage("lossV")]

        opt_cum_loss = float('inf')
        best_state = copy.deepcopy(self.model.state_dict())
        best_state_classifier_one = self.classifier_one.state_dict()
        best_state_classifier_two = self.classifier_two.state_dict()
        for e in range(epoch):
            iteration_per_epoch = int(xs.shape[0] / batch_size) + 1

            if verbose:
                timer = pb.ProgressBar(widgets=widget, maxval=iteration_per_epoch).start()

            self.model.train(True)
            self.classifier_one.train(True)
            self.classifier_two.train(True)
            cum_loss = 0
            cum_dis = 0
            for b, (source_num, source_label, target_num) in enumerate(train_loader):

                percentage = (e * iteration_per_epoch + b) / (epoch * iteration_per_epoch)
                optim_g = self.optimizer_regularization(optim_g, [lr], 10, percentage, 0.75)
                optim_f1 = self.optimizer_regularization(optim_f1, [lr], 10, percentage, 0.75)
                optim_f2 = self.optimizer_regularization(optim_f2, [lr], 10, percentage, 0.75)

                source_num = source_num.to(self.device)
                source_label = source_label.to(self.device)
                target_num = target_num.to(self.device)
                
                # update generator on source labeled dataset
                self.model.zero_grad()
                self.classifier_one.zero_grad()
                self.classifier_two.zero_grad()

                middle = self.model(source_num)
                pred_one = self.classifier_one(middle)
                pred_two = self.classifier_two(middle)
                l = nn.BCELoss()(pred_one, source_label) + nn.BCELoss()(pred_two, source_label)
                l.backward(retain_graph=True)
                cum_loss += l.detach().cpu().data

                optim_g.step()
                optim_f1.step()
                optim_f2.step()

                # maximum discrepancy on classifier
                self.classifier_one.zero_grad()
                self.classifier_two.zero_grad()

                middle = self.model(source_num)
                source_pred_one = self.classifier_one(middle)
                source_pred_two = self.classifier_two(middle)

                middle = self.model(target_num)
                target_pred_one = self.classifier_one(middle)
                target_pred_two = self.classifier_two(middle)

                dis = torch.mean(torch.abs(target_pred_one - target_pred_two))*2
                l = nn.BCELoss()(source_pred_one, source_label) + nn.BCELoss()(source_pred_two, source_label)
                (l - dis).backward(retain_graph=True)

                optim_f1.step()
                optim_f2.step()

                # minimum discrepancy on classifier
                for i in range(4):
                    self.classifier_one.zero_grad()
                    self.classifier_two.zero_grad()

                    middle = self.model(target_num)
                    target_pred_one = self.classifier_one(middle)
                    target_pred_two = self.classifier_two(middle)
                    dis = torch.mean(torch.abs(target_pred_one - target_pred_two))*2
                    dis.backward()
                    cum_dis += dis.detach().cpu().data / 4

                    optim_f1.step()
                    optim_f2.step()

                if verbose:
                    timer.update(b+1, epoch=e+1, lossT=float(cum_loss)/(b+1), disT=float(cum_dis)/(b+1))
            
            self.model.train(False)
            self.classifier_one.train(False)
            self.classifier_two.train(False)
            cum_loss = 0
            for b, (source_num, source_label) in enumerate(valid_loader):
                source_num = source_num.to(self.device)
                source_label = source_label.to(self.device)

                middle = self.model(source_num)
                source_pred_one = self.classifier_one(middle)
                source_pred_two = self.classifier_two(middle)

                l = nn.BCELoss()(source_pred_one, source_label) + nn.BCELoss()(source_pred_two, source_label)
                cum_loss += l.detach().cpu().data

                if verbose:
                    timer.update(iteration_per_epoch, epoch=e+1, lossV=float(cum_loss)/(b+1))
            if verbose:
                timer.finish()

            # if (cum_loss+cum_dis)/(b+1) < opt_cum_loss + opt_dis:
            #     opt_cum_loss = cum_loss/(b+1) 
            if cum_loss/(b+1) < opt_cum_loss:
                opt_cum_loss = cum_loss/(b+1)
                best_state = copy.deepcopy(self.model.state_dict())
                best_state_classifier_one = self.classifier_one.state_dict()
                best_state_classifier_two = self.classifier_two.state_dict()
            else:
                if early_stop or np.isnan(cum_loss) or np.isnan(cum_dis):
                    if verbose:
                        print("Early Stop", flush=True)
                    break
        if verbose:
            print("Opt Loss: {:.4f}".format(opt_cum_loss), flush=True)
        self.model.load_state_dict(best_state)
        self.classifier_one.load_state_dict(best_state_classifier_one)
        self.classifier_two.load_state_dict(best_state_classifier_two)
        self.model.eval()
        self.classifier_one.eval()
        self.classifier_two.eval()


    def predict(self, target, batch_size=1024, num_workers=5):
        data_tensor = data.TensorDataset(
            torch.from_numpy(target).float()
        )
        dataloader = data.DataLoader(data_tensor, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        res = torch.tensor([], device=self.device)
        for num_batch, (dataset_num,) in enumerate(dataloader):
            dataset_num = dataset_num.to(self.device)

            middle = self.model(dataset_num)
            prediction = (self.classifier_one(middle) + self.classifier_two(middle)) / 2
            res = torch.cat((res, prediction.detach()), 0)
        # probs = 1 / (1 + np.exp(-res.cpu().numpy().reshape(-1)))
        probs = res.cpu().numpy().reshape(-1)
        return probs

    def optimizer_regularization(self, optimizer, init_lr, alpha, percentage, beta):
        for i in range(len(init_lr)):
            optimizer.param_groups[i]["lr"] = init_lr[i] * (1 + alpha * percentage)**(-beta)

        return optimizer
