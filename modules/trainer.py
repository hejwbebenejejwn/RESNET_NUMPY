import torch
import os
import pickle
import numpy as np


class Trainer(object):
    def __init__(self, model, optimizer, metric, loss_fn, **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric

    def train(self, train_loader, val_loader, **kwargs):

        num_epochs = kwargs.get("num_epochs", 0)
        log_epochs = kwargs.get("log_epochs", 100)
        save_dir = kwargs.get("save_dir", None)
        lr = kwargs.get("lr", 0.1)
        best_score = 0
        count = 0
        optim = self.optimizer(self.model, lr=lr)
        for epoch in range(num_epochs):
            if count > 5:
                lr /= 10
                optim = self.optimizer(self.model, lr=lr)
                count = 0
            count += 1

            train_loss = 0
            self.model.train()
            for X, y in train_loader:
                if isinstance(y,torch.Tensor):
                    y = y.numpy()
                optim.zero_grad()
                logits = self.model(X)
                train_loss += float(self.loss_fn(logits, y)) * y.shape[0]
                self.loss_fn.backward()
                optim.step()
            train_loss /= len(train_loader)

            val_score, _ = self.evaluate(val_loader)
            if val_score > best_score:
                print(
                    f"[Evaluate] best accuracy performence has been updated: {best_score:.5f} --> {val_score:.5f}"
                )
                best_score = val_score
                count = 0
                if save_dir:
                    self.save_model(save_dir)

            if log_epochs and epoch % log_epochs == 0:
                print(f"[Train] epoch: {epoch}/{num_epochs}, loss: {train_loss}")

    def evaluate(self, val_loader):
        self.model.eval()
        loss = 0
        score = 0
        for X, y in val_loader:
            if isinstance(y,torch.Tensor):
                y = y.numpy()
            logits = self.model(X)
            loss += float(self.loss_fn(logits, y)) * y.shape[0]
            score += float(self.metric(logits, y)) * y.shape[0]
        loss /= len(val_loader)
        score /= len(val_loader)
        return score, loss

    def save_model(self, save_dir):
        for layer in self.model.layers:
            if isinstance(layer.params, dict):
                with open(
                    os.path.join(save_dir, layer.name + ".pdparams"), "wb"
                ) as fout:
                    pickle.dump(layer.params, fout)

            try:
                if isinstance(layer.unlearnable_params, dict):

                    with open(
                        os.path.join(save_dir, layer.name + "_unlearnable.pdparams"),
                        "wb",
                    ) as fout:
                        pickle.dump(layer.unlearnable_params, fout)
            except Exception:
                pass

    def load_model(self, model_dir):
        model_file_names = os.listdir(model_dir)
        name_file_dict = {}
        for file_name in model_file_names:
            name = file_name.replace(".pdparams", "")
            name_file_dict[name] = os.path.join(model_dir, file_name)

        for layer in self.model.layers:
            if isinstance(layer.params, dict):
                name = layer.name
                file_path = name_file_dict[name]
                with open(file_path, "rb") as fin:
                    layer.params = pickle.load(fin)
            try:
                if isinstance(layer.unlearnable_params,dict):
                    name = layer.name
                    file_path = name_file_dict[name+"_unlearnable"]
                    with open(file_path, "rb") as fin:
                        layer.unlearnable_params = pickle.load(fin)
            except Exception:
                pass
                    
