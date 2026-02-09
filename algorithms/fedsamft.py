# algorithms/fedsam.py

import sys, os
base_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(base_path)

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils.utils import get_log_path, select_model, get_global_model_path, get_local_model_path, get_record_path
from torch.utils.tensorboard import SummaryWriter

class FedSAMFT():
    def __init__(self, cfg, train_loaders, test_loaders, num_data, logfile):
        self.cfg = cfg
        self.train_loaders, self.test_loaders = train_loaders, test_loaders
        self.weights = np.array(num_data) / np.sum(num_data)
        self.logfile = logfile

        self.model = select_model(cfg)
        self.local_model_path = [get_local_model_path(cfg, i) for i in range(cfg.num_clients)]
        self.global_model_path = get_global_model_path(cfg)
        self.criterion = nn.CrossEntropyLoss()
        self.record = {'train_loss': [], 'test_loss': [], 'avg_acc': [], 'sys_acc': []}
        self.record_path = get_record_path(cfg)

        self.writer = SummaryWriter(log_dir=get_log_path(cfg))

        torch.save(self.model.state_dict(), self.global_model_path)
        for i in range(cfg.num_clients):
            torch.save(self.model.state_dict(), self.local_model_path[i])


    def train(self):
        print(f"Starting training with {self.cfg.algo}...")

        # Training
        for t in range(1, self.cfg.commu_round + 1):
            train_losses = []

            if t <= self.cfg.commu_round - self.cfg.ft_round:
                for i in range(self.cfg.num_clients):
                    train_loss = self.local_train(i, t)
                    train_losses.append(train_loss)

                self.aggregate()

                self.spread_model()
            else:
                for i in range(self.cfg.num_clients):
                    train_loss = self.finetune(i, t)
                    train_losses.append(train_loss)

            self.record['train_loss'].append(np.sum(self.weights * train_losses))

            avg_acc, test_loss = self.test_averaged_performance()
            self.record['avg_acc'].append(avg_acc)
            self.record['test_loss'].append(test_loss)

            sys_acc = self.test_system_performance()
            self.record['sys_acc'].append(sys_acc)

            print(f"[Round: {t}] [Train Loss: {self.record['train_loss'][-1]:.4f}] [Test Loss: {test_loss:.4f}] [Avg Accuracy: {avg_acc:.4f}] [Sys Accuracy: {sys_acc:.4f}]")
            self.logfile.write(f"Round: {t} | Train Loss: {self.record['train_loss'][-1]:.4f} | Test Loss: {test_loss:.4f} | Avg Accuracy: {avg_acc:.4f} | Sys Accuracy: {sys_acc:.4f}\n")
            self.logfile.flush()

            self.writer.add_scalar('Train Loss', self.record['train_loss'][-1], t)
            self.writer.add_scalar('Test Loss', self.record['test_loss'][-1], t)
            self.writer.add_scalar('Avg Accuracy', self.record['avg_acc'][-1], t)
            self.writer.add_scalar('Sys Accuracy', self.record['sys_acc'][-1], t)

            np.save(self.record_path, self.record)

        print("Training completed...")
        self.writer.close()


    def get_lr(self, t):
        lr_max = self.cfg.lr
        lr_min = 0.0
        T_max = self.cfg.commu_round
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * (t - 1) / T_max))


    def local_train(self, i, t):
        self.model.load_state_dict(torch.load(self.local_model_path[i], weights_only=True))
        self.model.train()

        optimizer = optim.SGD(self.model.parameters(), lr=self.get_lr(t), momentum=self.cfg.momentum, weight_decay=self.cfg.weight_decay)

        train_loss = []
        train_iter = iter(self.train_loaders[i])

        for _ in range(self.cfg.local_steps):
            try:
                images, labels, _ = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loaders[i])
                images, labels, _ = next(train_iter)

            images, labels = images.to(self.cfg.device), labels.to(self.cfg.device)

            # --- FedSAM Modification: Two-step Optimization ---

            # 1. First forward-backward pass (Ascent Step)
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            
            # Store original gradients and climb to the 'worst-case' point
            grad_clone = [p.grad.clone() for p in self.model.parameters()]
            optimizer.zero_grad()

            # Calculate perturbation 'epsilon'
            # We need to compute the norm of the gradients
            grad_norm = torch.stack([g.detach().norm(2) for g in grad_clone]).norm(2)
            # Avoid division by zero
            eps = 1e-12 
            # Scale factor for perturbation
            scale = self.cfg.fedsam_rho / (grad_norm + eps)

            # Apply perturbation to model weights
            with torch.no_grad():
                for p, g in zip(self.model.parameters(), grad_clone):
                    e_w = g * scale
                    p.add_(e_w)

            # 2. Second forward-backward pass (Descent Step)
            logits_adv = self.model(images)
            loss_adv = self.criterion(logits_adv, labels)
            loss_adv.backward()

            # Restore original weights before optimizer step
            with torch.no_grad():
                for p, g in zip(self.model.parameters(), grad_clone):
                    e_w = g * scale
                    p.sub_(e_w)
            
            # Optimizer uses gradients from the perturbed point to update original weights
            optimizer.step()
            optimizer.zero_grad()

            train_loss.append(loss.item())
            # --- End of Modification ---

        torch.save(self.model.state_dict(), self.local_model_path[i])
        return np.mean(train_loss)

    
    def finetune(self, i, t):
        self.model.load_state_dict(torch.load(self.local_model_path[i], weights_only=True))

        self.model.train()

        optimizer = optim.SGD(self.model.parameters(), lr=self.get_lr(t), momentum=self.cfg.momentum, weight_decay=self.cfg.weight_decay)

        train_loss = []

        train_iter = iter(self.train_loaders[i])

        for _ in range(self.cfg.local_steps):
            try:
                images, labels, _ = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loaders[i])
                images, labels, _ = next(train_iter)

            images, labels = images.to(self.cfg.device), labels.to(self.cfg.device)
            
            logits = self.model(images)
            loss = self.criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        torch.save(self.model.state_dict(), self.local_model_path[i])
        
        return np.mean(train_loss)

    
    def aggregate(self):
        agg = {}
        for k, v in self.model.state_dict().items():
            agg[k] = torch.zeros_like(v)
        for i in range(self.cfg.num_clients):
            w = torch.load(self.local_model_path[i], weights_only=True)
            for k in self.model.state_dict().keys():
                agg[k] += w[k] * self.weights[i]
        torch.save(agg, self.global_model_path)


    def spread_model(self):
        global_model = torch.load(self.global_model_path, weights_only=True)
        for i in range(self.cfg.num_clients):
            torch.save(global_model, self.local_model_path[i])


    def test_averaged_performance(self):
        accs = np.zeros(self.cfg.num_clients)
        losses = np.zeros(self.cfg.num_clients)
        with torch.no_grad():
            for i in range(self.cfg.num_clients):
                self.model.load_state_dict(torch.load(self.local_model_path[i], weights_only=True))
                self.model.eval()
                acc = 0
                total = 0
                per_losses = []
                for images, labels, _ in self.test_loaders[i]:
                    images, labels = images.to(self.cfg.device), labels.to(self.cfg.device)
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                    acc += (logits.argmax(dim=1) == labels).sum().item()
                    total += labels.size(0)
                    per_losses.append(loss.item())
                accs[i] = acc / total
                losses[i] = np.mean(per_losses)
        return np.sum(self.weights * accs), np.sum(self.weights * losses)


    def test_system_performance(self):
        acc = 0
        total = 0
        local_models = [torch.load(path, weights_only=True) for path in self.local_model_path]
        with torch.no_grad():
            for test_loader in self.test_loaders:
                for images, labels, _ in test_loader:
                    images, labels = images.to(self.cfg.device), labels.to(self.cfg.device)
                    preds = []
                    for i in range(self.cfg.num_clients):
                        self.model.load_state_dict(local_models[i])
                        self.model.eval()
                        output = self.model(images)
                        if isinstance(output, tuple):
                            feature_logits, _ = output
                        else:
                            feature_logits = output
                        preds.append(feature_logits.argmax(dim=1))
                    preds = torch.stack(preds, dim=0)
                    final_pred = preds.mode(dim=0).values
                    acc += (final_pred == labels).sum().item()
                    total += labels.size(0)
        return acc / total