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


class FedALA():
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

        # --- FedALA Specific: Initialize ALA weights for each client ---
        self.ala_weights = [
            {name: torch.ones_like(param, device=cfg.device) for name, param in self.model.named_parameters()}
            for _ in range(cfg.num_clients)
        ]

        torch.save(self.model.state_dict(), self.global_model_path)
        for i in range(cfg.num_clients):
            torch.save(self.model.state_dict(), self.local_model_path[i])


    def train(self):
        print(f"Starting training with {self.cfg.algo}...")

        for t in range(1, self.cfg.commu_round + 1):
            train_losses = []

            for i in range(self.cfg.num_clients):
                train_loss = self.local_train(i, t)
                train_losses.append(train_loss)

            self.aggregate()

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
        global_model_state = torch.load(self.global_model_path, weights_only=True)
        local_model_state = torch.load(self.local_model_path[i], weights_only=True)

        ala_steps = self.cfg.local_steps // 2
        train_steps = self.cfg.local_steps - ala_steps
        
        # --- Phase 1: Learn ALA weights ---
        client_ala_weights = {name: w.clone().requires_grad_(True) for name, w in self.ala_weights[i].items()}
        optimizer_ala = optim.SGD(client_ala_weights.values(), lr=self.get_lr(t), momentum=self.cfg.momentum, weight_decay=self.cfg.weight_decay)
        
        train_iter_ala = iter(self.train_loaders[i])

        for _ in range(ala_steps):
            try:
                images, labels, _ = next(train_iter_ala)
            except StopIteration:
                train_iter_ala = iter(self.train_loaders[i])
                images, labels, _ = next(train_iter_ala)

            images, labels = images.to(self.cfg.device), labels.to(self.cfg.device)

            initialized_model_state = {}
            for name, local_tensor in local_model_state.items():
                global_tensor = global_model_state[name]
                if name in client_ala_weights:
                    weight = client_ala_weights[name]
                    initialized_model_state[name] = local_tensor + (global_tensor - local_tensor) * weight
                else:
                    initialized_model_state[name] = local_tensor
            
            # OPTIMIZATION: Use the pre-initialized self.temp_model
            self.model.load_state_dict(initialized_model_state)
            
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            
            optimizer_ala.zero_grad()
            loss.backward()
            optimizer_ala.step()

            with torch.no_grad():
                for name in client_ala_weights:
                    client_ala_weights[name].clamp_(0, 1)

        self.ala_weights[i] = {name: w.detach() for name, w in client_ala_weights.items()}

        # --- Phase 2: Train the initialized local model ---
        final_initialized_state = {}
        with torch.no_grad():
            for name, local_tensor in local_model_state.items():
                global_tensor = global_model_state[name]
                if name in self.ala_weights[i]:
                    weight = self.ala_weights[i][name]
                    final_initialized_state[name] = local_tensor + (global_tensor - local_tensor) * weight
                else:
                    final_initialized_state[name] = local_tensor

        self.model.load_state_dict(final_initialized_state)
        self.model.train()

        optimizer = optim.SGD(self.model.parameters(), lr=self.get_lr(t), momentum=self.cfg.momentum, weight_decay=self.cfg.weight_decay)
        train_loss = []
        
        train_iter_model = iter(self.train_loaders[i])
        for _ in range(train_steps):
            try:
                images, labels, _ = next(train_iter_model)
            except StopIteration:
                train_iter_model = iter(self.train_loaders[i])
                images, labels, _ = next(train_iter_model)

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