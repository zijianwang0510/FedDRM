import sys, os
base_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(base_path)

import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils.utils import get_log_path, select_model, get_global_model_path, get_local_model_path, get_record_path
from torch.utils.tensorboard import SummaryWriter


class ConFREE():
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
        print(f"Start training with {self.cfg.algo}...")

        # Training
        for t in range(1, self.cfg.commu_round + 1):
            train_losses = []

            for i in range(self.cfg.num_clients):
                train_loss = self.local_train(i, t)
                train_losses.append(train_loss)

            self.aggregate()
            self.spread_model()

            self.record['train_loss'].append(np.mean(train_losses))

            avg_acc, test_loss = self.test_averaged_performance()
            self.record['avg_acc'].append(avg_acc)
            self.record['test_loss'].append(test_loss)

            sys_acc = self.test_system_performance()
            self.record['sys_acc'].append(sys_acc)

            print(f"[Round: {t}] [Train Loss: {self.record['train_loss'][-1]:.4f}] [Test Loss: {test_loss:.4f}] [Avg Accuracy: {avg_acc:.4f}] [Sys Accuracy: {sys_acc:.4f}]")
            self.logfile.write(f"Round: {t} | Train Loss: {self.record['train_loss'][-1]:.4f} | Test Loss: {test_loss:.4f} | Avg Accuracy: {avg_acc:.4f} | Sys Accuracy: {sys_acc:.4f}\n")
            self.logfile.flush()

            self.writer.add_scalar('Train Loss', self.record['train_loss'][-1], t)
            self.writer.add_scalar('Test Loss', test_loss, t)
            self.writer.add_scalar('Avg Accuracy', avg_acc, t)
            self.writer.add_scalar('Sys Accuracy', sys_acc, t)

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

        head_steps = self.cfg.local_steps // 2
        backbone_steps = self.cfg.local_steps - head_steps

        # ------------ Train head ------------

        head_parameters = [p for n, p in self.model.named_parameters() if 'image_classifier' in n]

        optimizer = optim.SGD(head_parameters, lr=self.get_lr(t), momentum=self.cfg.momentum, weight_decay=self.cfg.weight_decay)

        head_train_loss = []

        train_iter = iter(self.train_loaders[i])

        for _ in range(head_steps):
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

            head_train_loss.append(loss.item())

        # ------------ Train backbone ------------

        backbone_parameters = [p for n, p in self.model.named_parameters() if 'image_classifier' not in n]

        optimizer = optim.SGD(backbone_parameters, lr=self.get_lr(t), momentum=self.cfg.momentum, weight_decay=self.cfg.weight_decay)

        backbone_train_loss = []

        train_iter = iter(self.train_loaders[i])

        for _ in range(backbone_steps):
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

            backbone_train_loss.append(loss.item())

        torch.save(self.model.state_dict(), self.local_model_path[i])

        return (np.mean(head_train_loss) + np.mean(backbone_train_loss)) / 2

    
    def aggregate(self):
        # --- ConFREE Aggregation Logic ---
        old_global_model_state = torch.load(self.global_model_path, weights_only=True)

        backbone_keys = [key for key in old_global_model_state if 'image_classifier' not in key]
        
        # 1. Calculate client update deltas for the backbone
        client_deltas = []
        with torch.no_grad():
            for i in range(self.cfg.num_clients):
                client_model_state = torch.load(self.local_model_path[i], weights_only=True)
                delta = {key: client_model_state[key] - old_global_model_state[key] for key in backbone_keys}
                client_deltas.append(delta)

        # 2. Adjust deltas to be conflict-free
        adjusted_deltas = copy.deepcopy(client_deltas)
        for i in range(self.cfg.num_clients):
            for j in range(i + 1, self.cfg.num_clients):
                delta_i_flat = torch.cat([p.view(-1) for p in client_deltas[i].values()])
                delta_j_flat = torch.cat([p.view(-1) for p in client_deltas[j].values()])
                
                dot_product = torch.dot(delta_i_flat, delta_j_flat)

                if dot_product < 0:
                    # Conflict: project updates to be orthogonal
                    norm_j_sq = torch.dot(delta_j_flat, delta_j_flat)
                    proj_factor_i = dot_product / (norm_j_sq + 1e-8)
                    for key in adjusted_deltas[i]:
                        adjusted_deltas[i][key] -= proj_factor_i * client_deltas[j][key]
                    
                    norm_i_sq = torch.dot(delta_i_flat, delta_i_flat)
                    proj_factor_j = dot_product / (norm_i_sq + 1e-8)
                    for key in adjusted_deltas[j]:
                        adjusted_deltas[j][key] -= proj_factor_j * client_deltas[i][key]
        
        # 3. Aggregate the conflict-free deltas
        conflict_free_update = {key: torch.zeros_like(old_global_model_state[key]) for key in backbone_keys}
        for i in range(self.cfg.num_clients):
            for key in backbone_keys:
                conflict_free_update[key] += adjusted_deltas[i][key] * self.weights[i]

        # 4. Apply the conflict-free update to the global model
        new_global_model_state = old_global_model_state
        for key in backbone_keys:
            new_global_model_state[key] += conflict_free_update[key]
        
        torch.save(new_global_model_state, self.global_model_path)


    def spread_model(self):
        global_model = torch.load(self.global_model_path, weights_only=True)
        for i in range(self.cfg.num_clients):
            local_model = torch.load(self.local_model_path[i], weights_only=True)

            updated_model = {}
            for k in self.model.state_dict().keys():
                updated_model[k] = local_model[k] if 'image_classifier' in k else global_model[k]
            
            torch.save(updated_model, self.local_model_path[i])


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