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


class FedDRM():
    def __init__(self, cfg, train_loaders, test_loaders, num_data, logfile):
        self.cfg = cfg
        self.train_loaders, self.test_loaders = train_loaders, test_loaders
        self.weights = np.array(num_data) / np.sum(num_data)
        self.logfile = logfile

        self.model = select_model(cfg)
        self.local_model_path = [get_local_model_path(cfg, i) for i in range(cfg.num_clients)]
        self.global_model_path = get_global_model_path(cfg)
        self.criterion = nn.CrossEntropyLoss()

        self.record = {'train_loss': [],
                       'machine_train_loss': [],
                       'avg_acc': [], 
                       'machine_avg_acc': [],
                       'test_loss': [],
                       'machine_test_loss': [],
                       'sys_acc': []}
        
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
            machine_train_losses = []

            for i in range(self.cfg.num_clients):
                train_loss, machine_train_loss = self.local_train(i, t)
                train_losses.append(train_loss)
                machine_train_losses.append(machine_train_loss)

            self.aggregate()
            
            self.spread_model()

            # -----------------Record------------------
            self.record['train_loss'].append(np.sum(self.weights * train_losses))
            self.record['machine_train_loss'].append(np.sum(self.weights * machine_train_losses))

            avg_acc, machine_avg_acc, test_loss, machine_test_loss = self.test_averaged_performance()
            self.record['avg_acc'].append(avg_acc)
            self.record['machine_avg_acc'].append(machine_avg_acc)
            self.record['test_loss'].append(test_loss)
            self.record['machine_test_loss'].append(machine_test_loss)

            sys_acc = self.test_system_performance()
            self.record['sys_acc'].append(sys_acc)

            print(f"[Round: {t}] [Train Loss: {self.record['train_loss'][-1]:.4f}] [Machine Train Loss: {self.record['machine_train_loss'][-1]:.4f}] [Test Loss: {test_loss:.4f}] [Avg Accuracy: {avg_acc:.4f}] [Sys Accuracy: {sys_acc:.4f}] [Machine Avg Accuracy: {machine_avg_acc:.4f}] [Machine Test Loss: {machine_test_loss:.4f}]")
            self.logfile.write(f"[Round: {t}] [Train Loss: {self.record['train_loss'][-1]:.4f}] [Machine Train Loss: {self.record['machine_train_loss'][-1]:.4f}] [Test Loss: {test_loss:.4f}] [Avg Accuracy: {avg_acc:.4f}] [Sys Accuracy: {sys_acc:.4f}] [Machine Avg Accuracy: {machine_avg_acc:.4f}] [Machine Test Loss: {machine_test_loss:.4f}]\n")
            self.logfile.flush()

            self.writer.add_scalar('Train Loss', self.record['train_loss'][-1], t)
            self.writer.add_scalar('Machine Train Loss', self.record['machine_train_loss'][-1], t)
            self.writer.add_scalar('Test Loss', self.record['test_loss'][-1], t)
            self.writer.add_scalar('Avg Accuracy', self.record['avg_acc'][-1], t)
            self.writer.add_scalar('Sys Accuracy', self.record['sys_acc'][-1], t)
            self.writer.add_scalar('Machine Avg Accuracy', self.record['machine_avg_acc'][-1], t)
            self.writer.add_scalar('Machine Test Loss', self.record['machine_test_loss'][-1], t)

            np.save(self.record_path, self.record)

        print(f"{self.cfg.algo} training completed...")
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
        machine_train_loss = []

        train_iter = iter(self.train_loaders[i])

        for _ in range(self.cfg.local_steps):
            try:
                images, labels, client_labels = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loaders[i])
                images, labels, client_labels = next(train_iter)

            images, labels, client_labels = images.to(self.cfg.device), labels.to(self.cfg.device), client_labels.to(self.cfg.device)
            
            feature_logits, machine_logits = self.model(images)

            feature_loss = self.criterion(feature_logits, labels)
            machine_loss = self.criterion(machine_logits, client_labels)
            loss = self.cfg.feddrm_ratio * feature_loss + (1 - self.cfg.feddrm_ratio) * machine_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(feature_loss.item())
            machine_train_loss.append(machine_loss.item())

        torch.save(self.model.state_dict(), self.local_model_path[i])

        return np.mean(train_loss), np.mean(machine_train_loss)


    def aggregate(self):
        agg = {}
        for k, v in self.model.state_dict().items():
            agg[k] = torch.zeros_like(v)

        for i in range(self.cfg.num_clients):
            w = torch.load(self.local_model_path[i], weights_only=True)

            for k in self.model.state_dict().keys():
                if 'image_classifier' not in k:
                    agg[k] += w[k] * self.weights[i]
        
        torch.save(agg, self.global_model_path)


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
        machine_accs = np.zeros(self.cfg.num_clients)
        losses = np.zeros(self.cfg.num_clients)
        machine_losses = np.zeros(self.cfg.num_clients)
        
        with torch.no_grad():
            for i in range(self.cfg.num_clients):
                self.model.load_state_dict(torch.load(self.local_model_path[i], weights_only=True))
                self.model.eval()

                acc = 0
                machine_acc = 0
                total = 0
                per_losses = []
                per_machine_losses = []
                for images, labels, client_labels in self.test_loaders[i]:
                    images, labels, client_labels = images.to(self.cfg.device), labels.to(self.cfg.device), client_labels.to(self.cfg.device)

                    feature_logits, machine_logits = self.model(images)

                    feature_loss = self.criterion(feature_logits, labels)
                    machine_loss = self.criterion(machine_logits, client_labels)

                    acc += (feature_logits.argmax(dim=1) == labels).sum().item()
                    machine_acc += (machine_logits.argmax(dim=1) == client_labels).sum().item()
                    total += labels.size(0)

                    per_losses.append(feature_loss.item())
                    per_machine_losses.append(machine_loss.item())

                accs[i] = acc / total
                machine_accs[i] = machine_acc / total
                losses[i] = np.mean(per_losses)
                machine_losses[i] = np.mean(per_machine_losses)

        return np.sum(self.weights * accs), np.sum(self.weights * machine_accs), np.sum(self.weights * losses), np.sum(self.weights * machine_losses)


    def test_system_performance(self):
        acc = 0
        total = 0

        local_models = [torch.load(path, weights_only=True) for path in self.local_model_path]
        global_model = torch.load(self.global_model_path, weights_only=True)
        log_const = torch.log(torch.tensor(self.weights)).to(self.cfg.device)

        with torch.no_grad():
            for test_loader in self.test_loaders:
                for images, labels, _ in test_loader:
                    images, labels = images.to(self.cfg.device), labels.to(self.cfg.device)

                    self.model.load_state_dict(global_model)
                    self.model.eval()
                    _, machine_logits = self.model(images)
                    density_logits = machine_logits - log_const
                    pred_clients = density_logits.argmax(dim=1)

                    all_feature_logits = []
                    for i in range(self.cfg.num_clients):
                        self.model.load_state_dict(local_models[i])
                        self.model.eval()
                        
                        feature_logits, _ = self.model(images)
                        all_feature_logits.append(feature_logits)
                    
                    # size: [batch_size, num_clients, num_classes]
                    all_feature_logits = torch.stack(all_feature_logits, dim=1)
                    num_classes = all_feature_logits.shape[2]

                    # idx: [batch_size, 1, 1] -> [batch_size, 1, num_classes]
                    idx = pred_clients.view(-1, 1, 1).expand(-1, 1, num_classes)
                    
                    # final_logits: [batch_size, num_classes]
                    final_logits = all_feature_logits.gather(dim=1, index=idx).squeeze(dim=1)
                    
                    final_pred = final_logits.argmax(dim=1)
                    acc += (final_pred == labels).sum().item()
                    total += labels.size(0)

        return acc / total