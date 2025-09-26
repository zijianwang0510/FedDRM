import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNetGN(nn.Module):
    def __init__(self, cfg):
        super(LeNetGN, self).__init__()
        self.cfg = cfg
        
        num_classes = 10
        feature_dim = 256 # The output dimension of the main backbone
        decoupled_feature_dim = 128 # The dimension before the final classifier

        # --- Factory functions to create new, independent module instances ---
        def conv1_block():
            return nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5),
                nn.GroupNorm(num_groups=8, num_channels=32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

        def conv2_block():
            return nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
                nn.GroupNorm(num_groups=16, num_channels=64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            
        def fc_trunk_block():
            # This is the main FC part of the backbone
            return nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=64 * 5 * 5, out_features=512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(in_features=512, out_features=feature_dim),
                nn.LayerNorm(feature_dim),
                nn.ReLU()
            )
        
        def decoupled_block():
            return nn.Sequential(
                nn.Linear(in_features=feature_dim, out_features=decoupled_feature_dim),
                nn.LayerNorm(decoupled_feature_dim),
                nn.ReLU()
            )

        # --- Personalized Layer ---
        # CRITICAL: This is the ONLY personalized part of the network.
        self.image_classifier = nn.Linear(in_features=decoupled_feature_dim, out_features=num_classes)

        # --- Shared Machine Classifier ---
        # This entire path, including the final linear layer, is shared and aggregated.
        self.machine_classifier = nn.Linear(in_features=decoupled_feature_dim, out_features=cfg.num_clients)


        # --- Network construction based on the sharing strategy ---
        if self.cfg.share_level == 'no_share':
            self.feature_backbone = nn.Sequential(conv1_block(), conv2_block(), fc_trunk_block(), decoupled_block())
            # For machine path, its head is integrated into the machine_classifier module
            self.machine_backbone = nn.Sequential(conv1_block(), conv2_block(), fc_trunk_block(), decoupled_block())
        
        elif self.cfg.share_level == 'shallow':
            self.shared_backbone = conv1_block()
            self.feature_branch = nn.Sequential(conv2_block(), fc_trunk_block(), decoupled_block())
            self.machine_branch = nn.Sequential(conv2_block(), fc_trunk_block(), decoupled_block())

        elif self.cfg.share_level == 'mid':
            self.shared_backbone = nn.Sequential(conv1_block(), conv2_block())
            self.feature_branch = nn.Sequential(fc_trunk_block(), decoupled_block())
            self.machine_branch = nn.Sequential(fc_trunk_block(), decoupled_block())

        elif self.cfg.share_level == 'deep':
            # The full backbone is shared
            self.shared_backbone = nn.Sequential(conv1_block(), conv2_block(), fc_trunk_block())
            self.feature_branch = decoupled_block()
            self.machine_branch = decoupled_block()
        else:
            raise ValueError(f"Unknown share_level: {self.cfg.share_level}")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.share_level == 'no_share':
            image_embedding = self.feature_backbone(x)
            machine_embedding = self.machine_backbone(x)
        elif self.cfg.share_level == 'shallow':
            shared_embedding = self.shared_backbone(x)
            image_embedding = self.feature_branch(shared_embedding)
            machine_embedding = self.machine_branch(shared_embedding)
        elif self.cfg.share_level == 'mid':
            shared_embedding = self.shared_backbone(x)
            image_embedding = self.feature_branch(shared_embedding)
            machine_embedding = self.machine_branch(shared_embedding)
        else: # deep
            shared_embedding = self.shared_backbone(x)
            image_embedding = self.feature_branch(shared_embedding)
            machine_embedding = self.machine_branch(shared_embedding)

        # Get the final logits
        image_logits = self.image_classifier(image_embedding)
        machine_logits = self.machine_classifier(machine_embedding)

        return image_logits, machine_logits

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=16, num_channels=planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=16, num_channels=planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=16, num_channels=self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_Backbone(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNet_Backbone, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=16, num_channels=64)
        # ---------------------------

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


class ResNet(nn.Module):
    def __init__(self, cfg):
        super(ResNet, self).__init__()
        self.cfg = cfg
        self.backbone = ResNet_Backbone(BasicBlock, [2, 2, 2, 2])

        shared_feature_dim = 512
        decoupled_feature_dim = 256

        if cfg.dataset == 'cifar10':
            num_classes = 10
        elif cfg.dataset == 'cifar20':
            num_classes = 20
        elif cfg.dataset == 'cifar100':
            num_classes = 100

        self.image_fc = nn.Linear(in_features=shared_feature_dim, out_features=decoupled_feature_dim)
        self.image_ln = nn.LayerNorm(decoupled_feature_dim)
        self.image_classifier = nn.Linear(in_features=decoupled_feature_dim, out_features=num_classes)

        if 'feddrm' in cfg.algo:
            self.machine_fc = nn.Linear(in_features=shared_feature_dim, out_features=decoupled_feature_dim)
            self.machine_ln = nn.LayerNorm(decoupled_feature_dim)
            self.machine_classifier = nn.Linear(in_features=decoupled_feature_dim, out_features=cfg.num_clients)

        self._initialize_weights()

    def forward(self, x: torch.Tensor, FEATURE_ONLY=False):
        shared_feature = self.backbone(x)

        if FEATURE_ONLY == True:
            return shared_feature

        image_feature = F.relu(self.image_ln(self.image_fc(shared_feature)))
        image_logits = self.image_classifier(image_feature)

        if 'feddrm' in self.cfg.algo:
            machine_feature = F.relu(self.machine_ln(self.machine_fc(shared_feature)))
            machine_logits = self.machine_classifier(machine_feature)
            return image_logits, machine_logits
        else:
            return image_logits

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.gn2.weight, 0)
