import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleFCN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, use_batch_norm=False):
        super(SimpleFCN, self).__init__()
        self.input_size = input_size
        self.use_batch_norm = use_batch_norm
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layers(x)


class MediumFCN(nn.Module):
    def __init__(self, input_size, output_size, use_dropout=False):
        super(MediumFCN, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2) if use_dropout else nn.Identity()
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class DeepFCN(nn.Module):
    def __init__(self, input_size, output_size, use_dropout=False):
        super(DeepFCN, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2) if use_dropout else nn.Identity()
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.fc6(x)
        return x


def get_mnist_model(model_type='simple', pretrained=False):
    input_size = 28 * 28
    output_size = 10
    
    if model_type == 'tiny':
        # 超小模型：仅1个隐藏层，32个神经元
        model = SimpleFCN(input_size, [32], output_size)
    elif model_type == 'simple':
        # 小模型：2个隐藏层 [64, 32]（优化后）
        model = SimpleFCN(input_size, [64, 32], output_size)
    elif model_type == 'medium':
        # 中等模型：减少神经元数量
        model = SimpleFCN(input_size, [256, 128, 64], output_size)
    elif model_type == 'deep':
        model = DeepFCN(input_size, output_size)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def get_cifar10_model(model_type='simple', pretrained=False):
    input_size = 3 * 32 * 32
    output_size = 10
    
    if model_type == 'tiny':
        # 超小模型：仅1个隐藏层，32个神经元
        model = SimpleFCN(input_size, [32], output_size)
    elif model_type == 'simple':
        # 小模型：减少神经元数量
        model = SimpleFCN(input_size, [128, 64], output_size)
    elif model_type == 'medium':
        model = SimpleFCN(input_size, [256, 128, 64], output_size)
    elif model_type == 'deep':
        model = DeepFCN(input_size, output_size)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model
