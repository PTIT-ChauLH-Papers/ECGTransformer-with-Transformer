import sys
from tensorboard import summary
from torch import nn
import torch
import torch.nn.functional as F
from ResidualN import ResidualBasicBlock

class MultiScaleCNN(nn.Module):
    def __init__(self, input_channels, kernel_size, dropout_rate, n_heads, output_dim, attention_num_layers):
        super(MultiScaleCNN, self).__init__()
        
        self.output_dim = output_dim
        # First scale
        self.conv1_1 = nn.Conv1d(input_channels, 32, kernel_size=kernel_size, padding=kernel_size // 2, stride=1, bias=False)
        # Second scale
        self.conv2_1 = nn.Conv1d(input_channels, 32, kernel_size=kernel_size+2, padding= (kernel_size+2) //2, stride=1, bias=False)
        # Third scale
        self.conv3_1 = nn.Conv1d(input_channels, 32, kernel_size=kernel_size+6, padding= (kernel_size+6) // 2, stride=1, bias=False)
        
        self.batch_norm = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.max_pool_1D = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.dropout = nn.Dropout(p=dropout_rate)
        
        self.conv_block_2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, padding=4, stride=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        
        if output_dim >= 128:                               
            self.conv_block_3 = nn.Sequential(
                nn.Conv1d(64, 128, kernel_size=8, padding=4, stride=1, bias=False),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
            )
        
        if output_dim >= 256:
            self.conv_block_4 = nn.Sequential(
                nn.Conv1d(128, 256, kernel_size=8, padding=4, stride=1, bias=False),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
            )
        
        if output_dim >= 512:
            self.conv_block_5 = nn.Sequential(
                nn.Conv1d(256, 512, kernel_size=8, padding=4, stride=1, bias=False),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
            )
        
        if output_dim >= 1024:
            self.conv_block_6 = nn.Sequential(
                nn.Conv1d(512, output_dim, kernel_size=8, padding=4, stride=1, bias=False),
                nn.BatchNorm1d(output_dim),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
            )
        
        self.inplanes = output_dim
        crm_stride = 1
        downsample = None
        if crm_stride != 1 or self.inplanes != self.inplanes * ResidualBasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(output_dim, output_dim * ResidualBasicBlock.expansion, kernel_size=1, stride=crm_stride, bias=False),
                nn.BatchNorm1d(output_dim * ResidualBasicBlock.expansion),
            )
        self.crm = nn.Sequential(
            ResidualBasicBlock(output_dim, output_dim, stride=crm_stride, downsample=downsample),
            ResidualBasicBlock(output_dim, output_dim),
            ResidualBasicBlock(output_dim, output_dim),
        )
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=output_dim, nhead=n_heads, batch_first=True, activation=F.relu)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=attention_num_layers)

        self.aap = nn.AdaptiveAvgPool1d(1)
        self.clf = nn.Linear(output_dim, 4)
        
    def forward(self, x):
        #Multi-scale Convolutions
        x_1 = self.conv1_1(x)
        x_2 = self.conv2_1(x)
        x_3 = self.conv3_1(x)
                    
        x_mean = (x_1 + x_2 + x_3) / 3
        
        x = self.batch_norm(x_mean)
        x = self.relu(x)
        x = self.max_pool_1D(x)
        x = self.dropout(x)
        
        x = self.conv_block_2(x)
        
        # More convolutional blocks based on output_dim
        if (self.output_dim >= 128):
            x = self.conv_block_3(x)
        if (self.output_dim >= 256):
            x = self.conv_block_4(x)
        if (self.output_dim >= 512):
            x = self.conv_block_5(x)
        if (self.output_dim >= 1024):
            x = self.conv_block_6(x)
        
        x = self.crm(x)
        x = x.permute(0, 2, 1)  # Reshape for transformer (batch_size, seq_length, feature_dim)
        # Bi-directional Transformer
        x1 = self.transformer_encoder(x)
        x2 = self.transformer_encoder(torch.flip(x,[2]))
        x = x1
        x = x.permute(0, 2, 1)

        x = self.aap(x)
        x_flat = x.reshape(x.shape[0], -1)
        x_out = self.clf(x_flat)
        return x_out