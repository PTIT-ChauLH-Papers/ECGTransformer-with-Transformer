from torch import nn

class MultiScaleCNN(nn.Module):
    def __init__(self, kernel_sizes, upcoming_kernel_size, output_dim, dropout_rate):
        super(MultiScaleCNN, self).__init__()
        
        self.output_dim = output_dim
        # First scale
        self.conv1_1 = nn.Conv1d(1, 32, kernel_size=kernel_sizes[0], padding=kernel_sizes // 2, stride=1, bias=False)
        # Second scale
        self.conv2_1 = nn.Conv1d(1, 32, kernel_size=kernel_sizes[1], padding= (kernel_sizes[1]) //2, stride=1, bias=False)
        # Third scale
        self.conv3_1 = nn.Conv1d(1, 32, kernel_size=kernel_sizes[2], padding= (kernel_sizes[2]) // 2, stride=1, bias=False)
        
        self.batch_norm = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.max_pool_1D = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.dropout = nn.Dropout(p=dropout_rate)
        
        self.conv_block_2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=upcoming_kernel_size, padding=upcoming_kernel_size // 2, stride=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        
        if output_dim >= 128:                               
            self.conv_block_3 = nn.Sequential(
                nn.Conv1d(64, 128, kernel_size=upcoming_kernel_size, padding=upcoming_kernel_size // 2, stride=1, bias=False),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
            )
        
        if output_dim >= 256:
            self.conv_block_4 = nn.Sequential(
                nn.Conv1d(128, 256, kernel_size=upcoming_kernel_size, padding=upcoming_kernel_size // 2, stride=1, bias=False),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
            )
        
        if output_dim >= 512:
            self.conv_block_5 = nn.Sequential(
                nn.Conv1d(256, 512, kernel_size=upcoming_kernel_size, padding=upcoming_kernel_size // 2, stride=1, bias=False),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
            )
        
        if output_dim >= 1024:
            self.conv_block_6 = nn.Sequential(
                nn.Conv1d(512, output_dim, kernel_size=upcoming_kernel_size, padding=upcoming_kernel_size // 2, stride=1, bias=False),
                nn.BatchNorm1d(output_dim),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
            )
        
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
        
       