from torch import nn
from MultiScaleCNN import MultiScaleCNN
from ResidualN import ResidualBasicBlock
from torch.nn import functional as F

# from ECA import EcaLayer
# from CBAM1D import CBAM1D

class Main(nn.Module):
    def __init__(self, kernel_sizes, upcoming_kernel_size, out_channels, num_se_res_blocks, reduction, n_heads, attention_num_layers, dropout_rate):
        super(Main, self).__init__()
        
        # Multi-scale convolutional layers with different kernel sizes
        self.multi_scale_cnn = MultiScaleCNN(kernel_sizes=kernel_sizes, 
                                            upcoming_kernel_size=upcoming_kernel_size,
                                            output_dim=out_channels,
                                            dropout_rate=dropout_rate
                                            )
                
        # CRM module
        self.inplanes = out_channels
        crm_stride = 1
        downsample = None
        if crm_stride != 1 or self.inplanes != self.inplanes * ResidualBasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(out_channels, out_channels * ResidualBasicBlock.expansion, kernel_size=1, stride=crm_stride, bias=False),
                nn.BatchNorm1d(out_channels * ResidualBasicBlock.expansion),
            )
        
        self.crm = nn.Sequential(
            ResidualBasicBlock(
                out_channels,
                out_channels,
                stride=crm_stride,
                downsample=downsample
            ),
            *[
                ResidualBasicBlock(out_channels, out_channels, reduction=reduction)
                for _ in range(1, num_se_res_blocks + 1)
            ]
        )
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=out_channels, nhead=n_heads, batch_first=True, activation=F.relu)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=attention_num_layers)

        self.aap = nn.AdaptiveAvgPool1d(1)
        self.clf = nn.Linear(out_channels, 4)
        
        # # Transformer encoder for bi-directional attention
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.out_channels, nhead=nheads, batch_first=True)
        # self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        self.aap = nn.AdaptiveAvgPool1d(1)
        self.clf = nn.Linear(self.out_channels, 4)
        
    def forward(self, x):
        # Multi-scale feature extraction
        x = self.multi_scale_cnn(x)
        
        # # Channel recalibration module
        x = self.crm(x)
        
        # # Transformer
        x = x.permute(0, 2, 1)   
        x = self.transformer_encoder(x)
        x = x.permute(0, 2, 1)

        x = self.aap(x)
        x = x.reshape(x.shape[0], -1)
        x_out = self.clf(x)
        return x_out