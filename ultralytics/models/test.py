from SR.sr_decoder_noBN_noD import Decoder
from SR.edsr import EDSR
import torch


sr_model = Decoder(in_channels=3, out_channels=3)

# 输入两个特征图（模拟高层和低层）
x = torch.randn(1, 3, 64, 64)              # 高层特征
low_level_feat = torch.randn(1, 3, 128, 128)  # 低层特征
factor = 2

out = sr_model(x, low_level_feat, factor)
print(out.shape)
