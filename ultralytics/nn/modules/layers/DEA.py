import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv

class DEA(nn.Module):
    """DEA fusion block with optional gating and adaptive single-modality routing."""

    ENABLE_GATE = True
    ENABLE_ADAPTIVE_PATH = True

    def __init__(
        self, channel=512, kernel_size=80, p_kernel=None, m_kernel=None, reduction=16
    ):
        super().__init__()
        if p_kernel is None:
            p_kernel = 64

        self.deca = DECA(channel, kernel_size, p_kernel, reduction)
        self.depa = DEPA(channel, m_kernel)
        self.act = nn.Sigmoid()

        self.alpha_r = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel, 1),
            nn.Sigmoid(),
        )
        self.alpha_t = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel, 1),
            nn.Sigmoid(),
        )

        self.proj_r = nn.Conv2d(channel, channel, 1)
        self.proj_t = nn.Conv2d(channel, channel, 1)
        # Distribution alignment: BN+Sigmoid wraps the single-modality projection
        # so its output range matches F_bi (sigmoid output in [0,1]). Without this,
        # neck BN running stats (learned on F_bi) mis-normalize single-modality input.
        self.proj_r_norm = nn.BatchNorm2d(channel)
        self.proj_t_norm = nn.BatchNorm2d(channel)

        self.modality_flags = None

    def set_modality_flags(self, flags):
        self.modality_flags = flags

    def __setstate__(self, state):
        # Backward compat: old checkpoints saved before BN was added to DEA
        # don't have proj_r_norm / proj_t_norm. Fresh-init them on load.
        super().__setstate__(state)
        if "proj_r_norm" not in self._modules and hasattr(self, "proj_r"):
            channel = self.proj_r.out_channels
            self.proj_r_norm = nn.BatchNorm2d(channel)
            self.proj_t_norm = nn.BatchNorm2d(channel)

    def forward(self, x):
        F_r, F_t = x
        B = F_r.shape[0]

        if self.modality_flags is not None:
            m_t, m_r = self.modality_flags
            m_t  = m_t.to(F_t.device).view(B, 1, 1, 1).float()
            m_r = m_r.to(F_r.device).view(B, 1, 1, 1).float()
        else:
            m_t  = torch.ones(B, 1, 1, 1, device=F_t.device)
            m_r = torch.ones(B, 1, 1, 1, device=F_r.device)

        is_dual = (m_t * m_r).view(B) > 0.5  # [B]

        if is_dual.all():
            if self.ENABLE_GATE:
                F_r_gated = m_r * self.alpha_r(F_r).contiguous() * F_r
                F_t_gated  = m_t  * self.alpha_t(F_t).contiguous()   * F_t
                result_r, result_t = self.depa(self.deca([F_r_gated, F_t_gated]))
            else:
                result_r, result_t = self.depa(self.deca([F_r, F_t]))
            F_bi = self.act(result_r + result_t)

        if self.ENABLE_ADAPTIVE_PATH:
            F_r_out = torch.sigmoid(self.proj_r_norm(self.proj_r(F_r)))
            F_t_out  = torch.sigmoid(self.proj_t_norm(self.proj_t(F_t)))

            if is_dual.all():
                F_out = F_bi
            elif not is_dual.any():
                F_out = m_t * (1 - m_r) * F_t_out + (1 - m_t) * m_r * F_r_out
            else:
                _r, _t = self.depa(self.deca([
                    m_r * self.alpha_r(F_r).contiguous() * F_r if self.ENABLE_GATE else F_r,
                    m_t  * self.alpha_t(F_t).contiguous()   * F_t  if self.ENABLE_GATE else F_t,
                ]))
                F_bi = self.act(_r + _t)
                F_out = (m_t * m_r * F_bi
                         + m_t  * (1 - m_r) * F_t_out
                         + (1 - m_t) * m_r  * F_r_out)
        else:
            if not is_dual.all():
                _r, _t = self.depa(self.deca([
                    m_r * self.alpha_r(F_r).contiguous() * F_r if self.ENABLE_GATE else F_r,
                    m_t  * self.alpha_t(F_t).contiguous()   * F_t  if self.ENABLE_GATE else F_t,
                ]))
                F_bi = self.act(_r + _t)
            F_out = F_bi

        return F_out

class CMFT_DECA(nn.Module):
    """
    CMFT-DECA: Multi-modal Fusion Transformer with DE-Octuple Cross Attention
    稳定版本，避免所有动态计算
    """

    def __init__(self, channel, kernel_size, p_kernel, reduction):
        super().__init__()
        self.channel = channel

        # ===== 1) 所有参数都使用固定值 =====
        self.pool_size = 4  # 固定池化大小

        # 设置固定的嵌入维度
        self.embed_dim = 128  # 固定值，不依赖输入参数
        self.num_heads = 8    # 固定头数

        # 确保embed_dim能被num_heads整除
        while self.embed_dim % self.num_heads != 0:
            self.embed_dim += 1

        # ===== 2) 使用最简单稳定的模块 =====
        # 输入嵌入层
        self.r_project = nn.Conv2d(channel, self.embed_dim, 1, bias=False)
        self.t_project = nn.Conv2d(channel, self.embed_dim, 1, bias=False)

        # 简化的交叉注意力
        self.cross_attn_r = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            batch_first=True,
            dropout=0.0  # 固定为0
        )

        self.cross_attn_t = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            batch_first=True,
            dropout=0.0  # 固定为0
        )

        # 固定的前馈网络
        self.ffn_r = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.GELU(),
            nn.Linear(self.embed_dim * 2, self.embed_dim)
        )

        self.ffn_t = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.GELU(),
            nn.Linear(self.embed_dim * 2, self.embed_dim)
        )

        # 层归一化
        self.norm_r = nn.LayerNorm(self.embed_dim)
        self.norm_t = nn.LayerNorm(self.embed_dim)

        # 输出投影
        self.out_r = nn.Conv2d(self.embed_dim, channel, 1, bias=False)
        self.out_t = nn.Conv2d(self.embed_dim, channel, 1, bias=False)

        # 门控机制
        self.gate_r = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 1),
            nn.Sigmoid()
        )
        self.gate_t = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        """
        完全静态的前向传播，避免任何条件分支和动态计算
        """
        r, t = inputs
        B, C, H, W = r.shape

        # ===== 1) 特征投影 =====
        r_embed = self.r_project(r)  # [B, embed_dim, H, W]
        t_embed = self.t_project(t)     # [B, embed_dim, H, W]

        # ===== 2) 固定大小的空间降采样 =====
        target_size = self.pool_size
        r_pool = F.adaptive_avg_pool2d(r_embed, target_size)  # [B, embed_dim, 4, 4]
        t_pool = F.adaptive_avg_pool2d(t_embed, target_size)    # [B, embed_dim, 4, 4]

        # ===== 3) 转换为序列格式 =====
        seq_len = target_size * target_size  # 16
        r_seq = r_pool.flatten(2).transpose(1, 2)  # [B, 16, embed_dim]
        t_seq = t_pool.flatten(2).transpose(1, 2)    # [B, 16, embed_dim]

        # ===== 4) 交叉注意力 =====
        r_norm = self.norm_r(r_seq)
        t_norm = self.norm_t(t_seq)

        # R受T影响
        r_attended, _ = self.cross_attn_r(r_norm, t_norm, t_norm)
        r_out = r_seq + r_attended
        r_out = r_out + self.ffn_r(r_out)

        # T受R影响
        t_attended, _ = self.cross_attn_t(t_norm, r_norm, r_norm)
        t_out = t_seq + t_attended
        t_out = t_out + self.ffn_t(t_out)

        # ===== 5) 恢复空间维度 =====
        r_spatial = r_out.transpose(1, 2).reshape(B, self.embed_dim, target_size, target_size)
        t_spatial = t_out.transpose(1, 2).reshape(B, self.embed_dim, target_size, target_size)

        # ===== 6) 上采样回原始尺寸 =====
        r_up = F.interpolate(r_spatial, size=(H, W), mode='bilinear', align_corners=False)
        t_up = F.interpolate(t_spatial, size=(H, W), mode='bilinear', align_corners=False)

        # ===== 7) 输出投影 =====
        r_feat = self.out_r(r_up)
        t_feat = self.out_t(t_up)

        # ===== 8) 门控融合 =====
        gate_r = self.gate_r(torch.cat([r, t], dim=1))
        gate_t = self.gate_t(torch.cat([t, r], dim=1))

        result_r = r + gate_r * r_feat
        result_t = t + gate_t * t_feat

        return result_r, result_t

# 保持兼容性
class DECA(CMFT_DECA):
    pass

class DEPA(nn.Module):
    """x0 --> reflective,  x1 --> transmissive"""

    def __init__(self, channel=512, m_kernel=None):
        super().__init__()

        # 使用固定的卷积核配置
        if m_kernel is None:
            k1, k2 = 3, 7
        else:
            k1, k2 = m_kernel[0] if len(m_kernel) > 0 else 3, m_kernel[1] if len(m_kernel) > 1 else 7

        # 所有卷积层都使用固定配置，避免动态padding计算
        self.cv_v1 = nn.Conv2d(channel, 1, k1, padding=k1//2, bias=False)
        self.cv_v2 = nn.Conv2d(channel, 1, k2, padding=k2//2, bias=False)
        self.cv_i1 = nn.Conv2d(channel, 1, k1, padding=k1//2, bias=False)
        self.cv_i2 = nn.Conv2d(channel, 1, k2, padding=k2//2, bias=False)

        self.conv1 = nn.Conv2d(2, 1, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(2, 1, 3, padding=1, bias=False)

        self.compress1 = nn.Conv2d(channel, 1, 1, bias=False)
        self.compress2 = nn.Conv2d(channel, 1, 1, bias=False)

        self.act = nn.Sigmoid()

    def forward(self, x):
        x_r, x_t = x

        # 空间注意力权重计算
        r1 = self.cv_v1(x_r)
        r2 = self.cv_v2(x_r)
        t1 = self.cv_i1(x_t)
        t2 = self.cv_i2(x_t)

        w_r = self.conv1(torch.cat([r1, r2], dim=1))
        w_t = self.conv2(torch.cat([t1, t2], dim=1))

        # 全局权重
        glob = self.act(self.compress1(x_r) + self.compress2(x_t))

        # 最终权重
        w_r = self.act(glob + w_r)
        w_t = self.act(glob + w_t)

        # 交叉调制
        result_r = x_r * w_t
        result_t = x_t * w_r

        return result_r, result_t