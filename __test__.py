import torch
import torch.nn as nn

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    预计算频率矩阵（用于生成 cos 和 sin）
    dim: 向量维度（注意是 head_dim）
    end: 最大序列长度
    """
    # 1. 计算 theta_i: [dim/2]
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    
    # 2. 生成位置索引 m: [end]
    t = torch.arange(end, device=freqs.device) 
    
    # 3. 外积得到 m * theta_i: [end, dim/2]
    freqs = torch.outer(t, freqs).float()
    
    # 4. 为了方便计算，转为复数极坐标形式 e^{i*m*theta}
    # 这在代码中对应后续的 cos 和 sin 
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    """
    将旋转编码应用到 Query 和 Key 上
    xq, xk: [batch_size, seq_len, n_heads, head_dim]
    """
    # 1. 将最后的维度拆分为复数形式 (两两分组)
    # [..., head_dim] -> [..., head_dim/2, 2] -> 转为复数
    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # 2. 广播频率矩阵，使其与 xq 形状匹配
    # freqs_cis: [seq_len, head_dim/2] -> [1, seq_len, 1, head_dim/2]
    freqs_cis = freqs_cis.view(1, xq.shape[1], 1, -1)

    # 3. 复数乘法即代表旋转：(a+bi)(c+di) = (ac-bd) + (ad+bc)i
    xq_out = torch.view_as_real(xq_complex * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_complex * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)

# --- 测试用例 ---
batch, seq_len, n_heads, head_dim = 2, 512, 8, 64
xq = torch.randn(batch, seq_len, n_heads, head_dim)
xk = torch.randn(batch, seq_len, n_heads, head_dim)

# 预计算
freqs_cis = precompute_freqs_cis(head_dim, seq_len)
# 应用旋转
xq_rotated, xk_rotated = apply_rotary_emb(xq, xk, freqs_cis)

print(f"输出形状: {xq_rotated.shape}") # [2, 512, 8, 64]