from model import RMSNorm, Attention, ModelConfig
import torch
from model import precompute_freqs_cis, apply_rotary_emb

def test_attention():
    # 创建Attention实例
    args = ModelConfig()
    attention_model = Attention(args)

    # 模拟输入数据
    batch_size = 1
    seq_len = 50  # 假设实际使用的序列长度为50
    dim = args.dim
    x = torch.rand(batch_size, seq_len, dim)  # 随机生成输入张量
    # freqs_cos = torch.rand(seq_len, dim // 2)  # 模拟cos频率，用于RoPE
    # freqs_sin = torch.rand(seq_len, dim // 2)  # 模拟sin频率，用于RoPE

    freqs_cos, freqs_sin = precompute_freqs_cis(dim//args.n_heads, seq_len)

    # 运行Attention模型
    output = attention_model(x, freqs_cos, freqs_sin)

    # attention出来之后的形状 依然是[batch_size, seq_len, dim]
    print("Output shape:", output.shape)

if __name__ == "__main__":
    test_attention()
