from model import RMSNorm, Attention, ModelConfig, DecoderLayer, Transformer
import torch
from model import precompute_freqs_cis, apply_rotary_emb

def prepare_args():
    args = ModelConfig()

    # 模拟输入数据
    batch_size = 1
    seq_len = 50  # 假设实际使用的序列长度为50
    dim = args.dim
    x = torch.rand(batch_size, seq_len, dim)  # 随机生成输入张量
    # freqs_cos = torch.rand(seq_len, dim // 2)  # 模拟cos频率，用于RoPE
    # freqs_sin = torch.rand(seq_len, dim // 2)  # 模拟sin频率，用于RoPE

    freqs_cos, freqs_sin = precompute_freqs_cis(dim//args.n_heads, seq_len)
    return args, x, freqs_cos, freqs_sin

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

def test_decoder_layer():
    args, x, freqs_cos, freqs_sin = prepare_args()

    # 创建LLaMADecoderLayer实例
    decoderlayer = DecoderLayer(0, args)

    out = decoderlayer(x, freqs_cos, freqs_sin)

    print(out.shape) # 形状和输入的x一样 [batch_size, seq_len, dim]

def test_model():
    args = ModelConfig()
    # LLaMA2Model.forward 接受两个参数，tokens和targets，其中tokens是输入的张量, 应为int类型
    x = torch.randint(0, 6144, (1, 50)) # [bs, seq_len]
    # 实例化LLaMA2Model
    model = Transformer(args=args)
    # 计算model的全部参数
    num_params = sum(p.numel() for p in model.parameters())
    print('Number of parameters:', num_params)

    out = model(x)
    print(out.logits.shape) # [batch_size, 1, vocab_size]

if __name__ == "__main__":
    # test_attention()
    # test_decoder_layer()
    test_model()