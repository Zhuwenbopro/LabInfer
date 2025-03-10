import torch
import torch.nn.functional as F
import numpy as np

def reference_attention_no_scale(Q_2d, K_2d, V_2d):
    r"""
    使用 PyTorch 2.0+ 的 scaled_dot_product_attention，但不做 sqrt(d) 缩放。
    即 scale=1.0 => QK^T -> softmax -> V

    参数:
      Q_2d: [N, d]
      K_2d: [N, d]
      V_2d: [N, d]
    输出:
      out_2d: [N, d]
    """
    # expand 维度变成 [batch=1, heads=1, seqLen=N, headDim=d]
    Q_4d = Q_2d.unsqueeze(0).unsqueeze(0)  # [1,1,N,d]
    K_4d = K_2d.unsqueeze(0).unsqueeze(0)  # [1,1,N,d]
    V_4d = V_2d.unsqueeze(0).unsqueeze(0)  # [1,1,N,d]

    # 调用 scaled_dot_product_attention, 指定 scale=1.0 表示不做缩放
    out_4d = F.scaled_dot_product_attention(Q_4d, K_4d, V_4d,
                                            attn_mask=None,
                                            dropout_p=0.0,
                                            is_causal=False,
                                            scale=1.0)
    out_2d = out_4d.squeeze(0).squeeze(0)   # [N, d]
    return out_2d

def main():
    # 设置较大的测试规模（例如 N=1024, d=2048/32=64）
    N, d = 1024, 2048

    # 初始化数据：hQ[i] = 0.01 * (i % d) 等
    hQ = torch.zeros(N * d, dtype=torch.float32)
    hK = torch.zeros(N * d, dtype=torch.float32)
    hV = torch.zeros(N * d, dtype=torch.float32)
    for i in range(N * d):
        hQ[i] = 0.01 * (i % d)
        hK[i] = 0.02 * (i % d)
        hV[i] = 0.03 * (i % d)

    # reshape 成 [N, d]
    Q = hQ.reshape(N, d)
    K = hK.reshape(N, d)
    V = hV.reshape(N, d)

    # 计算不带缩放的 attention 结果
    out_no_scale = reference_attention_no_scale(Q, K, V)
    print("No-scale attention =>")
    print(out_no_scale)

    # 保存结果到本地文件（文本格式）
    # 每行保存一个 [d] 维向量，元素用空格分隔
    np.savetxt("ref_attention.txt", out_no_scale.cpu().numpy(), fmt="%.6f")
    print("Attention result saved to ref_attention.txt.")

if __name__ == "__main__":
    main()
