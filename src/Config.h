#include <string>
class Config{
public:
    int dim;            // 模型的维度 D
    int hidden_dim;     // 隐藏层维度 DD
    int n_layers;       // 层数 NL
    int n_heads;        // 注意力头数 QHN, HN, HD = 48
    int n_kv_heads;     // Key-Value头数 KVHN = 6
    int vocab_size;     // 词汇表大小 VS
    int max_seq_len;    // 最大序列长度 M
};