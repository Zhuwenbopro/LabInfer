#pragma once

#include <cstdint>
#include <vector>

using Tensor = float; // 之后再考虑咋办

using RequestId = uint64_t;
using Token = uint32_t;
using Position = uint32_t;
using StageId = uint32_t;
using BatchId = uint64_t;

class Request
{
    RequestId id_;
    std::vector<Token> inputs_tokens_;
};

class RequestState
{
    RequestId id_;
    Token generated_token;
    bool is_finished;
};

enum class DecodePhase
{
    PREFILL,
    DECODE
};

// 单个序列在批次中的输入信息
struct SequenceInputData
{
    RequestId request_id;               // 原始请求ID，用于追踪和结果关联
    std::vector<Token> token_ids;       // 当前轮次要处理的Token ID序列
    std::vector<Position> position_ids; // 对应的位置ID序列
    int sequence_length_q;              // Query 序列的当前长度 (不含 padding)  Prefill: prompt_len, Decode: prev_total_len + 1
    int context_length_kv;              // KV Cache 中该序列已有的上下文长度 Prefill: 0， Decode: prev_total_len
    // std::vector<BlockTableEntry> kv_block_table; // 该序列的KV Cache Block Table // Worker 根据这个表找到K和V的存储位置
    // DecodePhase phase;                        // 当前是预填充还是解码阶段
};

// 整个批次的输入
// TODO： Tensor 搞成支持多类型的
class ModelInputBatch
{
public:
    BatchId bid;
    std::vector<Token> tokens_batch;       // 包含所有序列在当前step需要处理的token ID。
    std::vector<Position> positions_batch; // 形状: 同 token_ids_batch

    // std::vector<std::vector<BlockTableEntry>> kv_cache_block_tables_batch;
    std::vector<int> sequence_lengths_q_batch; // 形状: [batch_size] 批次中每个序列的Query长度
    std::vector<int> context_lengths_kv_batch; // 形状: [batch_size] 批次中每个序列已有的KV Cache上下文长度
    std::vector<DecodePhase> phases_batch;     // 形状: [batch_size]
    std::vector<RequestId> request_ids_batch;  // 形状: [batch_size]
    int batch_size;                            // 当前批次中的序列数量
    int total_tokens_in_batch;                 // 当前批次中所有序列输入token的总数 (sum of sequence_lengths_q_batch)
    bool contains_prefill;                     // 批次中是否至少有一个请求处于PREFILL阶段
};

class ModelOutputBatch
{
public:
    BatchId bid;
    // StageId stage_id;                            // 阶段ID，表示当前输出的阶段
    std::vector<RequestId> req_ids;
    std::vector<Tensor> logits_batch;               // 也可以是最后的 logist 再说，再说
    std::vector<int> sequence_lengths_output;       // 形状: [batch_size] 批次中每个序列在 logits_batch 中对应的token数量。
};