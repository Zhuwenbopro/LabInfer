#include "model/ParamLoader.h"
#include <regex>
#define SAFETENSORS_IMPLEMENTATION
#include "safetensors.h"

float f16_to_f32(uint16_t h) {
    uint16_t sign = (h & 0x8000) >> 15;
    uint16_t exponent = (h & 0x7C00) >> 10;
    uint16_t fraction = h & 0x03FF;

    if (exponent == 0) {
        // Subnormal number
        if (fraction == 0) {
            return sign ? -0.0f : 0.0f;
        } else {
            return (sign ? -1 : 1) * ldexp(fraction / 1024.0f, -14);
        }
    } else if (exponent == 0x1F) {
        // Infinity or NaN
        if (fraction == 0) {
            return sign ? -INFINITY : INFINITY;
        } else {
            return NAN;
        }
    } else {
        // Normalized number
        return (sign ? -1 : 1) * ldexp(1.0f + fraction / 1024.0f, exponent - 15);
    }
}

float bf16_to_f32(uint16_t bf16_value) {
    // 创建一个 32 位的无符号整数变量
    uint32_t f32_value = uint32_t(bf16_value) << 16; // 左移 16 位

    // 将 uint32_t 的位模式解释为 float
    float result;
    std::memcpy(&result, &f32_value, sizeof(result));

    return result;
}

#define die(...) do{printf(__VA_ARGS__); fputc('\n',stdout); exit(EXIT_FAILURE);}while(0);
#define MIN(a,b) ((a)<(b)?(a):(b))
#define MAX(a,b) ((a)>(b)?(a):(b))

void ParamLoader::load_param(Layer* layer, char* data_file) {
    if(!data_file) die("%s is null", data_file);
    FILE *file = fopen(data_file, "rb");
	if (!file)  die("can't open %s", data_file);
	if(fseek(file, 0, SEEK_END)) die("can't fseek end on %s", data_file);
	int64_t file_size = ftell(file);
	if(file_size == -1LL) die("invalid file size");
    
    uint64_t header_len_u64 = 0;
    if(fseek(file, 0, SEEK_SET)) die("can't fseek start on %s", data_file);
	if(sizeof(header_len_u64) != (int64_t)fread(&header_len_u64, 1, sizeof(header_len_u64), file)) die("cant fread header_len");

    void *head_buffer = malloc(header_len_u64);
	if(!head_buffer) die("Can't malloc %lli bytes", (long long) header_len_u64);
    if(fseek(file, 8, SEEK_SET)) die("can't fseek start on %s", data_file);
    if(header_len_u64 != (int64_t)fread(head_buffer, 1, header_len_u64, file)) die("cant fread head_buffer");

    // 在这里读取文件头
    safetensors_File f = {0};
    char * result = safetensors_file_init(head_buffer, header_len_u64, &f);
    if(result) die("Error: load_state safetensors_file_init failed!");


    for(int i = 0; i < f.num_tensors; i++) {
        safetensors_TensorDescriptor t = f.tensors[i];
        uint64_t size = t.end_offset_bytes - t.begin_offset_bytes;
        std::string tensor_name(t.name.ptr, t.name.len);

        if(fseek(file, 8 + header_len_u64 + t.begin_offset_bytes, SEEK_SET)) die("can't fseek start on %s", data_file);
        void *temp_buffer = malloc(size);
	    if(!temp_buffer) die("Can't malloc %lli bytes", (long long) size);
        if(size != (int64_t)fread(temp_buffer, 1, size, file)) die("cant fread temp_buffer");

        // FIXME:现在只能够暂时将float16转换成float32
        // 且只能读到 cpu 中
        if(t.dtype == SAFETENSORS_F16) {
            size /= 2;
            std::shared_ptr<void> void_ptr(
                new float[size], // 创建大小为 size 的 float 数组
                [](void* ptr) {  // 自定义删除器
                    delete[] static_cast<float*>(ptr);  // 正确释放数组内存
                }
            );
            float* float_ptr = static_cast<float*>(void_ptr.get());
            // FIXME: 这里的device必是cpu
            for(int j = 0; j < size; j++) {
                uint16_t h = *(static_cast<uint16_t*>(temp_buffer) + j);
                float_ptr[j] = f16_to_f32(h);
            }
            state_map.emplace(tensor_name, void_ptr);
        } else if(t.dtype == SAFETENSORS_F32) {
            size /= 4;
            // FIXME: 这里的device必是cpu
            std::shared_ptr<void> void_ptr(
                new float[size], // 创建大小为 size 的 float 数组
                [](void* ptr) {  // 自定义删除器
                    delete[] static_cast<float*>(ptr);  // 正确释放数组内存
                }
            );
            float* float_ptr = static_cast<float*>(void_ptr.get());
            for(int j = 0; j < size; j++) {
                float_ptr[j] = *(static_cast<float*>(temp_buffer) + j);
            }
            state_map.emplace(tensor_name, void_ptr);
        } else if(t.dtype == SAFETENSORS_F64) {
            std::cerr << "Error: not supported dtype: SAFETENSORS_F64" << std::endl;
            exit(-1);
        } else if(t.dtype == SAFETENSORS_BF16) {    // FIXME: 这里能做大优化
            size /= 2;
            // FIXME: 这里的device必是cpu
            std::shared_ptr<void> void_ptr(
                new float[size], // 创建大小为 size 的 float 数组
                [](void* ptr) {  // 自定义删除器
                    delete[] static_cast<float*>(ptr);  // 正确释放数组内存
                }
            );
            float* float_ptr = static_cast<float*>(void_ptr.get());
            for(int j = 0; j < size; j++) {
                uint16_t h = *(static_cast<uint16_t*>(temp_buffer) + j);
                float_ptr[j] = bf16_to_f32(h);
            }
            state_map.emplace(tensor_name, void_ptr);
        }

        free(temp_buffer);
    }

    // 共享 embedding 和 decoding lm_head 权重
    if(tie_weights) {
        state_map.emplace("model.lm_head.weight", state_map.at("model.embed_tokens.weight"));
    }

    free(head_buffer);
    fclose(file);

    layer->load(state_map);

    return;
}
