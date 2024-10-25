#include "Layer.h"
#include <stdint.h>
#include <math.h>

#define SAFETENSORS_IMPLEMENTATION
#include "safetensors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define die(...) do{printf(__VA_ARGS__); fputc('\n',stdout); exit(EXIT_FAILURE);}while(0);
#define MIN(a,b) ((a)<(b)?(a):(b))
#define MAX(a,b) ((a)>(b)?(a):(b))

static float f16_to_f32(uint16_t h) {
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

static float bf16_to_f32(uint16_t bf16_value) {
    // 创建一个 32 位的无符号整数变量
    uint32_t f32_value = uint32_t(bf16_value) << 16; // 左移 16 位

    // 将 uint32_t 的位模式解释为 float
    float result;
    std::memcpy(&result, &f32_value, sizeof(result));

    return result;
}

void Layer::to(const std::string& new_dev) {
    if(new_dev == device) return;
    
    for (auto& [_name, param] : params) {
        param.to(new_dev);
    }
    
    F = std::ref(Manager::getInstance().getFunction(new_dev));
    device = new_dev;
    
    for (auto& [_name, ptr_layer] : layers) {
        ptr_layer->to(new_dev);
    }
}

void Layer::load_state(std::unordered_map<std::string, std::shared_ptr<float []>>& state_map) {
    remove_prefix_from_keys(state_map, name + ".");

    for (auto& [_name, param] : params) {
        if(param.Share()) continue;

        auto it = state_map.find(param.Name());
        if (it != state_map.end()) {
            param.setValue(it->second);
            state_map.erase(it);
        } else {
            std::cout << name << "  " << param.Name() << "  Key not found!!!! " << param.Share() << std::endl;
            exit(-1);
        }
    }

    for (auto& [name, ptr_layer] : layers) {
        ptr_layer->load_state(state_map);
    }
}

void Layer::load_state(char * filename) {
    if (!filename) {
        std::cerr << "Error: load_state filename is null!" << std::endl;
        exit(-1);
    }

    FILE *file = fopen(filename, "rb");
	if (!file)  die("can't open %s", filename);
	if(fseek(file, 0, SEEK_END)) die("can't fseek end on %s", filename);
	int64_t file_size = ftell(file);
	if(file_size == -1LL) die("invalid file size");

    uint64_t header_len_u64 = 0;
    if(fseek(file, 0, SEEK_SET)) die("can't fseek start on %s", filename);
	if(sizeof(header_len_u64) != (int64_t)fread(&header_len_u64, 1, sizeof(header_len_u64), file)) die("cant fread header_len");

    void *head_buffer = malloc(header_len_u64);
	if(!head_buffer) die("Can't malloc %lli bytes", (long long) header_len_u64);
    if(fseek(file, 8, SEEK_SET)) die("can't fseek start on %s", filename);
    if(header_len_u64 != (int64_t)fread(head_buffer, 1, header_len_u64, file)) die("cant fread head_buffer");

    // 在这里读取文件头
    safetensors_File f = {0};
    char * result = safetensors_file_init(head_buffer, header_len_u64, &f);
    if(result) {
		std::cerr << "Error: load_state safetensors_file_init failed!" << std::endl;
        exit(-1);
	}

    std::unordered_map<std::string, std::shared_ptr<float []>> state_map;
    Manager& manager = Manager::getInstance();

    for(int i = 0; i < f.num_tensors; i++) {
        safetensors_TensorDescriptor t = f.tensors[i];
        uint64_t size = t.end_offset_bytes - t.begin_offset_bytes;
        std::string tensor_name(t.name.ptr, t.name.len);

        if(fseek(file, 8 + header_len_u64 + t.begin_offset_bytes, SEEK_SET)) die("can't fseek start on %s", filename);
        void *temp_buffer = malloc(size);
	    if(!temp_buffer) die("Can't malloc %lli bytes", (long long) size);
        if(size != (int64_t)fread(temp_buffer, 1, size, file)) die("cant fread temp_buffer");
        //std::cout << t.dtype << "  " << tensor_name << "  " << size << std::endl;

        // FIXME:现在只能够暂时将float16转换成float32
        // 且只能读到 cpu 中
        if(t.dtype == SAFETENSORS_F16) {
            size /= 2;
            std::shared_ptr<float []> mem = manager.allocateShared(size, device);     // FIXME: 这里的device必是cpu
            for(int j = 0; j < size; j++) {
                uint16_t h = *(static_cast<uint16_t*>(temp_buffer) + j);
                mem[j] = f16_to_f32(h);
            }
            state_map.emplace(tensor_name, mem);
        } else if(t.dtype == SAFETENSORS_F32) {
            size /= 4;
            std::shared_ptr<float []> mem = manager.allocateShared(size, device);     // FIXME: 这里的device必是cpu
            for(int j = 0; j < size; j++) {
                mem[j] = *(static_cast<float*>(temp_buffer) + j);
            }
            state_map.emplace(tensor_name, mem);
        } else if(t.dtype == SAFETENSORS_F64) {
            std::cerr << "Error: not supported dtype: SAFETENSORS_F64" << std::endl;
            exit(-1);
        } else if(t.dtype == SAFETENSORS_BF16) {    // FIXME: 这里能做大优化
            size /= 2;
            std::shared_ptr<float []> mem = manager.allocateShared(size, device);     // FIXME: 这里的device必是cpu
            for(int j = 0; j < size; j++) {
                uint16_t h = *(static_cast<uint16_t*>(temp_buffer) + j);
                mem[j] = bf16_to_f32(h);
            }
            state_map.emplace(tensor_name, mem);
        }
        
        free(temp_buffer);
    }

    free(head_buffer);
    fclose(file);

    load_state(state_map);
    return;
}

void Layer::remove_prefix_from_keys(std::unordered_map<std::string, 
            std::shared_ptr<float []>>& state_map, const std::string& prefix) {
    std::unordered_map<std::string, std::shared_ptr<float []>> updated_map;
    // 遍历 state_map
    for (auto& pair : state_map) {
        std::string key = pair.first;
        // 检查是否以 prefix 开头
        if (key.rfind(prefix, 0) == 0) {  // rfind 返回0表示从首部开始匹配
            // 去掉 prefix 部分
            std::string new_key = key.substr(prefix.length());
            // 将新的 key 和对应的值插入到 updated_map
            updated_map[new_key] = pair.second;
        } else {
            // 如果不匹配 prefix，保留原来的键值对
            updated_map[key] = pair.second;
        }
    }
    // 用更新后的 map 替换原来的 map
    state_map = std::move(updated_map);
}