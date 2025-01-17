#include "model/Model.h"
#include <regex>
#define SAFETENSORS_IMPLEMENTATION
#include "safetensors.h"

#define die(...) do{printf(__VA_ARGS__); fputc('\n',stdout); exit(EXIT_FAILURE);}while(0);
#define MIN(a,b) ((a)<(b)?(a):(b))
#define MAX(a,b) ((a)>(b)?(a):(b))

// std::vector<std::string> expand_layer_line(const std::string& line) {
//     std::vector<std::string> expanded_layers;
//     std::istringstream iss(line);
//     std::string token;

//     while (iss >> token) {
//         if (token == "layers") {
//             // Read the next token
//             std::string range_token;
//             if (iss >> range_token) {
//                 // Check if range_token is a range
//                 std::regex range_regex(R"((\d+)-(\d+))");
//                 std::smatch match;
//                 if (std::regex_match(range_token, match, range_regex)) {
//                     int start = std::stoi(match[1]);
//                     int end = std::stoi(match[2]);
//                     for (int i = start; i <= end; ++i) {
//                         expanded_layers.push_back("layers " + std::to_string(i));
//                     }
//                 } else if (std::regex_match(range_token, std::regex(R"(\d+)"))) {
//                     // Single layer number
//                     expanded_layers.push_back("layers " + range_token);
//                 } else {
//                     // Unexpected format
//                     expanded_layers.push_back("layers " + range_token);
//                 }
//             } else {
//                 // 'layers' without following token
//                 expanded_layers.push_back("layers");
//             }
//         } else {
//             // Other tokens (e.g., 'embed_tokens', 'norm', 'lm_head')
//             expanded_layers.push_back(token);
//         }
//     }
//     return expanded_layers;
// }

// // Function to parse the model file and return a vector of DeviceSection
// std::vector<DeviceSection> parse_model_file(const std::string& filename) {
//     std::vector<DeviceSection> device_sections;
//     std::ifstream infile(filename);
//     if (!infile.is_open()) {
//         throw std::runtime_error("Unable to open file " + filename);
//     }
//     std::string line;
//     DeviceSection* current_section = nullptr;

//     while (std::getline(infile, line)) {
//         // Remove any leading/trailing whitespace
//         line = std::regex_replace(line, std::regex("^\\s+|\\s+$"), "");

//         if (line.empty()) continue;
//         if (line.front() == '[' && line.back() == ']') {
//             // New device section
//             std::string current_device = line.substr(1, line.size() - 2);
//             device_sections.push_back({current_device, {}});
//             current_section = &device_sections.back();
//         } else if (current_section != nullptr) {
//             // Layer names
//             auto expanded_layers = expand_layer_line(line);
//             current_section->layers.insert(
//                 current_section->layers.end(),
//                 expanded_layers.begin(),
//                 expanded_layers.end()
//             );
//         } else {
//             // Handle error: layers defined before any device section
//             throw std::runtime_error("Layer defined before any device section in " + filename);
//         }
//     }
//     return device_sections;
// }


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

Model::Model(const std::string& config_file, const std::string& model_file) {
    Config config("config.json");
    size_t hidden_size = config.get<size_t>("hidden_size");
    size_t vocab_size = config.get<size_t>("vocab_size");
    bool tie_weights = config.get<bool>("tie_word_embeddings");
    size_t num_hidden_layers = config.get<size_t>("num_hidden_layers");
    
    float epsilon = config.get<float>("rms_norm_eps");
    size_t middle_size = config.get<size_t>("intermediate_size");
    size_t head_dim = config.get<size_t>("head_dim");
    size_t attn_head = config.get<size_t>("num_attention_heads");
    size_t kv_head = config.get<size_t>("num_key_value_heads");
    // 下面这一段模型的构建，之后要根据输入 xxx.model 文件创建
    LayerList embed("model");
    embed.add_layer(new Embedding(vocab_size, hidden_size), "embed_tokens");
    LayerList decoders("model.layers");
    for(int i = 0; i < num_hidden_layers; i++) {
        decoders.add_layer(new DecoderLayer(attn_head, kv_head, hidden_size, middle_size, 250, epsilon), std::to_string(i));
    }
    LayerList backbone("model");
    backbone.add_layer(new RMSNorm(hidden_size, epsilon), "norm");
    backbone.add_layer(new Linear(hidden_size, vocab_size), "lm_head");
    backbone.add_layer(new Max(vocab_size), "max");
    
    printf("load weight\n");
    // 加载参数
    // std::unordered_map<std::string, std::shared_ptr<void>> state_map;
    // this->load_state("model.safetensors", state_map, tie_weights);
    // embed.load(state_map);
    // decoders.load(state_map);
    // backbone.load(state_map);
    
    // 读模型配置文件 xxx.model
    printf("queue\n");
    // std::vector<DeviceSection> device_sections = parse_model_file(model_file);
    device_sections.push_back(DeviceSection{std::string("cpu"), embed});
    device_sections.push_back(DeviceSection{std::string("cpu"), decoders});
    device_sections.push_back(DeviceSection{std::string("cpu"), backbone});

    // cpu : embed_tokens
    // cuda:0 : layers 0 layers 1 layers 2 layers 3 layers 4 layers 5 layers 6 layers 7
    // cuda:1 : layers 8 layers 9 layers 10 layers 11 layers 12 layers 13 layers 14 layers 15
    // cpu : norm lm_head
    // 建模型 靠着 DeviceSection.device = [DeviceSection.layer, ...]
    // 加载权重
    // 分配给 worker
    for(int i = 0; i <= device_sections.size(); i++)
        queues.push_back(std::make_shared<SafeQueue<InputWarp>>(std::to_string(i)));

    printf("workers\n");
    workers.push_back(std::make_unique<Worker>("embedding", queues[0], queues[1], embed));
    workers.push_back(std::make_unique<Worker>("decoders", queues[1], queues[2], decoders));
    workers.push_back(std::make_unique<Worker>("backbone", queues[2], queues[3], backbone));

}

void Model::load_state(char * filename, std::unordered_map<std::string, std::shared_ptr<void>>& state_map, bool tie_weights) {
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

    for(int i = 0; i < f.num_tensors; i++) {
        safetensors_TensorDescriptor t = f.tensors[i];
        uint64_t size = t.end_offset_bytes - t.begin_offset_bytes;
        std::string tensor_name(t.name.ptr, t.name.len);

        // std::cout << tensor_name << std::endl;

        if(fseek(file, 8 + header_len_u64 + t.begin_offset_bytes, SEEK_SET)) die("can't fseek start on %s", filename);
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

    return;
}

void Model::run() {
    for(int i = 0; i < workers.size(); i++) {
        workers[i].get()->run();
    }
}

void Model::stop() {
    for(int i = 0; i < workers.size(); i++) {
        workers[i].get()->stop();
    }
}

void Model::add_request(InputWarp& inputWarp) {
    if(queues.size() == 0)
        throw std::logic_error("no worker in working."); 
    queues[0]->push(inputWarp);
}

Model::~Model() {
    stop();
}
