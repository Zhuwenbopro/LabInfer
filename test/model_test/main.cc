#include "models/llama.h"


int main() {

    // check_lm_head();
    // return 1;

    Config config("config.json");
    Llama model(config);

    std::cout << "loading model..." << std::endl;
    model.load_state("./model.safetensors", true);
    std::cout << "loaded model" << std::endl;

    std::vector<std::vector<size_t>> input_ids = {{128000, 791, 1401, 311, 2324, 374}};
    std::vector<std::vector<size_t>> position = {{0, 1, 2, 3, 4, 5}};
    std::vector<size_t> uid = {112358};

    Tensor x(input_ids);
    x.addPos(position);
    x.setUid(uid);

    // x.to("cuda");
    // model.to("cuda");
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 10; i++) {
        //std::cout << i << std::endl;
        x = model.forward(x);
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << " executed in " << elapsed.count() << " ms.\n";

    // return -1;
    
}
