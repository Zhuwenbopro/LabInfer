#include "model/Sampler.h"
#include "../test.h"

int main() {
    int vocab_size = 12800;

    Tensor<int> input_ids(6, 1);
    input_ids[0] = 0; input_ids[1] = 3324; input_ids[2] = 34; input_ids[3] = 731; input_ids[4] = 7734; input_ids[5] = 455;
    std::cout << "构建 inputWarp\n";
    InputWarp inputWarp(input_ids);
    inputWarp.inter_value = Tensor<float>(1, vocab_size);
 
    std::cout << "\n初始化 inputWarp\n";
    rand_init(inputWarp.inter_value, inputWarp.inter_value.Size());

    InputWarp inputWarp2 = inputWarp;
    inputWarp2.to("cuda");

    Sampler sampler(0.7, 2, 0.7);
    sampler.sample(inputWarp);
    std::cout << inputWarp.output_ids[0] << std::endl;

    sampler.to("cuda");
    sampler.sample(inputWarp2);
    std::cout << inputWarp2.output_ids[0] << std::endl;
}