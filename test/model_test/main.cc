#include "model/Model.h"
#include "../test.h"

int main() {
    Title("initialize Model");
    Model model("./llama3_2");
    model.to("cuda");

    Tensor<int> input_ids(6, 1);
    input_ids[0] = 128000;  input_ids[1] = 791;     input_ids[2] = 1401; 
    input_ids[3] = 311;     input_ids[4] = 2324;    input_ids[5] = 374;

    Tensor<int> output_ids = model.infer(input_ids);
    std::cout <<  output_ids.Device();
    for(int i = 0; i < output_ids.Size(); i++) {
        std::cout << output_ids[i] << " ";
    }
    std::cout << "\n";

    // return -1;
}
