#include "model/Model.h"
#include "../test.h"

int main() {
    Title("initialize Model");
    Model model("./llama3_2");
    model.to("cuda");

    std::vector<int> input_ids(6, 0);
    input_ids[0] = 128000;  input_ids[1] = 791;     input_ids[2] = 1401; 
    input_ids[3] = 311;     input_ids[4] = 2324;    input_ids[5] = 374;

    std::vector<int> output_ids = model.infer(input_ids);

    for(int i = 0; i < output_ids.size(); i++) {
        std::cout << output_ids[i] << " ";
    }
    std::cout << "\n";

    // return -1;
}
