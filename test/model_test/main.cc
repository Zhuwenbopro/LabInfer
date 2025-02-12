#include "model/Model.h"
#include "../test.h"

int main() {
    Title("initialize Model");
    Model model("config.json", "xxx.model");


    while(true) {
        std::string cmd;
        std::cout << "可输入命令：\n";
        std::cin >> cmd;
        if(cmd == "quit") break;
        if(cmd == "run") model.run();
        if(cmd == "stop") model.stop();
        if(cmd == "add_request") {
            Tensor<int> input_ids(6, 1);
            input_ids[0] = 128000;  input_ids[1] = 791;     input_ids[2] = 1401; 
            input_ids[3] = 311;     input_ids[4] = 2324;    input_ids[5] = 374;
            InputWarp inputWarp(input_ids);
            model.add_request(inputWarp);
        }
    }
    // return -1;
}
