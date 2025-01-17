#include "model/Model.h"
#include "../test.h"

int main() {
    Title("initialize Model");
    Model model("llama_3", "xxx.model");


    while(true) {
        std::string cmd;
        std::cin >> cmd;
        if(cmd == "quit") break;
        if(cmd == "run") model.run();
        if(cmd == "stop") model.stop();
    }
    // return -1;
    
}
