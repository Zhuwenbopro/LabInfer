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
    }
    // return -1;
}
