#include "engine/Engine.h"

int main() {
    std::string model_name = "xxx.model";
    
    Engine engine(model_name);

    std::string msg;

    while(true) {
        std::cout << "请输入request：" << std::endl;
        std::cin >> msg;
        engine.add_request(msg);
        // engine.add_request(std::to_string(i++));
        // engine.step();
        // std::this_thread::sleep_for(std::chrono::milliseconds(300));  
    }

    return 0;
}