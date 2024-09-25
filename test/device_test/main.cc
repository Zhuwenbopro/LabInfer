#include "DeviceManager.h"
#include <iostream>

int main() {

    DeviceManager& manager = DeviceManager::getInstance();
    for(auto it : manager.getDevices()) {
        std::cout << "sss"<< std::endl;
    }
    return 0;
}

