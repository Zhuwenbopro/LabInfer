#include "DeviceManager.h"
#include <iostream>

int main() {

    DeviceManager& manager = DeviceManager::getInstance();
    std::cout << "start: deviceManager -> "<< &manager << std::endl;

    Device * dc = manager.getDevice("cpu");
    for(auto it : manager.getDevices()) {
        std::cout << "sss"<< std::endl;
    }
    return 0;
}

