#ifndef ENGINE_H
#define ENGINE_H

#include "Worker.h"
#include "SafeQueue.h"
#include "layers/Layer.h"
#include "layers/LayerList.h"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <regex>

#include <stdexcept>


// Struct to hold device sections
struct DeviceSection {
    std::string device;
    std::vector<std::string> layers;
};

std::vector<DeviceSection> parse_model_file(const std::string& filename);


// TODO: 给它搞成单例模式
class Engine {
private:
    std::vector<std::shared_ptr<SafeQueue<std::string>>> queues;
    std::vector<std::unique_ptr<Worker>> workers;

public:
    void add_request(const std::string& msg);

    Engine(std::string model_file) {
        // 读模型配置文件 xxx.model
        std::vector<DeviceSection> device_sections = parse_model_file(model_file);
        // 建模型 靠着 DeviceSection.device = [DeviceSection.layer, ...]

        // 加载权重

        // 分配给 worker
        for(int i = 0; i <= device_sections.size(); i++)
            queues.push_back(std::make_shared<SafeQueue<std::string>>(std::to_string(i)));

        for(int i = 0; i < device_sections.size(); i++) {
            auto section = device_sections[i];
            if(i == 0)
                workers.push_back(std::make_unique<Worker>(section.device, queues[i], queues[i+1], FIRST));
            else if(i == device_sections.size()-1)
                workers.push_back(std::make_unique<Worker>(section.device, queues[i], queues[0], LAST));
            else
                workers.push_back(std::make_unique<Worker>(section.device, queues[i], queues[i+1]));
        }
    }

    ~Engine() {
    }
};

#endif