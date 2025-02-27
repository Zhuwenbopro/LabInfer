#ifndef MODEL_H
#define MODEL_H

#include "Config.h"
#include "Worker.h"
#include "SafeQueue.h"
#include "InputWarp.h"
#include "Sampler.h"
#include "ParamLoader.h"


struct DeviceSection {
    std::string device;
    Layer* layers;
};

// std::vector<DeviceSection> parse_model_file(const std::string& filename);

class Model {
private:
    ParamLoader paramLoader;
    Sampler sampler;
    
    std::vector<std::shared_ptr<SafeQueue<InputWarp>>> queues;
    std::vector<std::unique_ptr<Worker>> workers;
    std::vector<DeviceSection> device_sections;

public:
    Model(const std::string& config_file, const std::string& model_file);
    ~Model();

    void stop();

    void add_request(InputWarp& inputWarp);
};


#endif