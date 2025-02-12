#ifndef MODEL_H
#define MODEL_H

#include "Config.h"
#include "Worker.h"
#include "SafeQueue.h"
#include "InputWarp.h"


struct DeviceSection {
    std::string device;
    Layer* layers;
};

// std::vector<DeviceSection> parse_model_file(const std::string& filename);

class Model {
private:
    std::vector<std::shared_ptr<SafeQueue<InputWarp>>> queues;
    std::vector<std::unique_ptr<Worker>> workers;
    std::vector<DeviceSection> device_sections;

    void load_state(char * filename, std::unordered_map<std::string, std::shared_ptr<void>>& state_map, bool tie_weights);

public:
    Model(const std::string& config_file, const std::string& model_file);
    ~Model();

    void run();

    void stop();

    void add_request(InputWarp& inputWarp);
};


#endif