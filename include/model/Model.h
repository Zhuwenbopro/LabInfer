#ifndef MODEL_H
#define MODEL_H

#include "layers/layers.h"
#include "Config.h"
#include "InputWarp.h"
#include "ParamLoader.h"


class Model {
private:
    ParamLoader paramLoader;
    Layer* model;
    std::string device;

public:
    Model(const std::string& config_file, const std::string& model_file);
    ~Model();

    void infer(InputWarp& inputWarp, int max_len = 250);

    void to(const std::string &_device);
};


#endif