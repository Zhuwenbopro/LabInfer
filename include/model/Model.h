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
    Model(const std::string& model_path);
    ~Model();

    std::vector<int> infer(std::vector<int>& input_ids, int max_len = 250);

    void to(const std::string &_device);
};


#endif