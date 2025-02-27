#ifndef PARAMLOADER_H
#define PARAMLOADER_H

#include "layers/Layer.h"

class ParamLoader {
private:
    // TODO : data type
    bool tie_weights;
    std::unordered_map<std::string, std::shared_ptr<void>> state_map;

public:
    ParamLoader(bool tie_weights = true) : tie_weights(tie_weights) {}

    void load_param(Layer* layer, char* data_file);
};

#endif