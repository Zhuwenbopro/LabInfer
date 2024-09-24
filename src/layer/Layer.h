// Layer.h

#ifndef LAYER_H
#define LAYER_H
#include "../device/DeviceFactory.h"
#include "../variable/Variable.h"
#include <string>
#include <vector>

class Layer {
public:
    virtual ~Layer() {}
    
    // 前向传播函数，接受输入矩阵，返回输出矩阵
    virtual std::vector<Tensor*> forward(const std::vector<Tensor*>& inputs) = 0;
    
protected:
    std::string name;       // 变量名称
    Device* dev;            // 设备
};

class LayerList : public Layer {
public:
    void addLayer(Layer* layer){ layerList.push_back(layer); }

    std::vector<Tensor*> forward(const std::vector<Tensor*>& inputs){
        std::vector<Tensor> outputs = inputs;

        for (const auto& elem : layerList) {
            outputs = elem->forward(outputs);
        }
        
        return outputs;
    }

private:
    std::vector<Layer*> layerList = {};
}

#endif // LAYER_H