#pragma once

class Parameter
{
public:
    Parameter(size_t size) : size_(size)
    {
        weights_ = new float[size_];
    }

    ~Parameter()
    {
        delete[] weights_;
    }

    float &operator[](size_t index)
    {
        return weights_[index];
    }

    const float &operator[](size_t index) const
    {
        return weights_[index];
    }

    size_t size() const
    {
        return size_;
    }
    
    
private:
    float *weights_;
    size_t size_;
};