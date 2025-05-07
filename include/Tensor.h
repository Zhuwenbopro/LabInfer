#pragma once

class Tensor
{
public:
    Tensor(size_t size)
        : size_(size)
    {
        data_ = new float[size_];
    }

    ~Tensor()
    {
        delete[] data_;
    }

    float &operator[](size_t index)
    {
        return data_[index];
    }

    const float &operator[](size_t index) const
    {
        return data_[index];
    }

    size_t size() const
    {
        return size_;
    }
private:
    float *data_;
    size_t size_;
};