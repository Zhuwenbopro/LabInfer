#pragma once
#include "SafePrinter.h"

class Device
{
public:
    Device(int id) : id_(id) {}
    ~Device() {}

    void init() {
        SafePrinter::print("Device " , id_ , " initialized.");
    }

private:
    int id_;
};