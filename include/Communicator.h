#pragma once
#include "SafePrinter.h"

class Communicator
{
private:
    /* data */
public:
    Communicator(/* args */) {

    }

    ~Communicator() {
        
    }

    void init() {
        SafePrinter::print("Communicator initialized.");
    }
};