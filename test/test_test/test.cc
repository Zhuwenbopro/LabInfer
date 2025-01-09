#include <memory>
#include <iostream>

int main() {
    float * ptr1 = new float[10];
    void * ptr2 = (void*)ptr1;

    std::shared_ptr<float> sptr(ptr1);
}