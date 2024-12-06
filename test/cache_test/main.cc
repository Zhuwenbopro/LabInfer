#include "Cache.h"

int main() {
    std::vector<int> a{};
    Tensor tensor({{0,1,2,3,4,5},{0,1,2,3}}, "cpu");
    Cache cache(1, 255, "cpu");

    tensor.addPos({{0,1,2,3,4,5},{0,1,2,3}});
    tensor.setUid({112231,54523});
    cache.add(tensor);

    for (size_t uid : tensor.Uid()) {
        float* data = cache.get(uid);
        assert(data != nullptr);
        std::cout << "UID " << uid << " retrieved successfully. data = " << data[2] << std::endl;
    }

    std::cout << "Add and Get passed.\n" << std::endl;

    cache.to("cuda");
    cache.to("cpu");

    for (size_t uid : tensor.Uid()) {
        float* data = cache.get(uid);
        assert(data != nullptr); // Ensuring data is still accessible
        std::cout << "UID " << uid;
        std::cout << "  data = " << data[0] << std::endl;
    }

    try {
        cache.get(999); // Attempt to access non-existent UID
    } catch (const std::logic_error& e) {
        std::cout << "Caught expected exception: " << e.what() << std::endl;
    }

    std::cout << "Exception handling passed.\n" << std::endl;

}