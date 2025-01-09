#include "Cache.h"

/* RESULT
len = 10
max_len = 255
device = cpu
Caught expected exception: no 999 in cache

0.02    1.02    2.02    3.02    4.02    5.02    6.02    7.02    8.02    9.02
10.02   11.02   12.02   13.02   14.02   15.02   16.02   17.02   18.02   19.02
20.02   21.02   22.02   23.02   24.02   25.02   26.02   27.02   28.02   29.02
32.22   33.22   34.22   35.22   36.22   37.22   38.22   39.22   40.22   41.22
42.22   43.22   44.22   45.22   46.22   47.22   48.22   49.22   50.22   51.22
52.22   53.22   54.22   55.22   56.22   57.22   58.22   59.22   60.22   61.22

len = 10
max_len = 255
device = cuda
Caught expected exception: no 999 in cache
*/

int main() {
    Cache cache(Config("config.json"));
    cache.print();

    try {
        cache.get(999); // Attempt to access non-existent UID
    } catch (const std::logic_error& e) {
        std::cout << "Caught expected exception: " << e.what() << std::endl;
    }

    Tensor<float> tensor1(3, 10);
    Tensor<float> tensor2(3, 10);
    for(int i = 0; i < 30; i++) {
        tensor1[i] = i + 0.02;
        tensor2[i] = i + 32.22;
    }
    cache.add(999, tensor1, 0);
    cache.add(999, tensor2, 3);

    Parameter<float> param = cache.get(999);
    for(int i = 0; i < 6; i++) {
        for(int j = 0; j < 10; j++) {
            std::cout << param[i*10+j] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    cache.to("cuda");
    cache.print();

    cache.clear(999);

    try {
        cache.get(999); // Attempt to access non-existent UID
    } catch (const std::logic_error& e) {
        std::cout << "Caught expected exception: " << e.what() << std::endl;
    }
}