#include <chrono>
#include <cstring>
#include <random>
#include <iostream>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <stdexcept>
#include <cassert>
#include <unordered_map>
#include <iomanip> // 用于设置输出格式


// 随机数生成器
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

std::chrono::time_point<std::chrono::high_resolution_clock, std::chrono::nanoseconds> start = std::chrono::high_resolution_clock::now();
std::chrono::time_point<std::chrono::high_resolution_clock, std::chrono::nanoseconds> end = std::chrono::high_resolution_clock::now();
std::chrono::duration<double> duration = end - start;

#define RESET   "\033[0m"
#define RED     "\033[31m"      // Red
#define GREEN   "\033[32m"      // Green

inline void check_pass(const std::string&  message){
    std::cout << GREEN << message << RESET << std::endl;
}

inline void check_error(const std::string&  message){
    std::cout << RED << message << RESET << std::endl;
}

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}


#define CHECK_CUBLAS(call) { \
    cublasStatus_t status = call; \
    if(status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS Error: " << status << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUDNN(call)                                               \
{                                                                       \
    cudnnStatus_t status = (call);                                      \
    if (status != CUDNN_STATUS_SUCCESS) {                               \
        std::cerr << "cuDNN error in " << __FILE__ << ":" << __LINE__   \
                  << " - " << cudnnGetErrorString(status) << std::endl; \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
}

const int CONSOLE_WIDTH = 80;
// Title 函数定义
void Title(const std::string &title) {
    // 计算装饰线的长度（与控制台宽度相同）
    std::string decoration(CONSOLE_WIDTH, '=');
    // 计算标题左侧的填充空格数以实现居中
    int padding = (CONSOLE_WIDTH - title.length()) / 2;
    if (padding < 0) padding = 0; // 防止负数
    // 打印装饰线
    std::cout << decoration << std::endl;
    // 打印居中的标题
    std::cout << std::setw(padding) << "" << title << std::endl;
    // 打印装饰线
    std::cout << decoration << std::endl;
}

class Device {
public:
    std::string dev;
    float _duration;

    virtual float* malloc(size_t size, bool autofill = false) = 0;
    virtual void free(float* ptr) = 0;
    virtual void copy(float* dst, float* src, size_t size) = 0;
    virtual void time_tick() = 0;
    virtual void time_stop() = 0;

    virtual ~Device() {}

    double duration() {
        return _duration;
    }

    void rand(float* a, size_t size) {
        for(int i = 0; i < size; i++) {
            a[i] = dist(gen);
        }
    }
};

class CPU : public Device {
private:
    std::chrono::time_point<std::chrono::high_resolution_clock, std::chrono::nanoseconds> start;
    std::chrono::time_point<std::chrono::high_resolution_clock, std::chrono::nanoseconds> end;
public:
    CPU() {
        dev = "cpu";
    }

    float* malloc(size_t size, bool autofill = false) override {
        float* ret = new float[size];
        if(autofill) {
            rand(ret, size);
        }
        return ret;
    }

    void free(float* ptr) override {
        delete[] ptr;
    }

    void copy(float* dst, float* src, size_t size) override {
        std::memcpy(dst, src, size*sizeof(float));
    }

    void time_tick() override {
        start = std::chrono::high_resolution_clock::now();
    }

    void time_stop() override {
        end = std::chrono::high_resolution_clock::now();
        auto _d = end - start;
        _duration = _d.count();
    }
 
};

class CUDA : public Device {
    cudaEvent_t start_kernel, stop_kernel;
public:
    CUDA() {
        dev = "cuda";
        cudaEventCreate(&start_kernel);
        cudaEventCreate(&stop_kernel);
    }

    ~CUDA() {
        CHECK_CUDA(cudaEventDestroy(start_kernel));
        CHECK_CUDA(cudaEventDestroy(stop_kernel));
    }

    float* malloc(size_t size, bool autofill = false) override {
        float* ret;
        CHECK_CUDA(cudaMalloc(&ret, size * sizeof(float)));
        if(autofill) {
            float* temp = new float[size];
            rand(temp, size);
            CHECK_CUDA(cudaMemcpy(ret, temp, size * sizeof(float), cudaMemcpyHostToDevice));
            delete[] temp;
        }
        return ret;
    }

    void free(float* ptr) override {
        CHECK_CUDA(cudaFree(ptr));
    }

    void copy(float* dst, float* src, size_t size) override {
        CHECK_CUDA(cudaMemcpy(dst, src, size*sizeof(float), cudaMemcpyDeviceToDevice));
    }

    void time_tick() override {
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(start_kernel));
    }

    void time_stop() override {
        CHECK_CUDA(cudaEventRecord(stop_kernel));
        CHECK_CUDA(cudaEventSynchronize(stop_kernel));
        CHECK_CUDA(cudaEventElapsedTime(&_duration, start_kernel, stop_kernel));
    }
};

class Test {
private:
    std::vector<float*> cpu_list;
    std::vector<float*> cuda_list;
    std::unordered_map<std::string, Device*> devices;
    std::string _device = "cpu";
    Device* cur_dev;

public:
    Test() {
        devices["cpu"] = new CPU();
        devices["cuda"] = new CUDA();
        cur_dev = devices["cpu"];
    }

    ~Test() {
        while(!cpu_list.empty()) {
            float* removed = cpu_list.back();
            cpu_list.pop_back();
            devices["cpu"]->free(removed);
        }

        while(!cuda_list.empty()) {
            float* removed = cuda_list.back();
            cuda_list.pop_back();
            devices["cuda"]->free(removed);
        }
    }

    void setDevice(const std::string& device) {
        if(device == "cpu") {
            cur_dev = devices["cpu"];
        } else if(device == "cuda") {
            cur_dev = devices["cuda"];
        } else {
            throw std::logic_error("Do not have device : " + _device);
        }
        _device = device;
    }

    float* getArr(size_t size, bool autofill = false) {
        float* ret = cur_dev->malloc(size, autofill);
        if(_device == "cpu")  cpu_list.push_back(ret);
        if(_device == "cuda") cuda_list.push_back(ret);
        return ret;
    }

    inline float* get_from_cpu(float* __restrict__ src, size_t size) {
        assert(_device == "cuda");
        float* tmp = getArr(size);
        CHECK_CUDA(cudaMemcpy(tmp, src, size*sizeof(float), cudaMemcpyHostToDevice));
        return tmp;
    }

    inline void copy(float* __restrict__ dst, float* __restrict__ src, size_t size) {
        cur_dev->copy(dst, src, size);
    }

    void print(float* a, size_t col, size_t row = 1, const std::string& msg = "") {
        float *a1;
        size_t size = col*row;
        if(_device == "cpu") {
            a1 = a;
        } else if(_device == "cuda") {
            a1 = new float[size];
            CHECK_CUDA(cudaMemcpy(a1, a, size*sizeof(float), cudaMemcpyDeviceToHost));
        }

        std::cout << msg << std::endl;
        for(int i = 0; i < row; i++) {
            for(int j = 0; j < col; j++) {
                std::cout << a1[i*col+j] << " ";
            }
            std::cout << std::endl;
        }

        if(_device == "cuda") {
            delete[] a1;
        }
    }

    void check(float*a , float* b, size_t size, const std::string& msg, float epsilion = 5e-2) {
        float *a1, *a2;
        if(_device == "cpu") {
            a1 = a;
            a2 = b;
        } else if(_device == "cuda") {
            a1 = new float[size];
            a2 = new float[size];
            CHECK_CUDA(cudaMemcpy(a1, a, size*sizeof(float), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(a2, b, size*sizeof(float), cudaMemcpyDeviceToHost));
        }

        // 检查两个版本的计算结果是否一致
        bool results_match = true;
        for (int i = 0; i < size; ++i) {
            if (std::fabs(a1[i] - a2[i]) > 1e-3f) {
                results_match = false;
                std::cout << "different at " << i << " :  " << a1[i] << " vs " << a2[i] << std::endl;
                break;
            }
        }

        // 输出是否一致
        if (results_match) {
            check_pass("\n["+msg+"] The results are consistent.\n");
        } else {
            check_error("\n["+msg+"] The results are NOT consistent!!!\n");
        }

        if(_device == "cuda") {
            delete[] a1;
            delete[] a2;
        }
    }

    void start_timing() {
        cur_dev->time_tick();
    }

    void end_timing() {
        cur_dev->time_stop();
    }

    float duration() {
        return cur_dev->duration();
    }
};