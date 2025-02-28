#include <stdio.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/iterator/counting_iterator.h>

int main() {
    const int n = 10;  // 数组大小
    const int k = 3;   // 要找的前 k 大的数
    float arr[n] = {1, 5, 8, 7, 3, 6, 9, 2, 4, 10};
    int indices[n];

    // 分配设备内存
    float *d_arr;
    int *d_indices;
    cudaMalloc(&d_arr, n * sizeof(float));
    cudaMalloc(&d_indices, n * sizeof(int));

    // 将数据从主机复制到设备
    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, indices, n * sizeof(int), cudaMemcpyHostToDevice);

    // 创建索引数组 [0, 1, 2, ..., n-1]
    thrust::device_ptr<int> d_indices_ptr(d_indices);
    thrust::sequence(d_indices_ptr, d_indices_ptr + n);  // 用thrust::sequence填充索引数组

    // 将原始指针转换为thrust::device_ptr，并使用thrust::sort_by_key
    thrust::device_ptr<float> d_arr_ptr(d_arr);

    // 对数组进行降序排序，并且排序相应的索引
    thrust::sort_by_key(d_arr_ptr, d_arr_ptr + n, d_indices_ptr, thrust::greater<float>());

    // 创建std::vector来存储前 k 个元素和索引
    std::vector<float> h_top_k_values(k);
    std::vector<int> h_top_k_indices(k);

    // 将前 k 个元素和对应的索引从设备复制到主机
    cudaMemcpy(h_top_k_values.data(), d_arr, k * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_top_k_indices.data(), d_indices, k * sizeof(int), cudaMemcpyDeviceToHost);

    // 输出前 k 大的数及其对应的索引
    printf("Top %d largest numbers are: ", k);
    for (int i = 0; i < k; i++) {
        printf("%f ", h_top_k_values[i]);
    }
    printf("\n");

    printf("Their corresponding indices are: ");
    for (int i = 0; i < k; i++) {
        printf("%d ", h_top_k_indices[i]);
    }
    printf("\n");

    // 释放设备内存
    cudaFree(d_arr);
    cudaFree(d_indices);

    return 0;
}
