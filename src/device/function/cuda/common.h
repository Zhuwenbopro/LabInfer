// common.h

#ifndef COMMON_H
#define COMMON_H

// 向上取整
inline int divUp(int a, int b) {
    return (a - 1) / b + 1;
}

// 定义线程块大小
const int num_threads_large = 1024; // 根据硬件规格调整
const int num_threads_small = 128;

#endif