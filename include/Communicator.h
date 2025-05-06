#ifndef COMMUNICATOR_H
#define COMMUNICATOR_H

#include <vector>

enum COMMError {
    COMM_SUCCESS = 0,
    COMM_ERROR_ALREADY_INITIALIZED = -1,
    COMM_ERROR_NOT_INITIALIZED = -2,
    COMM_ERROR_INVALID_ARGUMENT = -3,
    COMM_ERROR_SOCKET = -4,
    COMM_ERROR_COMM_FAILURE = -5,
    COMM_ERROR_MSG_TOO_LARGE = -6,
    COMM_ERROR_SET_CONN = -7,
};


class Communicator
{
public:
    // 构造函数和析构函数
    Communicator();
    ~Communicator();

    // 初始化通信环境，配置网络、硬件加速器和拓扑信息
    int init(int world_size, int rank, std::vector<int>& group);

    // 释放通信环境资源，关闭网络连接等
    int finalize();

    // 分发：root 节点将一大块数据拆分给多个节点（每节点一块）
    template <typename T>
    int scatter(const T* sendbuf, int sendCount, T* recvbuf) {
        if (!m_initialized) return COMM_ERROR_NOT_INITIALIZED;
        return scatter_bytes(sendbuf, sendCount * sizeof(T), recvbuf);
    }  
    virtual int scatter_bytes(const void* sendbuf, int sendCount, void* recvbuf) = 0;

    // 收集：各节点将数据发送到 root，root 将它们依次拼接
    template <typename T>
    int gather(const T* sendbuf, int sendCount, T* recvbuf) {
        if (!m_initialized) return COMM_ERROR_NOT_INITIALIZED;
        return gather_bytes(sendbuf, sendCount * sizeof(T), recvbuf);
    }
    virtual int gather_bytes(const void* sendbuf, int sendCount, void* recvbuf) = 0;

    // 屏障：等待所有节点到达此处再继续
    virtual int barrier() = 0;

    // 广播：root 节点向所有节点发送同一份数据
    template <typename T>
    int broadcast(const T* buffer, int count) {
        if (!m_initialized) return COMM_ERROR_NOT_INITIALIZED;
        return broadcast_bytes(buffer, count * sizeof(T));
    }
    virtual int broadcast_bytes(void* buffer, int count) = 0;

private:
    // 私有成员变量
    int world_size;                 // 总的节点数
    int rank;                       // 当前节点的 rank 编号
    int root_rank;                  // root 节点的 rank 编号
    bool m_initialized;             // 是否已初始化
    std::vector<int> ranks_group;   // 组内的 rank 编号

};

#endif // COMMUNICATOR_H