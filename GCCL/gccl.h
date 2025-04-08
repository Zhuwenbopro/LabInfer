#ifndef GCCL_HPP
#define GCCL_HPP
#include <cstdlib>
#include <cstring>
#include <vector>
#include <map>
#include <mutex>
#include <unistd.h>

namespace gccl {

enum GCCLError {
    GCCL_SUCCESS = 0,
    GCCL_ERROR_ALREADY_INITIALIZED = -1,
    GCCL_ERROR_NOT_INITIALIZED = -2,
    GCCL_ERROR_INVALID_ARGUMENT = -3,
    GCCL_ERROR_SOCKET = -4,
    GCCL_ERROR_COMM_FAILURE = -5,
    GCCL_ERROR_MSG_TOO_LARGE = -6,
    GCCL_ERROR_SET_CONN = -7,
};

class GCCL {
public:
    // 构造函数（不自动初始化，可由用户控制调用 init()）
    GCCL();

    // 析构函数（自动调用 finalize()）
    ~GCCL();

    // 初始化通信环境，配置网络、硬件加速器和拓扑信息
    int init(int world_size, int rank, std::vector<int>& group);

    // 释放通信环境资源，关闭网络连接等
    int finalize();

    // 分发：root 节点将一大块数据拆分给多个节点（每节点一块）
    template <typename T>
    int scatter(const T* sendbuf, int sendCount, T* recvbuf) {
        if (!m_initialized) return GCCL_ERROR_NOT_INITIALIZED;
        return scatter_bytes(sendbuf, sendCount * sizeof(T), recvbuf);
    }  
    int scatter_bytes(const void* sendbuf, int sendCount, void* recvbuf);

    // 收集：各节点将数据发送到 root，root 将它们依次拼接
    template <typename T>
    int gather(const T* sendbuf, int sendCount, T* recvbuf) {
        if (!m_initialized) return GCCL_ERROR_NOT_INITIALIZED;
        return gather_bytes(sendbuf, sendCount * sizeof(T), recvbuf);
    }
    int gather_bytes(const void* sendbuf, int sendCount, void* recvbuf);

    // 屏障：等待所有节点到达此处再继续
    int barrier();

    // 广播：root 节点向所有节点发送同一份数据
    template <typename T>
    int broadcast(const T* buffer, int count) {
        if (!m_initialized) return GCCL_ERROR_NOT_INITIALIZED;
        return broadcast_bytes(buffer, count * sizeof(T));
    }
    int broadcast_bytes(void* buffer, int count);


private:
    // Internal state
    int m_world_size;
    int m_rank;
    int m_root_rank;
    std::vector<int> m_ranks_group;

    static const uint16_t BASE_PORT = 5000;     // TCP 基础端口（实际监听端口 = BASE_PORT + rank）
    int m_server_fd;
    int set_listen_socket();

    // 持久链接
    std::map<int, int> m_conn_fd;
    int set_send_conn(int port, const char *ip, int &socket_fd);
    int set_recv_conn(int &socket_fd);

    bool m_initialized;


    // 预留 RDMA 标志（当前实现默认使用 TCP）
    bool useRDMA;

    // 内部辅助函数：确保从 socket 读写指定字节数
    ssize_t read_all(int sock, void* buf, size_t count);
    ssize_t write_all(int sock, const void* buf, size_t count);

    // 内部消息头结构，固定大小，用于 TCP 通信
    struct MessageHeader {
        int src;     // 发送者的 rank（网络字节序）
        int tag;     // 消息标记（网络字节序）
        int length;  // 数据长度（网络字节序）
    };

    int send_message_header(int sock_fd, int tag, int length);
    int recv_message_header(int& recv_rank, int &tag, int &len, int sock_fd);
};

} // namespace gccl

#endif // GCCL_HPP
