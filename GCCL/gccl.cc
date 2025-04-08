#include "gccl.h"

#include <future>
#include <thread>
#include <iostream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>

#include <future>

namespace gccl {

// -------------------------
// 同步通信接口实现（基于 TCP）
// ------------------------- 
int set_keep_alive(int socket_fd) {
    // 开启 TCP keepalive
    int opt = 1;
    if (setsockopt(socket_fd, SOL_SOCKET, SO_KEEPALIVE, &opt, sizeof(opt)) < 0) {
        perror("setsockopt(SO_KEEPALIVE) failed");
        return GCCL_ERROR_SOCKET; 
    }

    // 设置具体的 keepalive 参数（这些值可以根据实际情况调整）
    int keepIdle = 600;     // 空闲 600 秒后开始探测
    int keepInterval = 60;  // 探测间隔 60 秒
    int keepCount = 5;      // 探测次数 5 次

    if (setsockopt(socket_fd, IPPROTO_TCP, TCP_KEEPIDLE, &keepIdle, sizeof(keepIdle)) < 0) {
        perror("setsockopt(TCP_KEEPIDLE) failed");
        return GCCL_ERROR_SOCKET; 
    }
    if (setsockopt(socket_fd, IPPROTO_TCP, TCP_KEEPINTVL, &keepInterval, sizeof(keepInterval)) < 0) {
        perror("setsockopt(TCP_KEEPINTVL) failed");
        return GCCL_ERROR_SOCKET; 
    }
    if (setsockopt(socket_fd, IPPROTO_TCP, TCP_KEEPCNT, &keepCount, sizeof(keepCount)) < 0) {
        perror("setsockopt(TCP_KEEPCNT) failed");
        return GCCL_ERROR_SOCKET; 
    }
    return GCCL_SUCCESS;
}

int GCCL::set_listen_socket() {
    m_server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (m_server_fd < 0) { 
        return GCCL_ERROR_SOCKET; 
    }
    int opt = 1;
    setsockopt(m_server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in serverAddr;
    std::memset(&serverAddr, 0, sizeof(serverAddr));
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = INADDR_ANY; // 监听所有网卡
    serverAddr.sin_port = htons(BASE_PORT+m_rank); // 主节点监听端口
    if (bind(m_server_fd, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
        close(m_server_fd);
        return GCCL_ERROR_SOCKET;
    }

    if (listen(m_server_fd, 10) < 0) {
        close(m_server_fd);
        return GCCL_ERROR_SOCKET;
    }

    return GCCL_SUCCESS;
}

int GCCL::set_send_conn(int port, const char *ip, int &conn_fd) {
    if ((conn_fd = socket(AF_INET, SOCK_STREAM, 0)) < 0) { 
        return GCCL_ERROR_SOCKET; 
    }

    sockaddr_in masterAddr;
    std::memset(&masterAddr, 0, sizeof(masterAddr));
    masterAddr.sin_family = AF_INET;
    masterAddr.sin_port = htons(port);
    inet_pton(AF_INET, ip, &masterAddr.sin_addr);
    if (connect(conn_fd, (struct sockaddr*)&masterAddr, sizeof(masterAddr)) < 0) {
        close(conn_fd);
        return GCCL_ERROR_SOCKET;
    }

    return set_keep_alive(conn_fd);
}

int GCCL::set_recv_conn(int &conn_fd) {
    sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);
    conn_fd = accept(m_server_fd, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
    if (conn_fd < 0) {
        perror("accept error");
        return GCCL_ERROR_COMM_FAILURE;
    }

    if(set_keep_alive(conn_fd) != GCCL_SUCCESS) {
        return GCCL_ERROR_COMM_FAILURE;
    }
    return GCCL_SUCCESS;
}

int GCCL::send_message_header(int sock_fd, int tag, int length) {
    MessageHeader header;
    header.src = htonl(m_rank);         // 我准备发送
    header.tag = htonl(tag);            // tag 是
    header.length = htonl(length);      // 我接下来要发送的
    ssize_t bytesWritten = write_all(sock_fd, &header, sizeof(header));
    if (bytesWritten != sizeof(header)) {
        close(sock_fd);
        return GCCL_ERROR_COMM_FAILURE;
    }
    return GCCL_SUCCESS;
}

int GCCL::recv_message_header(int& recv_rank, int &tag, int &len, int sock_fd) {
    MessageHeader header;
    ssize_t bytesRead = read_all(sock_fd, &header, sizeof(header));
    if (bytesRead != sizeof(header)) {
        close(sock_fd);
        close(m_server_fd);
        return GCCL_ERROR_COMM_FAILURE;
    }

    recv_rank = ntohl(header.src);
    tag = ntohl(header.tag);
    len = ntohl(header.length);
    
    return GCCL_SUCCESS;
}


// -------------------------
// 内部辅助函数实现
// -------------------------
ssize_t GCCL::read_all(int sock, void* buf, size_t count) {
    size_t bytesRead = 0;
    char* buffer = static_cast<char*>(buf);
    while (bytesRead < count) {
        ssize_t ret = ::read(sock, buffer + bytesRead, count - bytesRead);
        if (ret <= 0) {
            return ret;
        }
        bytesRead += ret;
    }
    return bytesRead;
}

ssize_t GCCL::write_all(int sock, const void* buf, size_t count) {
    size_t bytesWritten = 0;
    const char* buffer = static_cast<const char*>(buf);
    while (bytesWritten < count) {
        ssize_t ret = ::write(sock, buffer + bytesWritten, count - bytesWritten);
        if (ret <= 0) {
            return ret;
        }
        bytesWritten += ret;
    }
    return bytesWritten;
}

// -------------------------
// 构造/析构与初始化管理
// -------------------------
GCCL::GCCL() : m_initialized(false), m_rank(0), m_world_size(1), m_server_fd(-1), useRDMA(false) {
    // 构造函数中不自动调用 init()，用户可自行调用 init() 以获得更大控制权
}

GCCL::~GCCL() {
    if (m_initialized) {
        finalize();
    }
}

int GCCL::init(int world_size, int rank, std::vector<int>& group) {
    if (m_initialized) {
        return GCCL_ERROR_ALREADY_INITIALIZED;
    }

    m_world_size = world_size;
    m_rank = rank;
    m_ranks_group = group;
    m_root_rank = m_ranks_group[0];

    if (useRDMA) {
        // TODO: 添加 RDMA 初始化代码
    } else { // 使用 TCP
        int ret = set_listen_socket();
        if(ret != GCCL_SUCCESS) { return ret; }
        
        if (m_rank == m_root_rank) { // 主节点 (Root Rank)
            for (int target_rank : m_ranks_group) {
                if (target_rank == m_root_rank) continue; // 跳过 root 自身

                m_conn_fd[target_rank] = -1;
                if(set_send_conn(BASE_PORT + target_rank, "127.0.0.1", m_conn_fd[target_rank]) != GCCL_SUCCESS) { 
                    return GCCL_ERROR_SET_CONN; 
                } 

                send_message_header(m_conn_fd[target_rank], 0, 0);
                int recv_rank, tag, len;
                recv_message_header(recv_rank, tag, len, m_conn_fd[target_rank]);
            }
        } else { // 子节点 (Non-Root Ranks)
            m_conn_fd[m_root_rank] = -1;
            if(set_recv_conn(m_conn_fd[m_root_rank]) != GCCL_SUCCESS) {
                return GCCL_ERROR_SET_CONN;
            }

            int recv_rank, tag, len;
            recv_message_header(recv_rank, tag, len, m_conn_fd[m_root_rank]);
            send_message_header(m_conn_fd[m_root_rank], 0, 0);
        }
    }

    m_initialized = true;
    return GCCL_SUCCESS;
}

int GCCL::finalize() {
    if (!m_initialized) {
        return GCCL_ERROR_NOT_INITIALIZED;
    }
    if (useRDMA) {
        // TODO: 添加 RDMA 资源释放代码
    } else {
        for (const auto& kv : m_conn_fd) {
            if (kv.second >= 0)
                close(kv.second);
        }
        if (m_server_fd >= 0) {
            close(m_server_fd);
            m_server_fd = -1;
        }
    }
    m_initialized = false;
    return GCCL_SUCCESS;
}

int GCCL::scatter_bytes(const void* sendbuf, int sendCount, void* recvbuf) {
    if (!m_initialized) return GCCL_ERROR_NOT_INITIALIZED;

    const char* src = reinterpret_cast<const char*>(sendbuf);
    char* dst = reinterpret_cast<char*>(recvbuf);

    if (m_rank == m_root_rank) { // Root Rank
        // Root copies data to itself
        std::vector<std::future<ssize_t>> futures;
        // Scatter to other ranks 
        for (int i = 0; i < m_ranks_group.size(); i++) {
            int target_rank = m_ranks_group[i];
            if (target_rank == m_root_rank) continue;
            
            futures.push_back(std::async(
                std::launch::async, 
                &GCCL::write_all, this, 
                m_conn_fd[target_rank], const_cast<char*>(src+i*sendCount), sendCount
            ));
        }
        std::memcpy(dst, src, sendCount);
        for (auto& future : futures) {
            if(future.get() != sendCount) 
                printf("Something wrong in barrier\n");
        }
    } else { // Non-Root Rank
        std::future<ssize_t> future = std::async(
            std::launch::async, 
            &GCCL::read_all, this, 
            m_conn_fd[m_root_rank], dst, sendCount
        );

        if(future.get() != sendCount) 
            printf("Something wrong in barrier\n");
    }
    return GCCL_SUCCESS;
}

int GCCL::gather_bytes(const void* sendbuf, int sendCount, void* recvbuf) {
    if (!m_initialized) return GCCL_ERROR_NOT_INITIALIZED;

    char* out = reinterpret_cast<char*>(recvbuf);
    if (m_rank == m_root_rank) { // 主节点 (Root Rank)

        std::vector<std::future<ssize_t>> futures;
        for (int i = 1; i < m_ranks_group.size(); ++i) { // 注意从 i=1 开始，跳过 root 自身
            int source_rank = m_ranks_group[i];
            futures.push_back(std::async(
                std::launch::async, 
                &GCCL::read_all, this, 
                m_conn_fd[source_rank], out + i* sendCount, sendCount
            ));
        }
        std::memcpy(out, sendbuf, sendCount);
        for (auto& future : futures) {
            if(future.get() != sendCount) 
                printf("Something wrong in barrier\n");
        }
    } else { // 子节点 (Non-Root Ranks)
        std::future<ssize_t> future = std::async(
            std::launch::async, 
            &GCCL::write_all, this, 
            m_conn_fd[m_root_rank], sendbuf, sendCount
        );

        if(future.get() != sendCount) printf("Something wrong in barrier\n");
    }
    return GCCL_SUCCESS;
}

int GCCL::broadcast_bytes(void* buffer, int count) {
    // root 负责发送，其他 rank 接收
    if (m_rank == m_root_rank) {
        std::vector<std::future<ssize_t>> futures;
        // broadcast to other ranks 
        for (int i = 0; i < m_ranks_group.size(); i++) {
            int target_rank = m_ranks_group[i];
            if (target_rank == m_root_rank) continue;
            
            futures.push_back(std::async(
                std::launch::async, 
                &GCCL::write_all, this, 
                m_conn_fd[target_rank], buffer, count
            ));
        }
        
        for (auto& future : futures) {
            if(future.get() != count) 
                printf("Something wrong in Broadcast\n");
        }
    } else {
        std::future<ssize_t> future = std::async(
            std::launch::async, 
            &GCCL::read_all, this, 
            m_conn_fd[m_root_rank], buffer, count
        );

        if(future.get() != count) 
            printf("Something wrong in Broadcast\n");
    }
    return GCCL_SUCCESS;
}

int GCCL::barrier() {
    if (!m_initialized) return GCCL_ERROR_NOT_INITIALIZED;

    if (m_rank == m_root_rank) { // Root Rank
        int dummy;
        {   // 1. Root receives "ping" from all non-root ranks
            std::vector<std::future<ssize_t>> futures;
            for (int target_rank : m_ranks_group) {
                if (target_rank == m_rank) continue;
                futures.push_back(std::async(
                    std::launch::async, 
                    &GCCL::read_all, this, 
                    m_conn_fd[target_rank], &dummy, sizeof(dummy)
                ));
            }

            for (auto& future : futures) {
                if(future.get() != sizeof(dummy)) 
                    printf("Something wrong in barrier\n");
            }
        }

        dummy+=1000;

        {   // 2. Root sends "pong" to all non-root ranks
            std::vector<std::future<ssize_t>> futures;
            for (int target_rank : m_ranks_group) {
                if (target_rank == m_rank) continue;
                futures.push_back(std::async(
                    std::launch::async, 
                    &GCCL::write_all, this, 
                    m_conn_fd[target_rank], &dummy, sizeof(dummy)
                ));
            }

            for (auto& future : futures) {
                if(future.get() != sizeof(dummy)) 
                    printf("Something wrong in barrier\n");
            }
        }
    } else { // Non-Root Rank
        int dummy = m_rank;
        { // 1. Non-root sends "ping" to root
            std::future<ssize_t> future = std::async(
                std::launch::async, 
                &GCCL::write_all, this, 
                m_conn_fd[m_root_rank], &dummy, sizeof(dummy)
            );
    
            if(future.get() != sizeof(dummy)) printf("Something wrong in barrier\n");
    
        }

        {   // 2. Non-root receives "pong" from root
            std::future<ssize_t> future = std::async(
                std::launch::async, 
                &GCCL::read_all, this, 
                m_conn_fd[m_root_rank], &dummy, sizeof(dummy)
            );
    
            if(future.get() != sizeof(dummy) && dummy != dummy+1000) 
                printf("Something wrong in barrier\n");
        }
    }
    return GCCL_SUCCESS;
}

} // namespace gccl
