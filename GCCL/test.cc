#include <iostream>
#include <cstring>
#include <cstdlib>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/wait.h>

// 定义消息头结构体
struct MessageHeader {
    int src;     // 发送者的 rank（网络字节序）
    int tag;     // 消息标记（网络字节序）
    int length;  // 数据长度（网络字节序）
};

int set_listen_socket(int port, int &socket_fd) {
    if ((socket_fd = socket(AF_INET, SOCK_STREAM, 0)) < 0) { 
        return -1; 
    }
    int opt = 1;
    setsockopt(socket_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in serverAddr;
    std::memset(&serverAddr, 0, sizeof(serverAddr));
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = INADDR_ANY; // 监听所有网卡
    serverAddr.sin_port = htons(port); // 主节点监听端口
    if (bind(socket_fd, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
        close(socket_fd);
        return -1;
    }

    if (listen(socket_fd, 10) < 0) {
        close(socket_fd);
        return -1;
    }

    return 1;
}

int set_send_socket(int port, int &socket_fd, const char *ip) {
    if ((socket_fd = socket(AF_INET, SOCK_STREAM, 0)) < 0) { 
        return -1; 
    }

    sockaddr_in masterAddr;
    std::memset(&masterAddr, 0, sizeof(masterAddr));
    masterAddr.sin_family = AF_INET;
    masterAddr.sin_port = htons(port); // 主节点端口
    inet_pton(AF_INET, ip, &masterAddr.sin_addr); // 假设主节点在同一台机器上
    if (connect(socket_fd, (struct sockaddr*)&masterAddr, sizeof(masterAddr)) < 0) {
        close(socket_fd);
        return -1;
    }

    return 1;
}


// 模拟你提供的 GCCL 类中读写全部数据的函数
namespace GCCL {
    ssize_t read_all(int sock, void* buf, size_t count) {
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

    ssize_t write_all(int sock, const void* buf, size_t count) {
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
}

// 测试函数：建立 TCP 连接，发送并接收 MessageHeader
void test_tcp_message_header() {
    const int port = 12345;
    pid_t pid = fork();
    if (pid < 0) {
        perror("fork error");
        exit(EXIT_FAILURE);
    } else if (pid == 0) {
        // 子进程作为客户端
        sleep(1); // 等待服务器启动
        
        int client_sock = socket(AF_INET, SOCK_STREAM, 0);
        if (client_sock < 0) {
            perror("client socket error");
            exit(EXIT_FAILURE);
        }
        sockaddr_in server_addr;
        memset(&server_addr, 0, sizeof(server_addr));
        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons(port);
        if (inet_pton(AF_INET, "127.0.0.1", &server_addr.sin_addr) <= 0) {
            perror("inet_pton error");
            exit(EXIT_FAILURE);
        }
        if (connect(client_sock, reinterpret_cast<sockaddr*>(&server_addr), sizeof(server_addr)) < 0) {
            perror("connect error");
            exit(EXIT_FAILURE);
        }
        // 构造并发送 MessageHeader
        MessageHeader header;
        header.src = htonl(1);      // 将发送者的 rank 转换为网络字节序
        header.tag = htonl(100);    // 消息标记
        header.length = htonl(0);   // 此处仅发送头部，不传送文件数据

        if (GCCL::write_all(client_sock, &header, sizeof(header)) != sizeof(header)) {
            perror("client write_all error");
        } else {
            std::cout << "客户端发送 MessageHeader 成功。" << std::endl;
        }
        close(client_sock);
        exit(EXIT_SUCCESS);
    } else {
        // 父进程作为服务器
        int server_sock;
        if(set_listen_socket(port, server_sock) < 0) {
            std::cout << "server error\n"; return ;
        }
       
        sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int conn_sock = accept(server_sock, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
        if (conn_sock < 0) {
            perror("accept error");
            exit(EXIT_FAILURE);
        }
        MessageHeader header;
        ssize_t ret = GCCL::read_all(conn_sock, &header, sizeof(header));
        if (ret != sizeof(header)) {
            perror("server read_all error");
        } else {
            // 转换回主机字节序后打印
            int src = ntohl(header.src);
            int tag = ntohl(header.tag);
            int length = ntohl(header.length);
            std::cout << "服务器接收到 MessageHeader: src = " << src
                      << ", tag = " << tag << ", length = " << length << std::endl;
        }
        close(conn_sock);
        close(server_sock);
        wait(nullptr); // 等待子进程结束
    }
}

int main() {
    test_tcp_message_header();
    return 0;
}
