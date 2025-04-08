#include "gccl.h"
#include <vector>
#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>
#include <sstream>
#include <algorithm>
#include <numeric> // for std::accumulate
#include <string>

// 测试配置
const int NUM_PROCESSES = 8; // 总进程数
const int ELEMS_PER_PROC = 2; // 每个进程处理元素数
const int BASE_PORT = 5000; // 基础端口号

std::mutex print_mtx; // 保证输出顺序

void thread_safe_print(const std::string& msg) {
    std::lock_guard<std::mutex> lock(print_mtx);
    std::cout << msg << std::endl;
}

void test_worker(int rank, int world_size) {
    // 设置进程组（0-3为group0，4-7为group1）
    std::vector<int> group;
    if (rank < 4) {
        group = {0, 1, 2, 3};
    } else {
        group = {4, 5, 6, 7};
    }


    // 初始化通信库（使用不同端口避免冲突）
    gccl::GCCL comm;
    int ret = comm.init(world_size, rank, group);
    thread_safe_print("Rank " + std::to_string(rank) + " init returned: " + std::to_string(ret));
    if (ret != gccl::GCCL_SUCCESS) {
        std::cerr << "Rank " << rank << " init failed with error: " << ret << std::endl;
        return;
    }
    if(rank == 1 || rank == 6) sleep(1);
    ret = comm.barrier();
    if (ret != gccl::GCCL_SUCCESS) {
        std::cerr << "Rank " << rank << " init failed with error: " << ret << std::endl;
        return;
    }
    thread_safe_print("Rank " + std::to_string(rank) + " barrier succeed");

    std::vector<int> local_data(ELEMS_PER_PROC, 0);
    for(auto &d : local_data) {
        d = rank + 100;
    }
    std::vector<int> group_data(ELEMS_PER_PROC * 4, 0);
    comm.gather<int>(local_data.data(), ELEMS_PER_PROC, group_data.data());
    // 生成测试数据
    if(rank == 0 || rank == 4) {
        std::string str = " ";
        for(int d : group_data) {
            str += std::to_string(d) + " ";
        }
        thread_safe_print(str);
    }
    
    

    // // 阶段2：组内收集数据 (Group 0 gather)
    // if (rank < 4) {
    //     std::vector<int> group_result(group.size() * ELEMS_PER_PROC);
    //     ret = comm.gather(local_data.data(), ELEMS_PER_PROC, group_result.data()); // count 参数改为元素个数
    //     if (ret != gccl::GCCL_SUCCESS) {
    //         std::cerr << "Rank " << rank << " gather error: " << ret << std::endl;
    //     } else {
    //         // 打印组内结果 (只让 rank 0 打印，避免重复)
    //         if (rank == 0) {
    //             std::stringstream ss;
    //             ss << "Rank " << rank << " Group Gather Result: [";
    //             for (auto v : group_result) ss << v << " ";
    //             ss << "]";
    //             thread_safe_print(ss.str());
    //         }
    //     }

    //     // 阶段3：Group0 rank 0 发送数据给 Group1 的 rank 4
    //     if (rank == 0) {
    //         int dest_rank = 4;
    //         int sock_to_rank4 = -1;
    //         if (comm.m_rank_to_socket.count(dest_rank)) { // Root rank uses pre-established connection for send
    //             sock_to_rank4 = comm.m_rank_to_socket[dest_rank];
    //         } else {
    //             // For send to rank outside the group, need to establish new connection (for simplicity, but not efficient in real scenario)
    //             sock_to_rank4 = socket(AF_INET, SOCK_STREAM, 0);
    //             if (sock_to_rank4 < 0) {
    //                 std::cerr << "Rank " << rank << " socket creation error for send to rank " << dest_rank << std::endl;
    //                 return;
    //             }
    //             sockaddr_in destAddr;
    //             memset(&destAddr, 0, sizeof(destAddr));
    //             destAddr.sin_family = AF_INET;
    //             destAddr.sin_port = htons(BASE_PORT + dest_rank);
    //             inet_pton(AF_INET, "127.0.0.1", &destAddr.sin_addr);
    //             if (connect(sock_to_rank4, (struct sockaddr*)&destAddr, sizeof(destAddr)) < 0) {
    //                 close(sock_to_rank4);
    //                 std::cerr << "Rank " << rank << " connect error to rank " << dest_rank << std::endl;
    //                 return;
    //             }
    //         }

    //         ret = comm.send_bytes(sock_to_rank4, group_result.data(), group_result.size() * sizeof(int), dest_rank, 0); // count 参数改为字节数
    //         if (sock_to_rank4 != comm.m_rank_to_socket[dest_rank]) close(sock_to_rank4); // Close only if new socket is created
    //         if (ret != gccl::GCCL_SUCCESS) {
    //             std::cerr << "Rank " << rank << " send_bytes error to rank 4: " << ret << std::endl;
    //         }
    //     }
    // }

    // // 阶段3/4: Group1 rank 4 接收数据 from Group0 rank 0 and Scatter (modified to just receive and print)
    // if (rank >= 4) {
    //     if (rank == 4) {
    //         std::vector<int> recv_buf(4 * ELEMS_PER_PROC);
    //         int src_rank = 0;
    //         int sock_from_rank0 = -1;

    //         // Rank 4 needs to listen for connection from rank 0 for receiving send data
    //         int server_fd_rank4 = socket(AF_INET, SOCK_STREAM, 0);
    //         if (server_fd_rank4 < 0) {
    //             std::cerr << "Rank " << rank << " socket creation error for recv from rank " << src_rank << std::endl;
    //             return;
    //         }
    //         int opt = 1;
    //         setsockopt(server_fd_rank4, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    //         sockaddr_in serverAddr_rank4;
    //         memset(&serverAddr_rank4, 0, sizeof(serverAddr_rank4));
    //         serverAddr_rank4.sin_family = AF_INET;
    //         serverAddr_rank4.sin_addr.s_addr = INADDR_ANY;
    //         serverAddr_rank4.sin_port = htons(BASE_PORT + rank);
    //         if (bind(server_fd_rank4, (struct sockaddr*)&serverAddr_rank4, sizeof(serverAddr_rank4)) < 0) {
    //             close(server_fd_rank4);
    //             std::cerr << "Rank " << rank << " bind error for recv from rank " << src_rank << std::endl;
    //             return;
    //         }
    //         if (listen(server_fd_rank4, 1) < 0) {
    //             close(server_fd_rank4);
    //             std::cerr << "Rank " << rank << " listen error for recv from rank " << src_rank << std::endl;
    //             return;
    //         }

    //         sockaddr_in clientAddr;
    //         socklen_t addrLen = sizeof(clientAddr);
    //         sock_from_rank0 = accept(server_fd_rank4, (struct sockaddr*)&clientAddr, &addrLen);
    //         close(server_fd_rank4); // Close listening socket after accept
    //         if (sock_from_rank0 < 0) {
    //             std::cerr << "Rank " << rank << " accept error for recv from rank " << src_rank << std::endl;
    //             return;
    //         }


    //         ret = comm.recv_bytes(sock_from_rank0, recv_buf.data(), recv_buf.size() * sizeof(int), src_rank, 0); // count 参数改为字节数
    //         close(sock_from_rank0); // Close recv socket after receiving
    //         if (ret != gccl::GCCL_SUCCESS) {
    //             std::cerr << "Rank " << rank << " recv_bytes error from rank 0: " << ret << std::endl;
    //         } else {
    //             std::stringstream ss_recv;
    //             ss_recv << "Rank " << rank << " Received from Rank 0: [";
    //             for (auto v : recv_buf) ss_recv << v << " ";
    //             ss_recv << "]";
    //             thread_safe_print(ss_recv.str());
    //         }

    //         // Scatter is removed for now to simplify test and focus on point-to-point and gather.
    //         // If scatter is needed, it should be a collective operation within Group1, and recv_buf should be scattered.
    //         // For now, just print received buffer.
    //     }
    // }

    // comm.barrier(); // Global barrier for all ranks to sync before finalize
    // comm.finalize();
}

int main() {
    std::vector<std::thread> threads;

    // 创建并启动所有线程
    for (int rank = 0; rank < NUM_PROCESSES; ++rank) {
        threads.emplace_back([rank]() {
            test_worker(rank, NUM_PROCESSES);
        });
    }

    // 等待所有线程完成
    for (auto& t : threads) {
        t.join();
    }

    return 0;
}