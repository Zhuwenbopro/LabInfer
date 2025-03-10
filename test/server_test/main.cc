// 编译命令示例：
// g++ -o server server.cpp -I "./third_party/" -lboost_system -lpthread

#include <iostream>
#include <memory>
#include <utility>
#include <boost/asio.hpp>
#include "nlohmann/json.hpp"
#include "model/Model.h"

using boost::asio::ip::tcp;
using json = nlohmann::json;

class session : public std::enable_shared_from_this<session> {
    std::shared_ptr<Model> model;
public:
    session(tcp::socket socket, std::shared_ptr<Model> model_ptr) : socket_(std::move(socket)), model(model_ptr) { }

    void start() {
        do_read();
    }

    private:
    void do_read() {
        auto self(shared_from_this());
        // 异步读取直到遇到换行符 "\n"
        boost::asio::async_read_until(socket_, buffer_, "\n",
            [this, self](boost::system::error_code ec, std::size_t length){
                if (!ec) {
                    std::istream is(&buffer_);
                    std::string line;
                    std::getline(is, line);
                    // 解析 JSON 数据
                    try {
                        json j = json::parse(line);
                        if (j.contains("input_ids") && j["input_ids"].is_array()) {
                            std::vector<int> input_ids = j["input_ids"].get<std::vector<int>>();
                            std::vector<int> output_ids = model->infer(input_ids);
                            json response;
                            response["output_ids"] = output_ids;
                            response["status"] = "ok";
                            std::string response_str = response.dump() + "\n";
                            do_write(response_str);
                        } else {
                            throw std::runtime_error("没有找到 'input_ids' 键或该键不是数组");
                        }
                    } catch (json::parse_error &e) {
                        json error;
                        error["error"] = "JSON parse error";
                        error["what"] = e.what();
                        std::string response_str = error.dump() + "\n";
                        do_write(response_str);
                    }
                }
            });
    }

    void do_write(const std::string& msg) {
        auto self(shared_from_this());
        boost::asio::async_write(socket_, boost::asio::buffer(msg),
            [this, self](boost::system::error_code ec, std::size_t /*length*/){
                if (!ec) {
                    do_read(); // 继续读取下一条消息
                }
            });
    }

    // socket_ 保存客户端连接的 socket 对象。
    tcp::socket socket_;
    // buffer_ 用于存储异步读取的数据。
    boost::asio::streambuf buffer_;
};

class server {
public:
    server(boost::asio::io_context& io_context, short port)
      : acceptor_(io_context, tcp::endpoint(tcp::v4(), port))
    {
        global_model = std::make_shared<Model>("./llama3_2");
        global_model->to("cuda");
        do_accept();
    }

private:
    void do_accept() {
        acceptor_.async_accept(
            [this](boost::system::error_code ec, tcp::socket socket){
                if (!ec) {
                    std::make_shared<session>(std::move(socket), global_model)->start();
                }

                do_accept();
            });
    }

    tcp::acceptor acceptor_;
    std::shared_ptr<Model> global_model;
};

int main() {
    try {
        boost::asio::io_context io_context;
        server s(io_context, 9000);  // 监听9000端口
        std::cout << "C++ Async Server is running on port 9000..." << std::endl;
        io_context.run();
    } catch (std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
    }
    return 0;
}
