#include "Config.h"


int main() {
    Config config("config.json");

    try {
        int vocab_size = config.get("vocab_size").get<int>();
        int rope_theta = config.get("rope_theta").get<float>();
        std::string rope_type = config.get("rope_scaling.rope_type").get<std::string>();

        std::cout << "vocab_size: " << vocab_size << "\n";
        std::cout << "rope_theta: " << rope_theta << "\n";
        std::cout << "rope_type: " << rope_type << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }

    return 0;
}
