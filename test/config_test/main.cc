#include "Config.h"


int main() {
    Config config("config.json");

    try {
        int vocab_size = config.get<int>("vocab_size");
        int hidden_size = config.get<int>("hidden_size");
        int rope_theta = config.get<float>("rope_theta");
        std::string rope_type = config.get<std::string>("rope_scaling.rope_type");
        std::vector<int> eos_token_id = config.get<std::vector<int>>("eos_token_id");

        std::cout << "vocab_size: " << vocab_size << std::endl;
        std::cout << "rope_theta: " << rope_theta << std::endl;
        std::cout << "rope_type: " << rope_type << std::endl;
        std::cout << "eos_token_id: [";
        for(int i = 0; i < eos_token_id.size(); i++)
            std::cout << " " << eos_token_id[i];
        std::cout << " ]" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
