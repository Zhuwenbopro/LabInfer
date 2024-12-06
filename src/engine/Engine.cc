#include "engine/Engine.h"

std::vector<std::string> expand_layer_line(const std::string& line) {
    std::vector<std::string> expanded_layers;
    std::istringstream iss(line);
    std::string token;

    while (iss >> token) {
        if (token == "layers") {
            // Read the next token
            std::string range_token;
            if (iss >> range_token) {
                // Check if range_token is a range
                std::regex range_regex(R"((\d+)-(\d+))");
                std::smatch match;
                if (std::regex_match(range_token, match, range_regex)) {
                    int start = std::stoi(match[1]);
                    int end = std::stoi(match[2]);
                    for (int i = start; i <= end; ++i) {
                        expanded_layers.push_back("layers " + std::to_string(i));
                    }
                } else if (std::regex_match(range_token, std::regex(R"(\d+)"))) {
                    // Single layer number
                    expanded_layers.push_back("layers " + range_token);
                } else {
                    // Unexpected format
                    expanded_layers.push_back("layers " + range_token);
                }
            } else {
                // 'layers' without following token
                expanded_layers.push_back("layers");
            }
        } else {
            // Other tokens (e.g., 'embed_tokens', 'norm', 'lm_head')
            expanded_layers.push_back(token);
        }
    }
    return expanded_layers;
}

// Function to parse the model file and return a vector of DeviceSection
std::vector<DeviceSection> parse_model_file(const std::string& filename) {
    std::vector<DeviceSection> device_sections;
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        throw std::runtime_error("Unable to open file " + filename);
    }
    std::string line;
    DeviceSection* current_section = nullptr;

    while (std::getline(infile, line)) {
        // Remove any leading/trailing whitespace
        line = std::regex_replace(line, std::regex("^\\s+|\\s+$"), "");

        if (line.empty()) continue;
        if (line.front() == '[' && line.back() == ']') {
            // New device section
            std::string current_device = line.substr(1, line.size() - 2);
            device_sections.push_back({current_device, {}});
            current_section = &device_sections.back();
        } else if (current_section != nullptr) {
            // Layer names
            auto expanded_layers = expand_layer_line(line);
            current_section->layers.insert(
                current_section->layers.end(),
                expanded_layers.begin(),
                expanded_layers.end()
            );
        } else {
            // Handle error: layers defined before any device section
            throw std::runtime_error("Layer defined before any device section in " + filename);
        }
    }
    return device_sections;
}

void Engine::add_request(const std::string& msg) {
    // std::vector<int> -> tensor
    if(queues.size() == 0)
        throw std::logic_error("no worker in working."); 
    queues[0]->push(msg);
    // queues[queues.size()-1]->push(msg);
}

void Engine::step() { 
    std::string msg = queues[queues.size()-1]->mergepop();
    std::cout << "pool push : " << msg << std::endl;
    queues[0]->push(msg);
}