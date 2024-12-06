#ifndef CONFIG_H
#define CONFIG_H

#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>

using Json = nlohmann::json;

class Config {
private:
    Json config;

    Config() = default;

public:
    Config(const std::string& _config_path) {
        std::ifstream inFile(_config_path);
        if (inFile.is_open()) {
            inFile >> config;
            inFile.close();
        } else {
            std::cerr << "Could not open file for reading\n";
            exit(-1);
        }
    }

    template<typename T>
    T get(const std::string& path) {
        std::istringstream ss(path);
        std::string token;
        
        // Split the string by '.' and navigate the JSON object
        Json current = config;
        while (std::getline(ss, token, '.')) {
            if (current.contains(token)) {
                current = current[token];  // Move to the next nested object
            } else {
                throw std::invalid_argument("Invalid path: " + path);
            }
        }

        return current.get<T>();
    }

};

#endif // CONFIG_H