/*
 * File: jsonconf.hpp
 * Project: impl
 * File Created: Saturday, 1st January 2022 7:06:54 pm
 * Author: Xu En (xuen@mokar.com)
 * -----
 * Last Modified: Saturday, 1st January 2022 7:06:57 pm
 * Modified By: Xu En (xuen@mokahr.com)
 * -----
 * Copyright 2021 - 2022 Your Company, Moka
 */
#ifndef SRC_IMPL_JSONCONF_HPP_
#define SRC_IMPL_JSONCONF_HPP_


#include <memory>
#include <string>

#include "../core/segment.hpp"
#include "../utils/file_utils.hpp"
#include "../utils/str_utils.hpp"

/**
 * @brief load the configuration
 *
 * @param jsonpath
 * @param segment point val pointer
 * @return int 1 error ,0 success
 */
int parseJsonConf(const char* json_conf_file, darts::Segment** segment) {
    std::string data;
    if (getFileText(json_conf_file, data)) {
        std::cerr << "ERROR: open conf data " << json_conf_file << " file failed " << std::endl;
        return EXIT_FAILURE;
    }

    Json::Value root;
    if (darts::getJsonRoot(data, root)) {
        std::cerr << "ERROR: parse json data " << json_conf_file << " file failed " << std::endl;
        return EXIT_FAILURE;
    }


    auto mem = root.getMemberNames();
    for (auto iter = mem.begin(); iter != mem.end(); iter++) {
        if (root[*iter].type() == Json::arrayValue) {
            auto cnt = root[*iter].size();
            std::string key(*iter);
            for (auto i = 0; i < cnt; i++) {
                std::string val = root[*iter][i].asString();
                _WordMap[val] = *iter;
            }
        } else {
            std::cerr << "ERROR: open json data " << dat << "key:" << *iter << "  failed " << std::endl;
            return EXIT_FAILURE;
        }
    }

    return EXIT_SUCCESS;
}

#endif  // SRC_IMPL_JSONCONF_HPP_
