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


#include <map>
#include <memory>
#include <string>
#include <vector>

#include "../core/segment.hpp"
#include "../utils/file_utils.hpp"
#include "../utils/str_utils.hpp"


void getParams(Json::Value& node, std::map<std::string, std::string>& params) {
    auto mem = node.getMemberNames();
    for (auto iter = mem.begin(); iter != mem.end(); iter++) {
        params[*iter] = node[*iter].asString();
    }
}

/**
 * @brief load the configuration
 *
 * @param jsonpath
 * @param segment point val pointer
 * @return int 1 error ,0 success
 */
int parseJsonConf(const char* json_conf_file, darts::Segment** segment) {
    *segment = NULL;
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
    // check key
    if (!root.isMember("start.mode")) {
        std::cerr << "ERROR: no root key [start.mode] found!" << std::endl;
        return EXIT_FAILURE;
    }
    auto mode = root["start.mode"].asString();
    if (!root.isMember("modes")) {
        std::cerr << "ERROR: no root key [modes] found!" << std::endl;
        return EXIT_FAILURE;
    }
    auto modes_node = root["modes"];
    if (!root.isMember("recognizers")) {
        std::cerr << "ERROR: no root key [recognizers] found!" << std::endl;
        return EXIT_FAILURE;
    }
    auto recognizers_node = root["recognizers"];
    if (!root.isMember("persenters")) {
        std::cerr << "ERROR: no root key [persenters] found!" << std::endl;
        return EXIT_FAILURE;
    }
    auto persenters_node = root["persenters"];

    // path check
    if (!modes_node.isMember(mode)) {
        std::cerr << "ERROR: no key [modes/" << mode << "] found!" << std::endl;
        return EXIT_FAILURE;
    }
    auto use_node = modes_node[mode];
    if (!use_node.isMember("recognizers")) {
        std::cerr << "ERROR: no key [modes/" << mode << "/recognizers] found!" << std::endl;
        return EXIT_FAILURE;
    }
    if (!use_node.isMember("persenter")) {
        std::cerr << "ERROR: no key [modes/" << mode << "/persenter] found!" << std::endl;
        return EXIT_FAILURE;
    }
    auto used_persenter = use_node["persenter"].asString();
    auto used_recognizers_nodes = use_node["recognizers"];
    std::vector<std::string> used_recognizers;
    if (used_recognizers_nodes.type() == Json::arrayValue) {
        auto cnt = used_recognizers_nodes.size();
        for (auto i = 0; i < cnt; i++) {
            used_recognizers.push_back(used_recognizers_nodes[i].asString());
        }
    } else {
        std::cerr << "ERROR: read key [modes/" << mode << "/recognizers] failed,not string list!" << std::endl;
        return EXIT_FAILURE;
    }

    // load the persenters
    if (!persenters_node.isMember(used_persenter)) {
        std::cerr << "ERROR: no key [persenters/" << used_persenter << "] found!" << std::endl;
        return EXIT_FAILURE;
    }
    auto persenter_node = persenters_node[used_persenter];
    std::string persenter_type = persenter_node["type"].asString();
    std::map<std::string, std::string> params;
    getParams(persenter_node, params);
    darts::CellPersenter* persenter = darts::CellPersenterRegisterer::GetInstanceByName(persenter_type);
    if (!persenter) {
        std::cerr << "ERROR: create class [" << persenter_type << "] CellPersenter failed!" << std::endl;
        return EXIT_FAILURE;
    }
    if (persenter->initalize(params)) {
        std::cerr << "ERROR: init class [" << persenter_type << "] CellPersenter obj failed!" << std::endl;
        delete persenter;
        persenter = NULL;
        return EXIT_FAILURE;
    }

    // load used_recognizers_objs
    std::vector<darts::CellRecognizer*> used_recognizers_objs;
    for (auto name : used_recognizers) {
        // load the persenters and recognizers
        if (!recognizers_node.isMember(name)) {
            std::cerr << "ERROR: no key [recognizers/" << name << "] found!" << std::endl;
            for (auto r : used_recognizers_objs) delete r;
            used_recognizers_objs.clear();
            return EXIT_FAILURE;
        }
        auto recognizer_node = recognizers_node[name];
        std::string recognizer_type = recognizers_node["type"].asString();
        params.clear();
        getParams(recognizer_node, params);
        darts::CellRecognizer* recognizer = darts::CellRecognizerRegisterer::GetInstanceByName(recognizer_type);
        if (!recognizer) {
            std::cerr << "ERROR: create class [" << persenter_type << "] CellPersenter failed!" << std::endl;
            for (auto r : used_recognizers_objs) delete r;
            used_recognizers_objs.clear();
            return EXIT_FAILURE;
        }
        if (recognizer->initalize(params)) {
            std::cerr << "ERROR: init class [" << persenter_type << "] CellPersenter obj failed!" << std::endl;
            for (auto r : used_recognizers_objs) delete r;
            used_recognizers_objs.clear();
            delete recognizer;
            recognizer = NULL;
            return EXIT_FAILURE;
        }
        used_recognizers_objs.push_back(recognizer);
    }

    *segment = new darts::Segment(std::shared_ptr<darts::CellPersenter>(persenter));
    for (auto r : used_recognizers_objs) {
        (*segment)->addRecognizer(std::shared_ptr<darts::CellRecognizer>(r));
    }
    used_recognizers_objs.clear();
    return EXIT_SUCCESS;
}

#endif  // SRC_IMPL_JSONCONF_HPP_
