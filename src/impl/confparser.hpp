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
#ifndef SRC_IMPL_CONFPARSER_HPP_
#define SRC_IMPL_CONFPARSER_HPP_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "../core/segment.hpp"
#include "../utils/file_utils.hpp"
#include "../utils/str_utils.hpp"
#include "./encoder.hpp"
#include "./quantizer.hpp"
#include "./recognizer.hpp"

int checkDep(Json::Value& root, Json::Value& node, std::string& fname,
             std::map<std::string, std::shared_ptr<darts::SegmentPlugin>>& cache,
             std::map<std::string, std::shared_ptr<darts::SegmentPlugin>>& ret);

inline void getParams(Json::Value& node, std::map<std::string, std::string>& params) {
    auto mem = node.getMemberNames();
    for (auto iter = mem.begin(); iter != mem.end(); iter++) {
        params[*iter] = node[*iter].asString();
    }
}

inline void getList(Json::Value& node, std::vector<std::string>& ret) {
    auto cnt = node.size();
    for (auto i = 0; i < cnt; i++) {
        ret.push_back(node[i].asString());
    }
}

inline std::shared_ptr<darts::SegmentPlugin> loadServices(
    Json::Value& root, Json::Value& dservices_node, std::string& name,
    std::map<std::string, std::shared_ptr<darts::SegmentPlugin>>& cache) {
    std::map<std::string, std::shared_ptr<darts::SegmentPlugin>>::iterator it = cache.find(name);
    if (it != cache.end()) {
        return it->second;
    }
    if (!dservices_node.isMember(name)) {
        std::cerr << "ERROR: no key [dservices/" << name << "] found!" << std::endl;
        return nullptr;
    }
    auto snode        = dservices_node[name];
    std::string stype = snode["type"].asString();
    std::map<std::string, std::string> params;
    getParams(snode, params);
    std::map<std::string, std::shared_ptr<darts::SegmentPlugin>> deps;
    if (checkDep(root, snode, name, cache, deps)) {
        deps.clear();
        return nullptr;
    }
    darts::SegmentPlugin* service = darts::SegmentPluginRegisterer::GetInstanceByName(stype);
    if (!service) {
        std::cerr << "ERROR: create class [" << stype << "] SegmentPlugin failed!" << std::endl;
        deps.clear();
        return nullptr;
    }
    if (service->initalize(params, deps)) {
        std::cerr << "ERROR: init class [" << stype << "] SegmentPlugin obj failed!" << std::endl;
        delete service;
        deps.clear();
        return nullptr;
    }
    std::shared_ptr<darts::SegmentPlugin> ret(service);
    cache[name] = ret;
    deps.clear();
    return ret;
}

inline std::shared_ptr<darts::CellPersenter> loadPersenter(
    Json::Value& root, Json::Value& persenters_node, std::string& used_persenter,
    std::map<std::string, std::shared_ptr<darts::SegmentPlugin>>& cache) {
    std::map<std::string, std::shared_ptr<darts::SegmentPlugin>>::iterator it = cache.find(used_persenter);
    if (it != cache.end()) {
        return std::dynamic_pointer_cast<darts::CellPersenter>(it->second);
    }
    if (!persenters_node.isMember(used_persenter)) {
        std::cerr << "ERROR: no key [persenters/" << used_persenter << "] found!" << std::endl;
        return nullptr;
    }
    auto persenter_node        = persenters_node[used_persenter];
    std::string persenter_type = persenter_node["type"].asString();
    std::map<std::string, std::string> params;
    getParams(persenter_node, params);
    std::map<std::string, std::shared_ptr<darts::SegmentPlugin>> deps;
    if (checkDep(root, persenter_node, used_persenter, cache, deps)) {
        deps.clear();
        return nullptr;
    }
    darts::CellPersenter* persenter = darts::CellPersenterRegisterer::GetInstanceByName(persenter_type);
    if (!persenter) {
        std::cerr << "ERROR: create class [" << persenter_type << "] CellPersenter failed!" << std::endl;
        deps.clear();
        return nullptr;
    }
    if (persenter->initalize(params, deps)) {
        std::cerr << "ERROR: init class [" << persenter_type << "] CellPersenter obj failed!" << std::endl;
        delete persenter;
        deps.clear();
        return nullptr;
    }
    std::shared_ptr<darts::CellPersenter> ret(persenter);
    cache[used_persenter] = std::dynamic_pointer_cast<darts::SegmentPlugin>(ret);
    deps.clear();
    return ret;
}
inline std::shared_ptr<darts::CellRecognizer> loadRecognizer(
    Json::Value& root, Json::Value& recognizers_node, std::string& name,
    std::map<std::string, std::shared_ptr<darts::SegmentPlugin>>& cache) {
    std::map<std::string, std::shared_ptr<darts::SegmentPlugin>>::iterator it = cache.find(name);
    if (it != cache.end()) {
        return std::dynamic_pointer_cast<darts::CellRecognizer>(it->second);
    }

    if (!recognizers_node.isMember(name)) {
        std::cerr << "ERROR: no key [recognizers/" << name << "] found!" << std::endl;
        return nullptr;
    }
    auto recognizer_node        = recognizers_node[name];
    std::string recognizer_type = recognizers_node["type"].asString();
    std::map<std::string, std::string> params;
    getParams(recognizer_node, params);
    std::map<std::string, std::shared_ptr<darts::SegmentPlugin>> deps;
    if (checkDep(root, recognizer_node, name, cache, deps)) {
        deps.clear();
        return nullptr;
    }
    darts::CellRecognizer* recognizer = darts::CellRecognizerRegisterer::GetInstanceByName(recognizer_type);
    if (!recognizer) {
        std::cerr << "ERROR: create class [" << recognizer << "] CellRecognizer failed!" << std::endl;
        deps.clear();
        return NULL;
    }
    if (recognizer->initalize(params, deps)) {
        std::cerr << "ERROR: init class [" << recognizer << "] CellRecognizer obj failed!" << std::endl;
        deps.clear();
        return NULL;
    }
    std::shared_ptr<darts::CellRecognizer> ret(recognizer);
    cache[name] = std::dynamic_pointer_cast<darts::SegmentPlugin>(ret);
    deps.clear();
    return ret;
}

inline int loadDep(Json::Value& root, std::string& depName,
                   std::map<std::string, std::shared_ptr<darts::SegmentPlugin>>& cache) {
    if (cache.find(depName) != cache.end()) {
        return EXIT_SUCCESS;
    }
    // set
    // search in dservices nodes
    if (root.isMember("dservices")) {
        auto dservices_node = root["dservices"];
        if (dservices_node.isMember(depName)) {
            auto ret = loadServices(root, dservices_node, depName, cache);
            if (!ret) {
                return EXIT_FAILURE;
            }
            return EXIT_SUCCESS;
        }
    }

    // search in persenters nodes
    auto persenters_node = root["persenters"];
    if (persenters_node.isMember(depName)) {
        auto ret = loadPersenter(root, persenters_node, depName, cache);
        if (!ret) {
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }
    // search in recognizers nodes
    auto recognizers_node = root["recognizers"];
    if (recognizers_node.isMember(depName)) {
        auto ret = loadRecognizer(root, recognizers_node, depName, cache);
        if (!ret) {
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }
    std::cerr << "ERROR: could not find any plugin name [" << depName << "] in conf file!" << std::endl;
    return EXIT_FAILURE;
}

inline int checkDep(Json::Value& root, Json::Value& node, std::string& fname,
                    std::map<std::string, std::shared_ptr<darts::SegmentPlugin>>& cache,
                    std::map<std::string, std::shared_ptr<darts::SegmentPlugin>>& ret) {
    if (!node.isMember("deps")) {
        // node has no deps
        return EXIT_SUCCESS;
    }
    // set fnode prevent circular dependencies
    cache[fname] = nullptr;
    // get mapcode
    std::map<std::string, std::string> params;
    getParams(node["deps"], params);
    std::map<std::string, std::string>::iterator it;
    // set ret
    for (it = params.begin(); it != params.end(); ++it) {
        if (loadDep(root, it->second, cache)) {
            ret.clear();
            return EXIT_FAILURE;
        }
        ret[it->first] = cache[it->second];
    }
    return EXIT_SUCCESS;
}

/**
 * @brief load the configuration
 *
 * @param jsonpath
 * @param segment point val pointer
 * @return int 1 error ,0 success
 */
inline int parseJsonConf(const char* json_conf_file, darts::Segment** segment, const char* start_mode = NULL) {
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
    // get start mode
    std::string mode;
    if (!start_mode) mode = start_mode;
    if (mode.empty()) {
        if (!root.isMember("default.mode")) {
            std::cerr << "ERROR: no root key [default.mode] found!" << std::endl;
            return EXIT_FAILURE;
        }
        mode = root["default.mode"].asString();
    }
    // get modes node
    if (!root.isMember("modes")) {
        std::cerr << "ERROR: no root key [modes] found!" << std::endl;
        return EXIT_FAILURE;
    }
    auto modes_node = root["modes"];
    // get recognizers node
    if (!root.isMember("recognizers")) {
        std::cerr << "ERROR: no root key [recognizers] found!" << std::endl;
        return EXIT_FAILURE;
    }
    auto recognizers_node = root["recognizers"];
    // get persenters node
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
    auto used_persenter         = use_node["persenter"].asString();
    auto used_recognizers_nodes = use_node["recognizers"];

    std::vector<std::string> used_recognizers;
    if (used_recognizers_nodes.type() == Json::arrayValue) {
        getList(used_recognizers_nodes, used_recognizers);
    } else {
        std::cerr << "ERROR: read key [modes/" << mode << "/recognizers] failed,not string list!" << std::endl;
        return EXIT_FAILURE;
    }

    std::map<std::string, std::shared_ptr<darts::SegmentPlugin>> _cache;

    // load the persenters
    std::shared_ptr<darts::CellPersenter> persenter = loadPersenter(root, persenters_node, used_persenter, _cache);
    if (!persenter) {
        _cache.clear();
        return EXIT_FAILURE;
    }
    // load used_recognizers_objs
    std::vector<std::shared_ptr<darts::CellRecognizer>> used_recognizers_objs;
    for (auto name : used_recognizers) {
        // load the persenters and recognizers
        std::shared_ptr<darts::CellRecognizer> recognizer = loadRecognizer(root, recognizers_node, name, _cache);
        if (!recognizer) {
            persenter = nullptr;
            used_recognizers_objs.clear();
            _cache.clear();
            return EXIT_FAILURE;
        }
        used_recognizers_objs.push_back(recognizer);
    }

    *segment = new darts::Segment(persenter);
    for (auto r : used_recognizers_objs) {
        (*segment)->addRecognizer(r);
    }
    used_recognizers_objs.clear();
    _cache.clear();
    return EXIT_SUCCESS;
}

#endif  // SRC_IMPL_CONFPARSER_HPP_
