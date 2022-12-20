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
#include "../utils/filetool.hpp"
#include "../utils/strtool.hpp"
#include "./encoder.hpp"
#include "./neuroplug.hpp"
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

inline std::shared_ptr<darts::Decider> loaddecider(
    Json::Value& root, Json::Value& deciders_node, std::string& used_decider,
    std::map<std::string, std::shared_ptr<darts::SegmentPlugin>>& cache) {
    std::map<std::string, std::shared_ptr<darts::SegmentPlugin>>::iterator it = cache.find(used_decider);
    if (it != cache.end()) {
        return std::dynamic_pointer_cast<darts::Decider>(it->second);
    }
    if (!deciders_node.isMember(used_decider)) {
        std::cerr << "ERROR: no key [deciders/" << used_decider << "] found!" << std::endl;
        return nullptr;
    }
    auto decider_node        = deciders_node[used_decider];
    std::string decider_type = decider_node["type"].asString();
    std::map<std::string, std::string> params;
    getParams(decider_node, params);
    std::map<std::string, std::shared_ptr<darts::SegmentPlugin>> deps;
    if (checkDep(root, decider_node, used_decider, cache, deps)) {
        deps.clear();
        return nullptr;
    }
    darts::Decider* decider = darts::DeciderRegisterer::GetInstanceByName(decider_type);
    if (!decider) {
        std::cerr << "ERROR: create class [" << decider_type << "] Decider failed!" << std::endl;
        deps.clear();
        return nullptr;
    }
    if (decider->initalize(params, deps)) {
        std::cerr << "ERROR: init class [" << decider_type << "] Decider obj failed!" << std::endl;
        delete decider;
        deps.clear();
        return nullptr;
    }
    std::shared_ptr<darts::Decider> ret(decider);
    cache[used_decider] = std::dynamic_pointer_cast<darts::SegmentPlugin>(ret);
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
        return nullptr;
    }
    if (recognizer->initalize(params, deps)) {
        std::cerr << "ERROR: init class [" << recognizer << "] CellRecognizer obj failed!" << std::endl;
        deps.clear();
        return nullptr;
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

    // search in deciders nodes
    auto deciders_node = root["deciders"];
    if (deciders_node.isMember(depName)) {
        auto ret = loaddecider(root, deciders_node, depName, cache);
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
inline int loadSegment(const char* json_conf_file, darts::Segment** segment, const char* start_mode = nullptr,
                       bool devel_mode = false) {
    *segment = nullptr;
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
    if (start_mode) mode = start_mode;
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
    // get deciders node
    if (!root.isMember("deciders")) {
        std::cerr << "ERROR: no root key [deciders] found!" << std::endl;
        return EXIT_FAILURE;
    }
    auto deciders_node = root["deciders"];

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
    if (!use_node.isMember("decider")) {
        std::cerr << "ERROR: no key [modes/" << mode << "/decider] found!" << std::endl;
        return EXIT_FAILURE;
    }
    auto used_decider           = use_node["decider"].asString();
    auto used_recognizers_nodes = use_node["recognizers"];

    std::vector<std::string> used_recognizers;
    if (used_recognizers_nodes.type() == Json::arrayValue) {
        getList(used_recognizers_nodes, used_recognizers);
    } else {
        std::cerr << "ERROR: read key [modes/" << mode << "/recognizers] failed,not string list!" << std::endl;
        return EXIT_FAILURE;
    }

    std::map<std::string, std::shared_ptr<darts::SegmentPlugin>> _cache;

    // load the deciders
    std::shared_ptr<darts::Decider> decider = nullptr;
    if (!devel_mode) {
        decider = loaddecider(root, deciders_node, used_decider, _cache);
        if (!decider) {
            _cache.clear();
            return EXIT_FAILURE;
        }
    }
    // load used_recognizers_objs
    std::vector<std::shared_ptr<darts::CellRecognizer>> used_recognizers_objs;
    for (auto name : used_recognizers) {
        // load the deciders and recognizers
        std::shared_ptr<darts::CellRecognizer> recognizer = loadRecognizer(root, recognizers_node, name, _cache);
        if (!recognizer) {
            decider = nullptr;
            used_recognizers_objs.clear();
            _cache.clear();
            return EXIT_FAILURE;
        }
        used_recognizers_objs.push_back(recognizer);
    }
    *segment = new darts::Segment(decider);
    for (auto r : used_recognizers_objs) {
        (*segment)->addRecognizer(r);
    }
    used_recognizers_objs.clear();
    _cache.clear();
    return EXIT_SUCCESS;
}

#endif  // SRC_IMPL_CONFPARSER_HPP_
