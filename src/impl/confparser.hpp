/*
 * File: jsonconf.hpp
 * Project: impl
 * File Created: Saturday, 1st January 2022 7:06:54 pm
 * Author: Xu En (xuen@mokar.com)
 * -----
 * Last Modified: Saturday, 1st January 2022 7:06:57 pm
 * Modified By: Xu En (nanhangxuen@163.com)
 * -----
 * Copyright 2021 - 2022 Your Company, Moka
 */
#ifndef SRC_IMPL_CONFPARSER_HPP_
#define SRC_IMPL_CONFPARSER_HPP_

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "../core/segment.hpp"
#include "../utils/filetool.hpp"
#include "../utils/strtool.hpp"
#include "./encoder.hpp"
#include "./neuroplug.hpp"
#include "./quantizer.hpp"
#include "./recognizer.hpp"

using PluginCache = std::unordered_map<std::string, std::shared_ptr<darts::SegmentPlugin>>;

inline void getParams(const Json::Value& node, std::map<std::string, std::string>& params) {
    for (const auto& name : node.getMemberNames()) {
        if (node[name].isString()) params.emplace(name, node[name].asString());
    }
}

inline void getList(const Json::Value& node, std::vector<std::string>& ret) {
    if (!node.isArray()) return;
    ret.reserve(ret.size() + node.size());
    for (const auto& value : node)
        if (value.isString()) ret.push_back(value.asString());
}

inline std::shared_ptr<darts::SegmentPlugin> loadPlugin(const Json::Value& root,
                                                        const std::string& name,
                                                        PluginCache& cache);

inline const char* findPluginSection(const Json::Value& root, const std::string& name) {
    static const char* sections[] = {"dservices", "deciders", "recognizers"};
    for (const char* section : sections) {
        if (root.isMember(section) && root[section].isMember(name)) return section;
    }
    return nullptr;
}

inline darts::SegmentPlugin* createPlugin(const std::string& section, const std::string& type) {
    if (section == "dservices") return darts::SegmentPluginRegisterer::GetInstanceByName(type);
    if (section == "deciders") return darts::DeciderRegisterer::GetInstanceByName(type);
    if (section == "recognizers") return darts::CellRecognizerRegisterer::GetInstanceByName(type);
    return nullptr;
}

inline int loadDependencies(const Json::Value& root, const Json::Value& node, PluginCache& cache,
                            std::map<std::string, std::shared_ptr<darts::SegmentPlugin>>& deps) {
    if (!node.isMember("deps")) return EXIT_SUCCESS;
    const auto& dep_node = node["deps"];
    if (!dep_node.isObject()) {
        std::cerr << "ERROR: plugin deps must be an object" << std::endl;
        return EXIT_FAILURE;
    }
    for (const auto& alias : dep_node.getMemberNames()) {
        if (!dep_node[alias].isString()) return EXIT_FAILURE;
        auto dependency = loadPlugin(root, dep_node[alias].asString(), cache);
        if (!dependency) return EXIT_FAILURE;
        deps.emplace(alias, std::move(dependency));
    }
    return EXIT_SUCCESS;
}

inline std::shared_ptr<darts::SegmentPlugin> loadPlugin(const Json::Value& root,
                                                        const std::string& name,
                                                        PluginCache& cache) {
    auto cached = cache.find(name);
    if (cached != cache.end()) {
        if (!cached->second) std::cerr << "ERROR: circular plugin dependency at [" << name << "]" << std::endl;
        return cached->second;
    }
    const char* section = findPluginSection(root, name);
    if (!section) {
        std::cerr << "ERROR: could not find plugin [" << name << "]" << std::endl;
        return nullptr;
    }
    const auto& node = root[section][name];
    if (!node.isObject() || !node["type"].isString() || node["type"].asString().empty()) {
        std::cerr << "ERROR: invalid plugin type at [" << section << "/" << name << "]" << std::endl;
        return nullptr;
    }
    cache.emplace(name, nullptr);
    std::map<std::string, std::string> params;
    getParams(node, params);
    std::map<std::string, std::shared_ptr<darts::SegmentPlugin>> deps;
    if (loadDependencies(root, node, cache, deps)) {
        cache.erase(name);
        return nullptr;
    }
    std::unique_ptr<darts::SegmentPlugin> plugin(createPlugin(section, node["type"].asString()));
    if (!plugin || plugin->initalize(params, deps)) {
        std::cerr << "ERROR: create or initialize plugin [" << name << "] failed" << std::endl;
        cache.erase(name);
        return nullptr;
    }
    auto result = std::shared_ptr<darts::SegmentPlugin>(plugin.release());
    cache[name] = result;
    return result;
}

inline std::shared_ptr<darts::Decider> loaddecider(const Json::Value& root, const std::string& name,
                                                   PluginCache& cache) {
    return std::dynamic_pointer_cast<darts::Decider>(loadPlugin(root, name, cache));
}

inline std::shared_ptr<darts::CellRecognizer> loadRecognizer(const Json::Value& root,
                                                             const std::string& name,
                                                             PluginCache& cache) {
    return std::dynamic_pointer_cast<darts::CellRecognizer>(loadPlugin(root, name, cache));
}

/**
 * @brief load the configuration
 *
 * @param jsonpath
 * @param segment point val pointer
 * @return int 1 error ,0 success
 */
inline int loadSegment(const char* conffile, darts::Segment** segment, const char* start_mode = nullptr,
                       bool devel_mode = false) {
    if (!conffile || !segment) return EXIT_FAILURE;
    std::string json_conf_file = getResource(conffile);

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
    const auto& modes_node = root["modes"];
    // get recognizers node
    if (!root.isMember("recognizers")) {
        std::cerr << "ERROR: no root key [recognizers] found!" << std::endl;
        return EXIT_FAILURE;
    }
    // get deciders node
    if (!root.isMember("deciders")) {
        std::cerr << "ERROR: no root key [deciders] found!" << std::endl;
        return EXIT_FAILURE;
    }

    // path check
    if (!modes_node.isMember(mode)) {
        std::cerr << "ERROR: no key [modes/" << mode << "] found!" << std::endl;
        return EXIT_FAILURE;
    }
    const auto& use_node = modes_node[mode];
    if (!use_node.isMember("recognizers")) {
        std::cerr << "ERROR: no key [modes/" << mode << "/recognizers] found!" << std::endl;
        return EXIT_FAILURE;
    }
    if (!use_node.isMember("decider")) {
        std::cerr << "ERROR: no key [modes/" << mode << "/decider] found!" << std::endl;
        return EXIT_FAILURE;
    }
    const auto used_decider = use_node["decider"].asString();
    const auto& used_recognizers_nodes = use_node["recognizers"];

    std::vector<std::string> used_recognizers;
    if (used_recognizers_nodes.type() == Json::arrayValue) {
        getList(used_recognizers_nodes, used_recognizers);
    } else {
        std::cerr << "ERROR: read key [modes/" << mode << "/recognizers] failed,not string list!" << std::endl;
        return EXIT_FAILURE;
    }

    // The cache starts empty for every Segment. Only the selected mode's
    // decider/recognizers and dependencies reached recursively from them are
    // instantiated; merely declaring a plugin in JSON never creates it.
    PluginCache _cache;

    // load the deciders
    std::shared_ptr<darts::Decider> decider = nullptr;
    if (!devel_mode) {
        decider = loaddecider(root, used_decider, _cache);
        if (!decider) {
            _cache.clear();
            return EXIT_FAILURE;
        }
    }
    // load used_recognizers_objs
    std::vector<std::shared_ptr<darts::CellRecognizer>> used_recognizers_objs;
    for (const auto& name : used_recognizers) {
        // load the deciders and recognizers
        std::shared_ptr<darts::CellRecognizer> recognizer = loadRecognizer(root, name, _cache);
        if (!recognizer) {
            decider = nullptr;
            used_recognizers_objs.clear();
            _cache.clear();
            return EXIT_FAILURE;
        }
        used_recognizers_objs.push_back(recognizer);
    }
    *segment = new darts::Segment(decider);
    for (const auto& r : used_recognizers_objs) {
        (*segment)->addRecognizer(r);
    }
    used_recognizers_objs.clear();
    _cache.clear();
    return EXIT_SUCCESS;
}

#endif  // SRC_IMPL_CONFPARSER_HPP_
