/*
 * File: norm_chr.hpp
 * Project: utils
 * 字符串规范化工具
 * File Created: Saturday, 25th December 2021 2:40:13 pm
 * Author: Xu En (xuen@mokar.com)
 * -----
 * Last Modified: Saturday, 25th December 2021 2:40:17 pm
 * Modified By: Xu En (xuen@mokahr.com)
 * -----
 * Copyright 2021 - 2021 Your Company, Moka
 */
#ifndef SRC_UTILS_NORM_CHR_HPP_
#define SRC_UTILS_NORM_CHR_HPP_

#include <stdint.h>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include "./file_utils.hpp"
#include "./str_utils.hpp"
#include "./utf8.hpp"

/**
 * @brief  字符串规范化使用的工具
 *
 */
static std::unordered_map<std::string, std::string> _WordMap;
/**
 * @brief add some special words map
 *
 */

inline void _addWordMap() {
    // basic chars
    for (uint32_t i = 0; i < 0x20 + 1; i++) {
        _WordMap[u32str(i)] = ' ';
    }
    _WordMap[u32str(0x7F)] = ' ';
    for (uint32_t i = 8198; i < 8208; i++) {
        _WordMap[u32str(i)] = ' ';
    }
    for (uint32_t i = 8232; i < 8240; i++) {
        _WordMap[u32str(i)] = ' ';
    }
    for (uint32_t i = 8287; i < 8304; i++) {
        _WordMap[u32str(i)] = ' ';
    }
    for (uint32_t i = 0xFE00; i < 0xFE0F + 1; i++) {
        _WordMap[u32str(i)] = ' ';
    }
    for (uint32_t i = 65281; i < 65374 + 1; i++) {
        _WordMap[u32str(i)] = u32str(i - 65248);
    }
    _WordMap[u32str(12288)] = u32str(32);

    // special map
    std::map<std::string, std::string> special = {
        {"“", "\""},  {"”", "\""},  {"、", ","},  {"〜", "~"},  {"～", "~"},  {"－", "-"},  {"–", "-"},
        {"\r", "\n"}, {"︳", "|"},  {"▎", "|"},   {"ⅰ", "i"},   {"丨", "|"},  {"│", "|"},   {"︱", "|"},
        {"｜", "|"},  {"／", "/"},  {"『", "《"}, {"《", "《"}, {"〖", "《"}, {"】", "》"}, {"〗", "》"},
        {"【", "《"}, {"》", "》"}, {"』", "》"}, {"「", "《"}, {"」", "》"}, {"❬", "《"},  {"❭", "》"},
        {"❮", "《"},  {"❯", "》"},  {"❰", "《"},  {"❱", "》"},  {"〘", "《"}, {"〙", "》"}, {"〚", "《"},
        {"〛", "》"}, {"〉", "》"}, {"《", "《"}, {"》", "》"}, {"「", "《"}, {"」", "》"}, {"『", "《"},
        {"』", "》"}, {"【", "《"}, {"】", "》"}, {"〔", "《"}, {"〕", "》"}, {"〖", "《"}, {"〗", "》"}};
    for (auto& kv : special) {
        _WordMap[kv.first] = kv.second;
    }

    // check wordmap and fix it
    std::map<std::string, std::string> _TMP;
    for (auto& kv : _WordMap) {
        if (kv.first != kv.second) {
            _TMP[kv.first] = kv.second;
        }
    }

    // '\n' don't replace,\t replace 4 space empty,notice!!!!!!!!!!
    _TMP.erase("\n");
    _TMP["\t"] = "    ";
    _TMP.erase(" ");

    _WordMap.clear();
    for (auto& kv : _TMP) {
        if (_TMP.find(kv.second) != _TMP.end()) {
            _WordMap[kv.first] = _TMP[kv.second];
        } else {
            _WordMap[kv.first] = kv.second;
        }
    }
}

/**
 * @brief 执行初始化
 *
 * @return int
 */
inline int initializeMap() {
    std::string dat = getResource("data/kernel/confuse.json");
    std::string data;
    if (getFileText(dat, data)) {
        std::cerr << "ERROR: open json data " << dat << " file failed " << std::endl;
        return EXIT_FAILURE;
    }
    Json::Value root;
    if (darts::getJsonRoot(data, root)) {
        std::cerr << "ERROR: parse json data " << dat << " file failed " << std::endl;
        return EXIT_FAILURE;
    }
    auto mem = root.getMemberNames();
    for (auto iter = mem.begin(); iter != mem.end(); iter++) {
        if (root[*iter].type() == Json::arrayValue) {
            auto cnt = root[*iter].size();
            std::string key(*iter);
            for (auto i = 0; i < cnt; i++) {
                std::string val = root[*iter][i].asString();
                _WordMap[val]   = *iter;
            }
        } else {
            std::cerr << "ERROR: open json data " << dat << "key:" << *iter << "  failed " << std::endl;
            return EXIT_FAILURE;
        }
    }

    _addWordMap();
    return EXIT_SUCCESS;
}

/**
 * @brief 对原始字符串进行字符归一化
 *
 * @param str
 * @return std::string
 */
inline std::string normalizeStr(const std::string& str) {
    std::stringstream output;
    utf8_iter ITER;
    utf8_initEx(&ITER, str.c_str(), str.length());
    std::string tmps;
    std::unordered_map<std::string, std::string>::const_iterator it;
    while (utf8_next(&ITER)) {
        tmps = utf8_getchar(&ITER);
        it   = _WordMap.find(tmps);
        if (it != _WordMap.end()) {
            output << it->second;
        } else {
            output << tmps;
        }
    }
    return output.str();
}

#endif  // SRC_UTILS_NORM_CHR_HPP_
