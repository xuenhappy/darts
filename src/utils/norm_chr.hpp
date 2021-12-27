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

#include <json/json.h>
#include <stdint.h>

#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include "file_utils.hpp"
#include "utf8.hpp"

/**
 * @brief  字符串规范化使用的工具
 *
 */
std::map<std::string, std::string> _WordMap;
/**
 * @brief add some special words map
 *
 */


void _addWordMap() {
    // basic chars
    for (uint32_t i = 0; i < 0x20 + 1; i++) {
        _WordMap[codeStr(i)] = ' ';
    }
    _WordMap[codeStr(0x7F)] = ' ';
    for (uint32_t i = 8198; i < 8208; i++) {
        _WordMap[codeStr(i)] = ' ';
    }
    for (uint32_t i = 8232; i < 8240; i++) {
        _WordMap[codeStr(i)] = ' ';
    }
    for (uint32_t i = 8287; i < 8304; i++) {
        _WordMap[codeStr(i)] = ' ';
    }
    for (uint32_t i = 0xFE00; i < 0xFE0F + 1; i++) {
        _WordMap[codeStr(i)] = ' ';
    }
    for (uint32_t i = 65281; i < 65374 + 1; i++) {
        _WordMap[codeStr(i)] = codeStr(i - 65248);
    }
    _WordMap[codeStr(12288)] = codeStr(32);


    // special map
    std::map<std::string, std::string> special = {
        {"“", "\""},  {"”", "\""},  {"、", ","},  {"〜", "~"},  {"～", "~"},  {"－", "-"},  {"–", "-"},
        {"\r", "\n"}, {"︳", "|"},  {"▎", "|"},   {"ⅰ", "i"},   {"丨", "|"},  {"│", "|"},   {"︱", "|"},
        {"｜", "|"},  {"／", "/"},  {"『", "《"}, {"《", "《"}, {"〖", "《"}, {"】", "》"}, {"〗", "》"},
        {"【", "《"}, {"》", "》"}, {"』", "》"}, {"「", "《"}, {"」", "》"}, {"❬", "《"},  {"❭", "》"},
        {"❮", "《"},  {"❯", "》"},  {"❰", "《"},  {"❱", "》"},  {"〘", "《"}, {"〙", "》"}, {"〚", "《"},
        {"〛", "》"}, {"〉", "》"}, {"《", "《"}, {"》", "》"}, {"「", "《"}, {"」", "》"}, {"『", "《"},
        {"』", "》"}, {"【", "《"}, {"】", "》"}, {"〔", "《"}, {"〕", "》"}, {"〖", "《"}, {"〗", "》"}};
    for (auto kv : special) {
        _WordMap[kv.first] = kv.second;
    }


    // check wordmap and fix it
    std::map<std::string, std::string> _TMP;
    for (auto kv : _WordMap) {
        if (kv.first != kv.second) {
            _TMP[kv.first] = kv.second;
        }
    }


    // '\n' don't replace,\t replace 4 space empty,notice!!!!!!!!!!
    _TMP.erase("\n");
    _TMP["\t"] = "    ";
    _TMP.erase(" ");

    _WordMap.clear();
    for (auto kv : _TMP) {
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
int initializeMap() {
    std::string dat = getResource("data/confuse.json");
    std::ifstream in(dat.c_str());
    if (!in.is_open()) {
        std::cerr << "open data " << dat << " file failed " << std::endl;
        return EXIT_FAILURE;
    }
    std::stringstream sin;
    sin << in.rdbuf();
    in.close();
    std::string data(sin.str());
    if (data.empty()) {
        return EXIT_SUCCESS;
    }

    bool res;
    JSONCPP_STRING errs;
    Json::Value root;
    Json::CharReaderBuilder readerBuilder;

    std::unique_ptr<Json::CharReader> const jsonReader(readerBuilder.newCharReader());
    res = jsonReader->parse(data.c_str(), data.c_str() + data.length(), &root, &errs);

    if (!res || !errs.empty()) {
        std::cerr << "parseJson err. " << errs << std::endl;
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
            std::cerr << "open json data " << dat << "key:" << *iter << "  failed " << std::endl;
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
std::string normalizeStr(const std::string& str) {
    std::stringstream output;


    utf8_iter ITER;
    utf8_init(&ITER, str.c_str());
    std::string tmps;
    while (utf8_next(&ITER)) {
        tmps = utf8_getchar(&ITER);
        if (_WordMap.find(tmps) != _WordMap.end()) {
            output << _WordMap.at(tmps);
        } else {
            output << tmps;
        }
    }
    return output.str();
}

#endif  // SRC_UTILS_NORM_CHR_HPP_
