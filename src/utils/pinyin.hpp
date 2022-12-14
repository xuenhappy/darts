/*
 * File: pinyin.hpp
 * Project: utils
 * 中文字符串拼音工具
 * File Created: Saturday, 25th December 2021 2:37:57 pm
 * Author: Xu En (xuen@mokar.com)
 * -----
 * Last Modified: Saturday, 25th December 2021 2:38:01 pm
 * Modified By: Xu En (xuen@mokahr.com)
 * -----
 * Copyright 2021 - 2021 Your Company, Moka
 */
#ifndef SRC_UTILS_PINYIN_HPP_
#define SRC_UTILS_PINYIN_HPP_

#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "./filetool.hpp"
#include "./strtool.hpp"
#include "./utf8.hpp"
/**
 * @brief 拼音数据
 *
 */

typedef struct _pinyin {
    char32_t code;
    std::string key;
    std::vector<std::string> piyins;
} PinyinInfo;

static std::unordered_map<std::string, std::shared_ptr<PinyinInfo>> _WordPinyin;
/**
 * @brief 加载拼音数据
 *
 */
inline int loadPinyin() {
    std::string dat = getResource("data/kernel/pinyin.txt");
    std::ifstream in(dat.c_str());
    if (!in.is_open()) {
        std::cerr << "ERROR: open data " << dat << " file failed " << std::endl;
        return EXIT_FAILURE;
    }
    std::string line;
    size_t s, e;
    while (std::getline(in, line)) {
        darts::trim(line);
        if (line.empty() || line.length() < 4) {
            continue;
        }
        if (line[0] == '#') continue;  // comment
        if (line[0] != 'U' || line[1] != '+') {
            std::cerr << "WARN: Bad line input " << line << std::endl;
            continue;
        }

        s = line.find(":");
        e = line.find("#", s + 1);
        if (s == std::string::npos || e <= s) {
            std::cerr << "WARN: Bad line input " << line << std::endl;
            continue;
        }
        std::string nums = line.substr(2, s);
        darts::trim(nums);
        std::string pinyin = line.substr(s + 1, e);
        darts::trim(pinyin);
        uint32_t code = std::stoul(nums, nullptr, 16);

        std::shared_ptr<PinyinInfo> info = std::make_shared<PinyinInfo>();

        info->code = code;
        info->key  = unicode_to_utf8(code);
        darts::split(pinyin, ",", info->piyins);
        std::vector<std::string>::iterator it = info->piyins.begin();
        while (it != info->piyins.end()) {
            darts::trim(*it);
            if (it->empty()) {
                it = info->piyins.erase(it);
                continue;
            }
            ++it;
        }
        if (!info->piyins.empty()) {
            // std::cout << info->key << "," << darts::join(info->piyins, "|") << std::endl;
            _WordPinyin[info->key] = std::move(info);
        }
    }

    in.close();

    return EXIT_SUCCESS;
}

/**
 * @brief 获取给定中文的全部拼音
 *
 * @param word
 * @return const PinyinInfo*
 */
inline const std::shared_ptr<PinyinInfo> pinyin(const std::string& word) {
    auto it = _WordPinyin.find(word);
    if (it != _WordPinyin.end()) return it->second;
    return nullptr;
}

#endif  // SRC_UTILS_PINYIN_HPP_
