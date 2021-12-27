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
#include <map>
#include <string>
#include <vector>

#include "file_utils.hpp"
/**
 * @brief 拼音数据
 *
 */

typedef struct _pinyin {
    std::string key;
    std::vector<std::string> piyins;
} * PinyinInfo;

static std::map<std::string, PinyinInfo> _WordPinyin;
/**
 * @brief 加载拼音数据
 *
 */
int loadPinyin() {
    std::string dat = getResource("data/pinyin.txt");
    std::ifstream in(dat.c_str());
    if (!in.is_open()) {
        std::cerr << "open data " << dat << " file failed " << std::endl;
        return EXIT_FAILURE;
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
const PinyinInfo pinyin(const std::string &word) {
    if (_WordPinyin.find(word) != _WordPinyin.end()) {
        return _WordPinyin[word];
    }
    return NULL;
}

#endif  // SRC_UTILS_PINYIN_HPP_
