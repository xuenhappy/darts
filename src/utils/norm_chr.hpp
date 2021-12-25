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
#include <map>
#include <sstream>
#include <string>

#include "utf8.h"

/**
 * @brief  字符串规范化使用的工具
 *
 */
const std::map<std::string, std::string> _WordMap;

/**
 * @brief 对原始字符串进行字符归一化
 *
 * @param str
 * @return std::string
 */
std::string normalizeStr(const std::string &str) {
    std::stringstream output;

    return output.str();
}

#endif  // SRC_UTILS_NORM_CHR_HPP_
