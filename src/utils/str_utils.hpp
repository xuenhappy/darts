/*
 * File: StrUtils.hpp
 * Project: utils
 * 提供一些基础的字符串处理工具
 * File Created: Saturday, 11th December 2021 7:55:02 pm
 * Author: Xu En (xuen@mokar.com)
 * -----
 * Last Modified: Saturday, 11th December 2021 7:55:13 pm
 * Modified By: Xu En (xuen@mokahr.com)
 * -----
 * Copyright 2021 - 2021 Your Company, Moka
 */

#ifndef SRC_UTILS_STR_UTILS_HPP_
#define SRC_UTILS_STR_UTILS_HPP_

#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "../core/wtype.h"
namespace darts {
/**
 * @brief  字符串规范化使用的工具
 *
 */
const std::map<std::string, std::string> _WordMap;

/**
 * @brief 拼音数据
 *
 */
const std::map<std::string, std::string[]> _WordPinyin;

/**
 * @brief 字符串类型数据
 *
 */
const std::map<std::string, WordType> _charType;

/**
 * @brief 返回某个字符对应的字符类型
 *
 * @param chr
 * @return WordType
 */
WordType charType(const std::string& chr) {
    return WordType::UNK;
}

/**
 * @brief 字符串连接
 *
 * @param v
 * @param delt
 * @return std::string
 */
std::string join(const std::vector<std::string>& v, std::string& delt) {
    std::stringstream ss;
    std::vector<std::string>::const_iterator it = v.begin();
    if (it != v.end()) {
        ss << *it;
        ++it;
    }
    for (; it != v.end(); ++it) {
        ss << delt << *it;
    }
    ss << std::endl;
    return ss.str();
}
}  // namespace darts

#endif  // SRC_UTILS_STR_UTILS_HPP_
