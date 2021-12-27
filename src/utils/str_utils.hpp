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

#include <algorithm>
#include <cctype>
#include <functional>
#include <locale>
#include <sstream>
#include <string>
#include <vector>

namespace darts {

/**
 * @brief 字符串连接
 *
 * @param v
 * @param delt
 * @return std::string
 */
std::string join(const std::vector<std::string> &v, const std::string &delt) {
    std::stringstream ss;
    std::vector<std::string>::const_iterator it = v.begin();
    if (it != v.end()) {
        ss << *it;
        ++it;
    }
    for (; it != v.end(); ++it) {
        ss << delt << *it;
    }
    return ss.str();
}
}  // namespace darts


/**
 * @brief trim from start (in place)
 *
 * @param s
 */
static inline std::string &ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int c) { return !std::isspace(c); }));
    return s;
}

/**
 * @brief
 *
 * @param s
 */
static inline std::string &rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](int c) { return !std::isspace(c); }).base(), s.end());
    return s;
}

// trim from both ends (in place)
static inline std::string &trim(std::string &s) { return rtrim(ltrim(s)); }


static inline std::string ltrim_copy(const char *cs) {
    std::string s(cs);
    return ltrim(s);
}


static inline std::string rtrim_copy(const char *cs) {
    std::string s(cs);
    return rtrim(s);
}

static inline std::string trim_copy(const char *cs) {
    std::string s(cs);
    return trim(s);
}

#endif  // SRC_UTILS_STR_UTILS_HPP_
