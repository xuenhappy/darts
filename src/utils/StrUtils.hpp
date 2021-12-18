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

#ifndef SRC_UTILS_STRUTILS_HPP_
#define SRC_UTILS_STRUTILS_HPP_

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
namespace darts {
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

#endif  // SRC_UTILS_STRUTILS_HPP_
