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

#include <map>
#include <sstream>
#include <string>

#include "utf8.hpp"

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
std::string normalizeStr(const std::string& str) {
  std::stringstream output;

  utf8_iter ITER;
  utf8_init(&ITER, str.c_str());
  const char* tmps;
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

/**
 * @brief
 *
 */
void test() {}

#endif  // SRC_UTILS_NORM_CHR_HPP_
