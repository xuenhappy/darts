/*
 * File: chspliter.hpp
 * Project: utils
 * 基础字符串切分统计工具
 * File Created: Saturday, 25th December 2021 2:35:51 pm
 * Author: Xu En (xuen@mokar.com)
 * -----
 * Last Modified: Saturday, 25th December 2021 2:35:54 pm
 * Modified By: Xu En (xuen@mokahr.com)
 * -----
 * Copyright 2021 - 2021 Your Company, Moka
 */

#ifndef SRC_UTILS_CHSPLITER_HPP_
#define SRC_UTILS_CHSPLITER_HPP_
#include <fstream>
#include <map>
#include <string>

#include "../core/wtype.h"
#include "file_utils.hpp"
/**
 * @brief 字符串类型数据
 *
 */
std::map<std::string, WordType> _charType;

int loadCharMap() {
  std::string dat = getResource("data/chars.tmap");
  std::ifstream in(dat.c_str());
  if (!in.is_open()) {
    std::cerr << "open data " << dat << " file failed " << std::endl;
    return EXIT_FAILURE;
  }

  in.close();

  return EXIT_SUCCESS;
}

/**
 * @brief 返回某个字符对应的字符类型
 *
 * @param chr
 * @return WordType
 */
WordType charType(const std::string& chr) { return WordType::UNK; }

/**
 * @brief 计算字符串word个数
 *
 * @param str
 * @return size_t
 */
size_t wordLen(const std::string& str) { return 0; }

/**
 * @brief 对原始字符串进行切分
 *
 * @param str 原始字符串
 * @param accept hook函数
 */
void atomSplit(const std::string& str,
               void (*accept)(const char*, WordType, size_t, size_t)) {}

#endif  // SRC_UTILS_CHSPLITER_HPP_
