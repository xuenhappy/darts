/*
 * File: utils_base.hpp
 * Project: utils
 * File Created: Saturday, 25th December 2021 9:06:57 pm
 * Author: Xu En (xuen@mokar.com)
 * -----
 * Last Modified: Saturday, 25th December 2021 9:07:01 pm
 * Modified By: Xu En (nanhangxuen@163.com)
 * -----
 * Copyright 2021 - 2021 Your Company, Moka
 */
#ifndef SRC_UTILS_UTILS_BASE_HPP_
#define SRC_UTILS_UTILS_BASE_HPP_

#include <mutex>
#include "chrtool.hpp"
#include "chspliter.hpp"
#include "pinyin.hpp"

inline std::once_flag init_flag;
inline int init_result = EXIT_SUCCESS;

inline int initUtils() {
    std::call_once(init_flag, []() {
        if (loadCharMap() || loadPinyin() || loadPhrasePinyin() || initializeMap()) init_result = EXIT_FAILURE;
    });
    return init_result;
}

#endif  // SRC_UTILS_UTILS_BASE_HPP_
