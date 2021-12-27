/*
 * File: utils_base.hpp
 * Project: utils
 * File Created: Saturday, 25th December 2021 9:06:57 pm
 * Author: Xu En (xuen@mokar.com)
 * -----
 * Last Modified: Saturday, 25th December 2021 9:07:01 pm
 * Modified By: Xu En (xuen@mokahr.com)
 * -----
 * Copyright 2021 - 2021 Your Company, Moka
 */
#ifndef SRC_UTILS_UTILS_BASE_HPP_
#define SRC_UTILS_UTILS_BASE_HPP_

#include "chspliter.hpp"
#include "norm_chr.hpp"
#include "pinyin.hpp"

void initUtils() {
    if (loadCharMap()) {
        exit(1);
    }
    if (loadPinyin()) {
        exit(1);
    }
    if (initializeMap()) {
        exit(1);
    }
}

#endif  // SRC_UTILS_UTILS_BASE_HPP_
