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
#include <cctype>
#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <string>

#include "../core/wtype.h"
#include "file_utils.hpp"
#include "utf8.hpp"

/**
 * @brief 字符串类型数据
 *
 */
static std::map<uint32_t, WordType> _charType;

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
 * @param code
 * @return WordType
 */
WordType charType(uint32_t code) {
    if (std::isspace(code)) {
        return WordType::EMPTY;
    }

    // if (!std::iswprint(code)) {
    //     return WordType::POS;
    // }
    if (_charType.find(code) != _charType.end()) {
        return _charType[code];
    }

    if (code < 255) {
        if ('A' <= code && code <= 'Z') {
            return WordType::ENG;
        }
        if ('a' <= code && code <= 'z') {
            return WordType ::ENG;
        }
        if ('0' <= code && code <= '9') {
            return WordType::NUM;
        }
        return WordType::POS;
    }
    if (65296 <= code && code <= 65305) {
        return WordType::NUM;
    }
    if (65313 <= code && code <= 65338) {
        return WordType::ENG;
    }
    if (0x4e00 <= code && code <= 0x9fa5) {
        return WordType::CJK;
    }
    if (0x3130 <= code && code <= 0x318F) {
        return WordType::CJK;
    }
    if (0xAC00 <= code && code <= 0xD7A3) {
        return WordType::CJK;
    }
    if (0x0800 <= code && code <= 0x4e00) {
        return WordType::CJK;
    }
    if (0x3400 <= code && code <= 0x4DB5) {
        return WordType::CJK;
    }
    return WordType::UNK;
}

/**
 * @brief 计算字符串word个数
 *
 * @param str
 * @return size_t
 */
size_t wordLen(const char* str) {
    if (!str) return 0;
    size_t nums = 0;
    WordType buf_type = WordType::NONE;
    utf8_iter ITER;
    utf8_init(&ITER, str);
    while (utf8_next(&ITER)) {
        auto ctype = charType(ITER.codepoint);
        if (ctype != buf_type) {
            if ((ctype == WordType::EMPTY && buf_type == WordType::POS) ||
                (buf_type == WordType::EMPTY && ctype == WordType::POS)) {
                buf_type = WordType::POS;
                continue;
            }
            if (buf_type != WordType::NONE) {
                nums += 1;
                buf_type = WordType::NONE;
            }
        }
        buf_type = ctype;
        if (buf_type == WordType::CJK) {
            nums += 1;
            buf_type = WordType::NONE;
            continue;
        }
    }
    if (buf_type != WordType::NONE) {
        nums += 1;
    }
    return nums;
}

/**
 * @brief 对原始字符串进行切分
 *
 * @param str 原始字符串
 * @param accept hook函数
 */
void atomSplit(const char* str, std::function<void(const char*, WordType, size_t, size_t)> accept) {
    std::string chr_buffer;
    size_t bufstart = 0;
    auto buf_type = WordType::NONE;
    int position = -1;
    utf8_iter ITER;
    utf8_init(&ITER, str);
    while (utf8_next(&ITER)) {
        position++;
        auto ctype = charType(ITER.codepoint);
        if (ctype != buf_type) {
            if (ctype == WordType::EMPTY && buf_type == WordType::POS) {
                buf_type = WordType::POS;
                chr_buffer.append(utf8_getchar(&ITER));
                continue;
            }
            if (buf_type == WordType::EMPTY && ctype == WordType::POS) {
                buf_type = WordType::POS;
                chr_buffer.append(utf8_getchar(&ITER));
                continue;
            }

            if (!chr_buffer.empty()) {
                accept(chr_buffer.c_str(), buf_type, bufstart, position);
            }
            bufstart = position;
            chr_buffer.clear();
        }

        buf_type = ctype;

        if (buf_type == WordType::CJK) {
            accept(utf8_getchar(&ITER), buf_type, position, position + 1);
            chr_buffer.clear();
            continue;
        }
        chr_buffer.append(utf8_getchar(&ITER));
    }

    if (!chr_buffer.empty()) {
        accept(chr_buffer.c_str(), buf_type, bufstart, position + 1);
    }
}

#endif  // SRC_UTILS_CHSPLITER_HPP_
