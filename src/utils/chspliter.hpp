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
#include <sstream>
#include <string>
#include <unordered_map>
#include "file_utils.hpp"
#include "str_utils.hpp"
#include "utf8.hpp"

/**
 * @brief 字符串类型数据
 *
 */
static std::unordered_map<uint32_t, std::string> _charType;

int loadCharMap() {
    std::string dat = getResource("data/chars.tmap");
    std::ifstream in(dat.c_str());
    if (!in.is_open()) {
        std::cerr << "ERROR: open data " << dat << " file failed " << std::endl;
        return EXIT_FAILURE;
    }
    std::string head("-%");
    std::string line;
    std::string prefix = "";
    while (std::getline(in, line)) {
        darts::trim(line);
        if (line.empty()) {
            continue;
        }
        if (!strncmp(line.c_str(), head.c_str(), head.size())) {
            prefix = line.substr(head.size());
            darts::trim(prefix);
            if (prefix.empty()) {
                std::cerr << "WARN: Bad word type tag name" << line.substr(head.size()) << std::endl;
            }
            continue;
        }
        utf8_iter ITER;
        utf8_init(&ITER, line.c_str());
        while (utf8_next(&ITER)) {
            _charType[ITER.codepoint] = prefix;
        }
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
std::string charType(uint32_t code) {
    if (std::isspace(code)) {
        return "EMPTY";
    }

    // if (!std::iswprint(code)) {
    //     return WordType::POS;
    // }
    auto it = _charType.find(code);
    if (it != _charType.end()) {
        return it->second;
    }

    if (code < 255) {
        if ('A' <= code && code <= 'Z') {
            return "ENG";
        }
        if ('a' <= code && code <= 'z') {
            return "ENG";
        }
        if ('0' <= code && code <= '9') {
            return "NUM";
        }
        return "POS";
    }
    if (65296 <= code && code <= 65305) {
        return "NUM";
    }
    if (65313 <= code && code <= 65338) {
        return "ENG";
    }
    if (0x4e00 <= code && code <= 0x9fa5) {
        return "CJK";
    }
    if (0x3130 <= code && code <= 0x318F) {
        return "CJK";
    }
    if (0xAC00 <= code && code <= 0xD7A3) {
        return "CJK";
    }
    if (0x0800 <= code && code <= 0x4e00) {
        return "CJK";
    }
    if (0x3400 <= code && code <= 0x4DB5) {
        return "CJK";
    }
    return "WUNK";
}

/**
 * @brief 计算字符串word个数
 *
 * @param str
 * @return size_t
 */
size_t wordLen(const char* str) {
    if (!str) return 0;
    size_t nums          = 0;
    std::string buf_type = "";
    utf8_iter ITER;
    utf8_init(&ITER, str);
    while (utf8_next(&ITER)) {
        auto ctype = charType(ITER.codepoint);
        if (ctype != buf_type) {
            if ((ctype == "EMPTY" && buf_type == "POS") || (buf_type == "EMPTY" && ctype == "POS")) {
                buf_type = "POS";
                continue;
            }
            if (buf_type != "") {
                nums += 1;
                buf_type = "";
            }
        }
        buf_type = ctype;
        if (buf_type == "CJK") {
            nums += 1;
            buf_type = "";
            continue;
        }
    }
    if (buf_type != "") {
        nums += 1;
    }
    return nums;
}

class _UTF8StringIter : public darts::U32Iter {
   private:
    const char* str;

   public:
    explicit _UTF8StringIter(const char* str) { this->str = str; }
    void iter(std::function<void(int64_t, const char*, size_t)> hit) {
        utf8_iter ITER;
        int position = -1;
        utf8_init(&ITER, this->str);
        while (utf8_next(&ITER)) {
            position++;
            hit(ITER.codepoint, utf8_getchar(&ITER), position);
        }
    }
};

class _U32StringIter : public darts::U32Iter {
   private:
    const std::u32string* str;

   public:
    explicit _U32StringIter(const std::u32string& str) { this->str = &str; }
    void iter(std::function<void(int64_t, const char*, size_t)> hit) {
        for (size_t position = 0; position < str->length(); position++) {
            int64_t code = (*str)[position];
            hit(code, unicode_to_utf8(code), position);
        }
    }
};

void atomSplit(darts::U32Iter& str, std::function<void(const char*, std::string&, size_t, size_t)> accept) {
    std::string chr_buffer;
    size_t bufstart      = 0;
    std::string buf_type = "";
    size_t last_pos      = 0;
    str.iter([&](int64_t code, const char* ct, size_t position) {
        last_pos   = position;
        auto ctype = charType(code);
        if (ctype != buf_type) {
            if (ctype == "EMPTY" && buf_type == "POS") {
                buf_type = "POS";
                chr_buffer.append(ct);
                return;
            }
            if (buf_type == "EMPTY" && ctype == "POS") {
                buf_type = "POS";
                chr_buffer.append(ct);
                return;
            }

            if (!chr_buffer.empty()) {
                accept(chr_buffer.c_str(), buf_type, bufstart, position);
            }
            bufstart = position;
            chr_buffer.clear();
        }

        buf_type = ctype;

        if (buf_type == "CJK") {
            accept(ct, buf_type, position, position + 1);
            chr_buffer.clear();
            return;
        }
        chr_buffer.append(ct);
    });
    if (!chr_buffer.empty()) {
        accept(chr_buffer.c_str(), buf_type, bufstart, last_pos + 1);
    }
}
/**
 * @brief 对原始字符串进行切分
 *
 * @param str 原始字符串
 * @param accept hook函数
 */
void atomSplit(const char* str, std::function<void(const char*, std::string&, size_t, size_t)> accept) {
    _UTF8StringIter iter(str);
    atomSplit(iter, accept);
}

void atomSplit(const std::u32string& str, std::function<void(const char*, std::string&, size_t, size_t)> accept) {
    _U32StringIter iter(str);
    atomSplit(iter, accept);
}

#endif  // SRC_UTILS_CHSPLITER_HPP_
