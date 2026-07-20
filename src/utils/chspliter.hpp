/*
 * File: chspliter.hpp
 * Project: utils
 * 基础字符串切分统计工具
 * File Created: Saturday, 25th December 2021 2:35:51 pm
 * Author: Xu En (nanhangxuen@163.com)
 * -----
 * Last Modified: Saturday, 25th December 2021 2:35:54 pm
 * Modified By: Xu En (nanhangxuen@163.com)
 * -----
 * Copyright 2021 - 2021 XuEn
 */

#ifndef SRC_UTILS_CHSPLITER_HPP_
#define SRC_UTILS_CHSPLITER_HPP_
#include <cctype>
#include <cwctype>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include "./filetool.hpp"
#include "./strtool.hpp"
#include "./utf8.hpp"

/**
 * @brief 字符串类型数据
 *
 */
inline std::unordered_map<uint32_t, std::string> _charType;
inline std::unordered_set<uint32_t> _cjkCodes;

namespace char_type {
static const std::string ENG("ENG");
static const std::string EMPTY("EMPTY");
static const std::string POS("POS");
static const std::string NUM("NUM");
static const std::string CJK("CJK");
static const std::string WUNK("WUNK");
}  // namespace char_type

inline void load_cjk_words_() {
    _cjkCodes.insert(0x3005);
    _cjkCodes.insert(0x3007);
    _cjkCodes.insert(0x303B);
    const uint32_t LHan[][2] = {
        {0x3400, 0x4DBF},   {0x4E00, 0x9FFF},   {0xF900, 0xFAFF},   {0x20000, 0x2A6DF},
        {0x2A700, 0x2B73F}, {0x2B740, 0x2B81F}, {0x2B820, 0x2CEAF}, {0x2CEB0, 0x2EBEF},
        {0x2EBF0, 0x2EE5F}, {0x2F800, 0x2FA1F}, {0x30000, 0x3134F}, {0x31350, 0x3347F},
    };

    size_t row = sizeof(LHan) / sizeof(LHan[0]);
    for (size_t r = 0; r < row; ++r)
        for (uint32_t a = LHan[r][0]; a < LHan[r][1] + 1; ++a)
            if (_charType.find(a) == _charType.end()) _cjkCodes.insert(a);
}

inline int loadCharMap() {
    std::string dat = getResource("data/kernel/chars.tmap");
    std::ifstream in(dat.c_str());
    if (!in.is_open()) {
        std::cerr << "ERROR: open data " << dat << " file failed " << std::endl;
        return EXIT_FAILURE;
    }
    std::string head("-%");
    std::string line;
    std::string prefix = "";
    _charType.clear();
    _charType.reserve(10000);
    _cjkCodes.clear();
    while (std::getline(in, line)) {
        darts::trim(line);
        if (line.empty()) continue;
        if (!strncmp(line.c_str(), head.c_str(), head.size())) {
            prefix = line.substr(head.size());
            darts::trim(prefix);
            if (prefix.empty()) std::cerr << "WARN: Bad word type tag name" << line.substr(head.size()) << std::endl;
            continue;
        }
        utf8_iter ITER;
        utf8_initEx(&ITER, line.c_str(), line.length());
        while (utf8_next(&ITER)) _charType[ITER.codepoint] = prefix;
    }
    in.close();
    load_cjk_words_();
    return EXIT_SUCCESS;
}

/**
 * @brief 返回某个字符对应的字符类型
 *
 * @param code
 * @return WordType
 */
inline const std::string& charType(uint32_t code) {
    if (std::iswspace(code)) return char_type::EMPTY;
    if (!std::iswprint(code)) return char_type::POS;
    auto it = _charType.find(code);
    if (it != _charType.end()) return it->second;
    if (code < 255) {
        if ('A' <= code && code <= 'Z') return char_type::ENG;
        if ('a' <= code && code <= 'z') return char_type::ENG;
        if ('0' <= code && code <= '9') return char_type::NUM;
        return char_type::POS;
    }
    if (65296 <= code && code <= 65305) return char_type::NUM;
    if (65313 <= code && code <= 65338) return char_type::ENG;
    if (_cjkCodes.find(code) != _cjkCodes.end()) return char_type::CJK;
    return char_type::WUNK;
}

/**
 * @brief 计算字符串word个数
 *
 * @param str
 * @return size_t
 */
static inline size_t wordLen(const std::string& str) {
    if (str.empty()) return 0;
    size_t nums = 0;
    std::string btype("");
    utf8_iter ITER;
    utf8_initEx(&ITER, str.c_str(), str.size());
    while (utf8_next(&ITER)) {
        auto& ctype = charType(ITER.codepoint);
        if (ctype != btype) {
            if ((ctype == char_type::EMPTY && btype == char_type::POS) ||
                (btype == char_type::EMPTY && ctype == char_type::POS)) {
                btype = char_type::POS;
                continue;
            }
            if (!btype.empty()) {
                nums += 1;
                btype = "";
            }
        }
        btype = ctype;
        // CJK characters and punctuation are semantic boundaries. Keeping
        // adjacent punctuation in one atom (for example "%。" or "”，")
        // prevents recognizers from building spans that end at the first mark.
        if (btype == char_type::CJK || btype == char_type::POS) {
            nums += 1;
            btype = "";
            continue;
        }
    }
    if (!btype.empty()) nums += 1;

    return nums;
}
class U32StrIter {
   public:
    virtual void iter(std::function<void(int64_t code, const char* str, size_t idx)> hit) const = 0;
    virtual ~U32StrIter() {}
};

class U8T32Iter_ : public U32StrIter {
   private:
    const std::string* str;

   public:
    explicit U8T32Iter_(const std::string& str) { this->str = &str; }
    void iter(std::function<void(int64_t, const char*, size_t)> hit) const {
        utf8_iter ITER;
        int position = -1;
        utf8_initEx(&ITER, str->c_str(), str->length());
        while (utf8_next(&ITER)) {
            position++;
            hit(ITER.codepoint, utf8_getchar(&ITER), position);
        }
    }
};

class U32T32Iter_ : public U32StrIter {
   private:
    const std::u32string* str;

   public:
    explicit U32T32Iter_(const std::u32string& str) { this->str = &str; }
    void iter(std::function<void(int64_t, const char*, size_t)> hit) const {
        for (size_t position = 0; position < str->length(); position++) {
            int64_t code = (*str)[position];
            hit(code, unicode_to_utf8(code), position);
        }
    }
};

typedef std::function<void(const std::string&, const std::string&, size_t, size_t)> asplit_hit_func;

inline void atomSplit(const U32StrIter& str, asplit_hit_func accept, bool merge_pos_space = true) {
    size_t bufstart = 0, pos = 0;
    std::string btype = "", cbuffer = "";

    auto hit = [&](int64_t code, const char* ct, size_t position) {
        pos         = position;
        auto& ctype = charType(code);
        if (ctype != btype) {
            if (merge_pos_space && ctype == char_type::EMPTY && btype == char_type::POS) {
                btype = char_type::POS;
                cbuffer.append(ct);
                return;
            }
            if (merge_pos_space && btype == char_type::EMPTY && ctype == char_type::POS) {
                btype = char_type::POS;
                cbuffer.append(ct);
                return;
            }

            if (!cbuffer.empty()) accept(cbuffer, btype, bufstart, position);

            bufstart = position;
            cbuffer.clear();
        }

        btype = ctype;

        if (btype == char_type::CJK || btype == char_type::POS) {
            accept(ct, btype, position, position + 1);
            cbuffer.clear();
            return;
        }
        cbuffer.append(ct);
    };
    str.iter(hit);
    if (!cbuffer.empty()) accept(cbuffer, btype, bufstart, pos + 1);
}
/**
 * @brief 对原始字符串进行切分
 *
 * @param str 原始字符串
 * @param accept hook函数
 */
inline void atomSplit(const std::string& str, asplit_hit_func accept, bool merge_pos_space = true) {
    U8T32Iter_ iter(str);
    atomSplit(iter, accept, merge_pos_space);
}

inline void atomSplit(const std::u32string& str, asplit_hit_func accept, bool merge_pos_space = true) {
    U32T32Iter_ iter(str);
    atomSplit(iter, accept, merge_pos_space);
}

#endif  // SRC_UTILS_CHSPLITER_HPP_
