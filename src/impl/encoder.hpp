/*
 * File: bpe_model.hpp
 * Project: impl
 * File Created: Saturday, 24th September 2022 3:17:35 pm
 * Author: dell (xuen@mokar.com)
 * this module is use for word pice and new word regnizer
 * -----
 * Last Modified: Saturday, 24th September 2022 3:17:39 pm
 * Modified By: dell (xuen@mokahr.com)
 * -----
 * Copyright 2021 - 2022 Your Company, Moka
 */
#ifndef __BPE_MODEL__H__
#define __BPE_MODEL__H__

#include <functional>
#include <map>
#include <string>
#include <vector>
#include "../core/darts.hpp"
bool is_digits(const std::string& str) {
    return std::all_of(str.begin(), str.end(), ::isdigit);  // C++11
}
class WordPice {
   private:
    std::map<std::string, int> codes;
    std::vector<std::string> chars_list;
    int unk_code;
    int cls_code;
    int sep_code;

   private:
    void numToken(const std::string& num, int piceNum, std::vector<std::string>& ret) {
        // token num str
        if (num.length() < piceNum) {
            ret.push_back(num);
            return;
        }
        int sidx    = 0;
        int leftnum = num.length() % piceNum;
        int lenOuts = num.length();
        if (leftnum != 0) {
            lenOuts += 1;
        } else {
            leftnum = piceNum;
        }
        for (int i = 0; i < lenOuts; i++) {
            int endi = piceNum * i + leftnum;
            ret.push_back(num.substr(sidx, endi - sidx));
            sidx = endi;
        }
    }

    void engToken(const std::string& eng, std::vector<std::string>& ret) {
        // token english str
    }

   public:
    /**
     * @brief 对给定的字符串句子进行分解
     *
     */
    void encode(const darts::AtomList& input, std::function<void(int code, int atom_postion)> hit) const {
        std::vector<std::string> _cache;
        hit(sep_code, -1);
        std::map<std::string, int>::const_iterator _it;
        int postion = -1;
        for (std::shared_ptr<darts::Atom> atom : input) {
            postion += 1;
            if (atom->hasLabel("ENG")) {
                _cache.clear();
                engToken(atom->image, _cache);
                for (std::string w : _cache) {
                    _it = codes.find(w);
                    if (_it == codes.end()) {
                        hit(unk_code, postion);
                        continue;
                    }
                    hit(_it->second, postion);
                }
                continue;
            }
            if (atom->hasLabel("NUM") && is_digits(atom->image)) {
                _cache.clear();
                numToken(atom->image, 3, _cache);
                for (std::string w : _cache) {
                    _it = codes.find(w);
                    if (_it == codes.end()) {
                        hit(unk_code, postion);
                        continue;
                    }
                    hit(_it->second, postion);
                }
                continue;
            }
            _it = codes.find(atom->image);
            if (_it == codes.end()) {
                hit(unk_code, postion);
                continue;
            }
            hit(_it->second, postion);
        }
        hit(cls_code, -1);
    }

    /**
     * @brief 输出原始的code对应的字符串
     *
     * @param code
     * @return const char32_t*
     */
    const char* decode(int code) const {
        if (code >= 0 && code < chars_list.size()) {
            return chars_list[code].c_str();
        }
        return NULL;
    }
};

#endif  //!__BPE_MODEL__H__