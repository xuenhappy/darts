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
#include <string>
#include <vector>
#include "../core/darts.hpp"

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

class WordPice {
   public:
    /**
     * @brief 对给定的字符串句子进行分解
     *
     */
    void encode(const darts::AtomList& input, std::function<void(int code, int atom_postion)> hit) const {}

    /**
     * @brief 输出原始的code对应的字符串
     *
     * @param code
     * @return const char32_t*
     */
    const char32_t* decode(int code) const;
};

#endif  //!__BPE_MODEL__H__