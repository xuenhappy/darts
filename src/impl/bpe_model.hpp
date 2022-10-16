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
#include "../core/darts.hpp"

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

/**
 * @brief  对文本预料进行新词发现，并在当前目录输出词频文件与二元语法统计词表
 * 算法参见 https://zhuanlan.zhihu.com/p/80385615
 * @param text_corpus_path 文本语料
 * @param mini_freq 最低的词频
 * @param max_words_num 输出的最大单词数目
 * @param topk_percent 输出前%k的最大的单词数目，这个值会与max_words_num并取最小值
 * @param min_pmi 最小的词概率墒
 * @param min_lw 最小词左右集合墒
 */
void bpe_train(const char* text_corpus_path, const int mini_freq, const int max_words_num, const float topk_percent,
               const float min_pmi, const float min_lw) {}

#endif  //!__BPE_MODEL__H__