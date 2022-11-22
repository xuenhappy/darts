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
#ifndef SRC_IMPL_ENCODER_HPP_
#define SRC_IMPL_ENCODER_HPP_

#include <fmt/core.h>
#include <algorithm>
#include <functional>
#include <limits>
#include <list>
#include <memory>
#include <queue>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "../core/darts.hpp"
#include "../utils/biggram.hpp"

struct _SymbolPair {
    int left;      // left index of this pair
    int right;     // right index of this pair
    double score;  // score of this pair. small is better
};

struct _Symbol {
    int start;
    int size;
    int idx;
    bool freeze;
};
bool symbol_compare(_Symbol i1, _Symbol i2) {
    return (i1.start < i2.start) || (i1->start == i2->start && i1->size < i2->size);
}

bool is_digits(const std::string& str) {
    return std::all_of(str.begin(), str.end(), ::isdigit);  // C++11
}

namespace codemap {
// const code
static const int pad_code  = 0;
static const int unk_code  = 1;
static const int cls_code  = 2;
static const int sep_code  = 3;
static const int mask_code = 4;

// const code char
static const char* unk_char  = "[UNK]";
static const char* cls_char  = "[CLS]";
static const char* sep_char  = "[SEP]";
static const char* mask_char = "[MASK]";
static const char* pad_char  = "[PAD]";
}  // namespace codemap

class WordPice {
   private:
    darts::BigramDict english_token_dict;
    std::unordered_map<std::string, int> codes;
    std::vector<std::string> chars_list;
    // code it

   private:
    void numToken(const std::string& num, int piceNum, std::vector<std::string>& ret) const {
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

    void engToken(const std::string& eng, std::vector<std::string>& ret) const {
        // token english str
        if (eng.length() < 2 || eng.length() > 50) {  // too long or short codes
            ret.push_back(fmt::format("▁{}", eng));
            return;
        }
        // load code
        std::vector<_Symbol> symbols;
        symbols.reserve(eng.size() * 2);

        // Pre-allocates SymbolPair for efficiency.
        symbol_pair_pre_nums = 256;
        std::vector<_SymbolPair> symbol_pairs(symbol_pair_pre_nums);
        size_t symbol_pair_allocator_nums = 0;

        // Lookup new symbol pair at [left, right]
        auto add_symbol_pair = [this, &symbol_pair_allocator_nums, &symbol_pair_pre_nums, &symbol_pairs, &symbols](
                                   int left, int right) {
            if (symbol_pair_allocator_nums >= symbol_pair_pre_nums) {
                symbol_pair_pre_nums *= 2;
                symbol_pairs.resize(symbol_pair_pre_nums);
            }
            auto* h  = &symbol_pairs[symbol_pair_allocator_nums++];
            h->left  = left;
            h->right = right;
            h->score =
                left < 0 || right < 0 ? 0 : this->english_token_dict.wordDist(symbols[left].idx, symbols[right].idx);
        };

        // Splits the input into character sequence
        auto token_hit = [&symbols](int pos, int size, int idx) {
            if (size > 1) {
                _Symbol s;
                s.start  = pos;
                s.size   = size;
                s.idx    = idx;
                s.freeze = false;
                symbols.emplace_back(s);
            }
        };
        for (size_t i = 0; i < eng.size(); i++) {
            _Symbol s;
            s.start  = i;
            s.size   = 1;
            s.idx    = english_token_dict.getWordKey(eng.substr(i, 1));
            s.freeze = false;
            symbols.emplace_back(s);
        }
        english_token_dict.matchKey(eng, token_hit);
        std::sort(symbols.begin(), symbols.end(), symbol_compare);
        // Lookup all bigrams.
        for (size_t i = 0; i < symbols.size(); i++) {
            if (symbols[i].start == 0) {
                add_symbol_pair(-1, i);
                continue;
            }
            break;
        }
        for (size_t i = 0; i < symbols.size(); i++) {
            auto pre_symbol = &symbols[i];
            int nextpos     = pre_symbol->start + pre_symbol->size;
            if (nextpos >= eng.size()) {
                add_symbol_pair(i, -1);
                break;
            }
            for (size_t j = 0; j < symbols.size(); j++) {
                auto next_symbol = &symbols[j];
                if (next_symbol->start == nextpos) {
                    add_symbol_pair(i, j);
                    continue;
                }
                if (next_symbol->start > nextpos) break;
            }
        }
        // select best tokens
        std::vector<double> dist(eng.size() + 2, std::numeric_limits<double>::max);
        std::vector<int> prev(eng.size() + 2, -2);
        using std::pair<double, int> iPair;
        std::priority_queue<iPair, std::vector<iPair>, std::greater<iPair> > pq;
        pq.push(std::make_pair(0.0, -1));
        dist[0] = 0;
        prev[0] = -2;
        while (!pq.empty()) {
            int u = pq.top()->right;
            pq.pop();
            if (u == -1) continue;

            for (size_t i = u; i < symbol_pair_allocator_nums; i++) {
                if (symbol_pairs[i].right == u) {
                    // adjacent of u.
                    int v         = symbol_pairs[i].left;
                    double weight = symbol_pairs[i].score;

                    int pv = v < 0 ? dist.size() - 1 : v + 1;

                    // If there is shorted path to v through u.
                    if (dist[pv] > dist[u + 1] + weight) {
                        // Updating distance of v
                        dist[pv] = dist[u + 1] + weight;
                        prev[pv] = u;
                        pq.push(std::make_pair(dist[v], v));
                    }
                    continue;
                }
                if (symbol_pairs[i].right > u) break;
            }
        }
    }

   public:
    /**
     * @brief Construct a new Word Pice object
     *
     * @param dict_path 字符映射表
     * @param engdict_dir 英文单词切分使用的数据
     */
    WordPice(const std::string& dict_path, const std::string& engdict_dir) {
        std::string releng_dir = getResource(engdict_dir, true);
        if (!english_token_dict.loadDict(ddir)) {
            std::cerr << "ERROR: load english token dict dir " << engdict_dir << " failed " << std::endl;
            return;
        }
        std::string relpath = getResource(dict_path);
        // load codes and chars_list
        std::set<std::string> _list;
        // load relpath data into _list
        std::ifstream fin(relpath.c_str());
        if (!fin.is_open()) {
            std::cerr << "ERROR: open data " << relpath << " file failed " << std::endl;
            return;
        }
        std::string line;
        size_t s;
        while (std::getline(fin, line)) {
            darts::trim(line);
            if (line.empty()) continue;
            // split line use first item
            s = line.find(' ');
            if (s != std::string::npos) {
                line = line.substr(0, s);
            }
            s = line.find('\t');
            if (s != std::string::npos) {
                line = line.substr(0, s);
            }
            if (line.empty()) continue;
            _list.insert(line);
        }
        fin.close();
        // init data
        chars_list.reserve(_list.size() + 10);
        chars_list.resize(5);
        chars_list[codemap::pad_code]  = codemap::pad_char;
        chars_list[codemap::unk_code]  = codemap::unk_char;
        chars_list[codemap::cls_code]  = codemap::cls_char;
        chars_list[codemap::sep_code]  = codemap::sep_char;
        chars_list[codemap::mask_code] = codemap::mask_char;
        chars_list.insert(chars_list.end(), _list.begin(), _list.end());
        for (int i = 0; i < chars_list.size(); i++) {
            codes[chars_list[i]] = i;
        }
    }
    /**
     * @brief 对给定的字符串句子进行分解
     *
     */
    void encode(const darts::AtomList& input, std::function<void(int code, int atom_postion)> hit,
                bool skip_empty_token) const {
        std::vector<std::string> _cache;
        hit(codemap::sep_code, -1);
        std::unordered_map<std::string, int>::const_iterator _it;
        int postion = -1;
        for (std::shared_ptr<darts::Atom> atom : input) {
            postion += 1;
            if (skip_empty_token && atom->hasLabel("EMPTY")) continue;
            if (atom->masked) {  // mask atom
                hit(codemap::mask_code, postion);
                continue;
            }
            if (atom->hasLabel("ENG")) {
                _cache.clear();
                engToken(atom->image, _cache);
                for (std::string w : _cache) {
                    _it = codes.find(w);
                    if (_it == codes.end()) {
                        hit(codemap::unk_code, postion);
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
                        hit(codemap::unk_code, postion);
                        continue;
                    }
                    hit(_it->second, postion);
                }
                continue;
            }
            _it = codes.find(atom->image);
            if (_it == codes.end()) {
                hit(codemap::unk_code, postion);
                continue;
            }
            hit(_it->second, postion);
        }
        hit(codemap::cls_code, -1);
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

#endif  // SRC_IMPL_ENCODER_HPP_
