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
#include <functional>
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
    int left;     // left index of this pair
    int right;    // right index of this pair
    float score;  // score of this pair. large is better.
    size_t size;  // length of this piece
};

class _SymbolPairComparator {
   public:
    const bool operator()(_SymbolPair* h1, _SymbolPair* h2) {
        return (h1->score < h2->score || (h1->score == h2->score && h1->left > h2->left));
    }
};

struct _Symbol {
    int prev;     // prev index of this symbol. -1 for BOS.
    int next;     // next index of tihs symbol. -1 for EOS.
    bool freeze;  // this symbol is never be merged.
    std::string piece;
};

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
        using Agenda = std::priority_queue<_SymbolPair*, std::vector<_SymbolPair*>, _SymbolPairComparator>;
        Agenda agenda;
        std::vector<_Symbol> symbols;
        symbols.reserve(normalized.size());

        // Reverse merge rules.
        // key: merged symbol, value: pair of original symbols.
        std::unordered_map<std::string, std::pair<std::string, std::string>> rev_merge;

        // Pre-allocates SymbolPair for efficiency.
        constexpr size_t kPreallocateSymbolPairSize = 256;
        model::FreeList<_SymbolPair> symbol_pair_allocator(kPreallocateSymbolPairSize);

        // Lookup new symbol pair at [left, right] and inserts it to agenda.
        auto MaybeAddNewSymbolPair = [this, &symbol_pair_allocator, &symbols, &agenda, &rev_merge](int left,
                                                                                                   int right) {
            if (left == -1 || right == -1 || symbols[left].freeze || symbols[right].freeze) return;
            const std::string piece(symbols[left].piece.data(),
                                    symbols[left].piece.size() + symbols[right].piece.size());
            const auto it = pieces_.find(piece);
            if (it == pieces_.end()) {
                return;
            }
            auto* h  = symbol_pair_allocator.Allocate();
            h->left  = left;
            h->right = right;
            h->score = GetScore(it->second);
            h->size  = piece.size();
            agenda.push(h);

            // Makes `rev_merge` for resegmentation.
            if (IsUnusedInlined(it->second)) {
                rev_merge[piece] = std::make_pair(symbols[left].piece, symbols[right].piece);
            }
        };

        // Splits the input into character sequence
        int index = 0;
        while (!normalized.empty()) {
            _Symbol s;
            const int mblen = matcher_->PrefixMatch(normalized, &s.freeze);
            s.piece         = std::string(normalized.data(), mblen);
            s.prev          = index == 0 ? -1 : index - 1;
            normalized.remove_prefix(mblen);
            s.next = normalized.empty() ? -1 : index + 1;
            ++index;
            symbols.emplace_back(s);
        }

        if (symbols.empty()) {
            return {};
        }

        // Lookup all bigrams.
        for (size_t i = 1; i < symbols.size(); ++i) {
            MaybeAddNewSymbolPair(i - 1, i);
        }

        // BPE-dropout: https://arxiv.org/pdf/1910.13267.pdf
        std::mt19937* rand_gen = nullptr;
        auto skip_merge        = [&]() {
            if (alpha <= 0.0) return false;
            if (alpha >= 1.0) return true;
            if (rand_gen == nullptr) rand_gen = random::GetRandomGenerator();
            std::uniform_real_distribution<> gen(0.0, 1.0);
            return gen(*rand_gen) < alpha;
        };

        // Main loop.
        while (!agenda.empty()) {
            _SymbolPair* top = agenda.top();
            agenda.pop();

            // `top` is no longer available.
            if (symbols[top->left].piece.empty() || symbols[top->right].piece.empty() ||
                symbols[top->left].piece.size() + symbols[top->right].piece.size() != top->size) {
                continue;
            }

            // Note that orignal BPE-dropout paper assumes that all merged symbols are
            // pre computed, but here we randomly skip merge opration inside this loop.
            // This implemenation is theoretically equivalent to the original one.
            if (skip_merge()) continue;

            // Replaces symbols with `top` rule.
            symbols[top->left].piece = std::string(symbols[top->left].piece.data(),
                                                   symbols[top->left].piece.size() + symbols[top->right].piece.size());

            // Updates prev/next pointers.
            symbols[top->left].next = symbols[top->right].next;
            if (symbols[top->right].next >= 0) {
                symbols[symbols[top->right].next].prev = top->left;
            }
            symbols[top->right].piece = std::string("");

            // Adds new symbol pairs which are newly added after symbol replacement.
            MaybeAddNewSymbolPair(symbols[top->left].prev, top->left);
            MaybeAddNewSymbolPair(top->left, symbols[top->left].next);
        }

        std::function<void(std::string, EncodeResult*)> resegment;
        resegment = [this, &resegment, &rev_merge](std::string w, EncodeResult* output) -> void {
            const int id = PieceToId(w);
            if (id == -1 || !IsUnusedInlined(id)) {
                output->emplace_back(w, id);
                return;
            }
            const auto p = rev_merge.find(w);
            if (p == rev_merge.end()) {
                // This block will never be called, as `rev_merge` stores all the
                // resegmentation info for unused id.
                output->emplace_back(w, id);
                return;
            }
            // Recursively resegment left and right symbols.
            resegment(p->second.first, output);
            resegment(p->second.second, output);
        };

        EncodeResult output;
        for (int index = 0; index != -1; index = symbols[index].next) {
            CHECK_GE(index, 0);
            CHECK_LT(index, static_cast<int>(symbols.size()));
            resegment(symbols[index].piece, &output);
        }

        return output;
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
