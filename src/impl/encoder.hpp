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
#include <list>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include "../core/darts.hpp"
#include "../utils/biggram.hpp"

class EngWordTokenizer {
   private:
    darts::BigramDict dict;

   public:
    /**
     * @brief load freq data
     *
     * @param ddir
     * @return int
     */
    int loaddata(const std::string& ddir) { return dict.loadDict(ddir); }

    void engToken(const std::string& eng, std::vector<std::string>& ret) const {
        // token english str
        if (eng.length() < 2 || eng.length() > 50) {  // too long or short codes
            ret.push_back(eng);
            return;
        }
        // load code
    }
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
static const std::string unk_char  = "[UNK]";
static const std::string cls_char  = "[CLS]";
static const std::string sep_char  = "[SEP]";
static const std::string mask_char = "[MASK]";
static const std::string pad_char  = "[PAD]";
}  // namespace codemap

class WordPice {
   private:
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
            ret.push_back(eng);
            return;
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

#endif  //!__BPE_MODEL__H__