/*
 * File: quantizer.hpp
 * Project: impl
 * File Created: Saturday, 25th December 2021 11:05:37 pm
 * Author: Xu En (xuen@mokar.com)
 * -----
 * Last Modified: Saturday, 1st January 2022 7:40:19 pm
 * Modified By: Xu En (xuen@mokahr.com)
 * -----
 * Copyright 2021 - 2022 Your Company, Moka
 */
#ifndef SRC_IMPL_QUANTIZER_HPP_
#define SRC_IMPL_QUANTIZER_HPP_
#include <darts.pb.h>
#include <math.h>

#include <algorithm>
#include <filesystem>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "../core/segment.hpp"
#include "../utils/cedar.h"
#include "../utils/file_utils.hpp"
#include "../utils/str_utils.hpp"

namespace darts {
class MinCoverPersenter {
   public:
    int initalize(const std::map<std::string, std::string> &param) { return EXIT_SUCCESS; }
    void embed(AtomList *dstSrc, CellMap *cmap) {}
    /**
     * @brief
     *
     * @param pre
     * @param next
     * @return double must >=0
     */
    double ranging(const Word *pre, const Word *next) const {
        double len_a = (pre == NULL) ? 0.0 : 100.0 / (1.0 + pre->word->image.length());
        double len_b = (next == NULL) ? 0.0 : 100.0 / (1.0 + next->word->image.length());
        return len_a + len_b;
    }
    ~MinCoverPersenter() {}
};

REGISTER_Persenter(MinCoverPersenter);

struct bigram_key {
    size_t i, j;  // words - indexes of the words in a dictionary
    // a constructor to be easily constructible
    bigram_key(size_t a_i, size_t a_j) : i(a_i), j(a_j) {}

    // you need to sort keys to be used in a map container
    bool operator<(bigram_key const &other) const { return i < other.i || (i == other.i && j < other.j); }

    bool operator==(const bigram_key &rhs) { return i == rhs.i && j == rhs.j; }
};
struct bigram_data {
    size_t count;  // n(ij)
};

/**
 * @brief this use cedar store thing
 *
 */
class BigramPersenter {
   private:
    static const char *DAT_DIR_KEY;
    static const char *KEY_IDX_FILE;
    static const char *TABLE_FILE;
    cedar::da<int> idx;
    std::map<bigram_key, bigram_data> bigrams;
    std::map<size_t, size_t> freqs;
    size_t avg_single_freq = 0;
    size_t max_single_freq = 0;
    size_t avg_union_freq = 0;


    /**
     * @brief Get the Word Freq object
     *
     * @param word
     * @return size_t
     */
    int getWordKey(const std::string &word) const {
        auto widx = idx.exactMatchSearch<int>(word.c_str(), word.length());
        if (widx != cedar::da<int>::CEDAR_NO_VALUE) {
            return -1;
        }
        return widx;
    }


    /**
     * @brief Get the Single Nlog Prop object
     * 获取某个字独立出现的词频负对数
     * @param word
     * @return double
     */
    double getSingleNlogProp(const std::string &word) const {
        int widx = getWordKey(word);
        if (widx < 0) {
            return log((1000.0 + max_single_freq) / (1.0 + avg_single_freq / 2));
        }
        return log((1000.0 + max_single_freq) / (1.0 + freqs[widx]));
    }

    /**
     * @brief Get the Nlog Prop object
     * 获取两个词联合规律负对数
     * @param word
     * @param next
     * @return double
     */
    double getNlogProp(const std::string &word, const std::string &next) const {
        double a = 0.0, b = 0.0, n_ij = 0.0;
        int a_widx = getWordKey(word), b_widx = getWordKey(next);
        if (a_widx < 0 || b_widx < 0) {
            if (a_widx < 0) {
                a = avg_single_freq / 2;
            }
            if (b_widx < 0) {
                b = avg_single_freq / 2;
            }
            n_ij = min(a, b, avg_union_freq) / 2.0;
        } else {
            a = freqs[a_widx], b = freqs[b_widx];
            bigram_key key(a_widx, b_widx);
            auto it = bigrams.find(key);
            if (it == bigrams.end()) {
                n_ij = min(a, b, avg_union_freq) / 2.0;
            } else {
                n_ij = it->second.count * 0.85 + 0.15 * min(a, b);
            }
        }
        return log((1.0 + a) / (1.0 + n_ij));
    }


   public:
    int initalize(const std::map<std::string, std::string> &param) {
        auto it = param.find(DAT_DIR_KEY);
        if (it == param.end() || it->second == NULL) {
            std::cerr << "ERROR: could not find key:" << DAT_DIR_KEY << std::endl;
            return EXIT_FAILURE;
        }
        std::string path = getResource(it->second, true);
        return loadDict(path);
    }
    /**
     * @brief do nothing
     *
     * @param dstSrc
     * @param cmap
     */
    void embed(AtomList *dstSrc, CellMap *cmap) const {}

    static int readTable(BigramPersenter &dat, const std::string &outfile) {
        // idx.update(line, std::strlen(line) - 1, n++);
        // idx.save("");
        // protobuf save the dict
        return EXIT_SUCCESS;
    }

    static int writeTable(const BigramPersenter &dat, const std::string &outfile) {
        // idx.update(line, std::strlen(line) - 1, n++);
        // idx.save("");
        // protobuf save the dict
        return EXIT_SUCCESS;
    }

    /**
     * @brief
     *
     * @param single_freq_dict 单个词频的数据
     * @param union_freq_dict   联合词频数据
     * @param outdir 输出目录
     * @return int
     */
    static int buildDict(const std::string &single_freq_dict, const std::string &union_freq_dict,
                         const std::string &outdir) {
        // get output file
        namespace fs = fs::filesystem;
        fs::path widx_path(outdir);
        widx_path.append(KEY_IDX_FILE);
        fs::path table_path(outdir);
        table_path.append(TABLE_FILE);

        // read data
        std::ifstream idx_in(single_freq_dict.c_str());
        if (!idx_in.is_open()) {
            std::cerr << "ERROR: open data " << dat << " file failed " << std::endl;
            return EXIT_FAILURE;
        }
        std::string line;
        size_t n = 0;
        while (std::getline(idx_in, line)) {
            darts::trim(line);
            if (line.empty()) {
                continue;
            }
            idx.update(line, line.length(), n++);
        }


        // save data
        if (idx.save(widx_path.string())) {
            std::cerr << "ERROR: cannot save trie: " << widx_path.string() << std::endl;
            return EXIT_FAILURE;
        }
        return writeTable(table_path.string());
    }


    /**
     * @brief load the dictionary
     *
     * @param dictionary_dir
     * @return int
     */
    int loadDict(const std::string &dictionary_dir) {
        namespace fs = fs::filesystem;
        fs::path widx_path(dictionary_dir);
        widx_path.append(KEY_IDX_FILE);
        fs::path table_path(dictionary_dir);
        table_path.append(TABLE_FILE);
        if (idx.open(widx_path.string())) {
            std::cerr << "cannot open word idx: " << widx_path.string() << std::endl;
            return EXIT_FAILURE;
        }
        return readTable(*this, table_path.string());
    }
    /**
     * @brief
     *
     * @param pre
     * @param next
     * @return double must >=0
     */
    double ranging(const Word *pre, const Word *next) const {
        if (pre == NULL || next == NULL) {
            if (pre) {
                return getSingleNlogProp(pre->word->image);
            }
            if (next) {
                return getSingleNlogProp(next->word->image);
            }
            return 0.0;
        }
        return getNlogProp(pre->word->image, next->word->image);
    }

    ~BigramPersenter() {
        idx.close();
        bigrams.clear();
        freqs.clear();
    }
};
const char *BigramPersenter::KEY_IDX_FILE = "keyidx.da";
const char *BigramPersenter::TABLE_FILE = "table.pb";
const char *BigramPersenter::DAT_DIR_KEY = "dat.dir";
REGISTER_Persenter(BigramPersenter);

class ElmoPersenter {
   private:
    static const char *MODEL_PATH_KEY;
    static const char *CHAR_TABLE_FILE_KEY;

   public:
    /**
     * @brief init this
     *
     * @param param
     * @return int
     */
    int initalize(const std::map<std::string, std::string> &param) { return EXIT_SUCCESS; }
    /**
     * @brief set all word embeding
     *
     * @param dstSrc
     * @param cmap
     */
    void embed(AtomList *dstSrc, CellMap *cmap) const {}
    /**
     * @brief
     *
     * @param pre
     * @param next
     * @return double must >=0
     */
    double ranging(const Word *pre, const Word *next) const { return 0.0; }
    ~ElmoPersenter() {}
};

const char *ElmoPersenter::MODEL_PATH_KEY = "model.path";
REGISTER_Persenter(ElmoPersenter);


class TinyBertPersenter {
   private:
    static const char *MODEL_PATH_KEY;

   public:
    /**
     * @brief init this
     *
     * @param param
     * @return int
     */
    int initalize(const std::map<std::string, std::string> &param) { return EXIT_SUCCESS; }
    /**
     * @brief set all word embeding
     *
     * @param dstSrc
     * @param cmap
     */
    void embed(AtomList *dstSrc, CellMap *cmap) const {}
    /**
     * @brief
     *
     * @param pre
     * @param next
     * @return double must >=0
     */
    double ranging(const Word *pre, const Word *next) const { return 0.0; }
    ~TinyBertPersenter() {}
};

const char *TinyBertPersenter::MODEL_PATH_KEY = "model.path";
REGISTER_Persenter(TinyBertPersenter);


}  // namespace darts

#endif  // SRC_IMPL_QUANTIZER_HPP_
