/*
 * File: biggram.hpp
 * Project: utils
 * File Created: Sunday, 20th November 2022 3:27:45 pm
 * Author: dell (xuen@mokar.com)
 * -----
 * Last Modified: Sunday, 20th November 2022 3:31:20 pm
 * Modified By: dell (xuen@mokahr.com)
 * -----
 * Copyright 2021 - 2022 Your Company, Moka
 */
#ifndef SRC_UTILS_BIGGRAM_HPP_
#define SRC_UTILS_BIGGRAM_HPP_

#pragma once
#include <darts.pb.h>
#include <math.h>
#include <algorithm>
#include <map>
#include <string>
#include <unordered_map>
#include "cedar.h"
#include "str_utils.hpp"
#include "zipfile.hpp"

namespace darts {

struct bigram_key {
    size_t i, j;  // words - indexes of the words in a dictionary
    // a constructor to be easily constructible
    bigram_key(size_t a_i, size_t a_j) : i(a_i), j(a_j) {}

    // you need to sort keys to be used in a map container
    bool operator<(bigram_key const& other) const { return i < other.i || (i == other.i && j < other.j); }

    bool operator==(const bigram_key& rhs) { return i == rhs.i && j == rhs.j; }
};
struct bigram_data {
    size_t count;  // n(ij)
    bigram_data() : count(0) {}
    explicit bigram_data(size_t n_ij) : count(n_ij) {}
};

class BigramDict {
   private:
    static const char* KEY_IDX_FILE;
    static const char* TABLE_FILE;

    cedar::da<int> idx;
    std::map<bigram_key, bigram_data> bigrams;
    std::unordered_map<size_t, size_t> freqs;

    size_t avg_single_freq = 0;
    size_t max_single_freq = 0;
    size_t avg_union_freq  = 0;

    /**
     * @brief Get the Single Nlog Prop object
     * 获取某个字独立出现的词频负对数
     * @param word
     * @return double
     */
    double getSingleNlogProp(const std::string& word) const {
        int widx = getWordKey(word);
        if (widx < 0) {
            return log((1000.0 + max_single_freq) / (1.0 + avg_single_freq / 2));
        }
        return log((1000.0 + max_single_freq) / (1.0 + freqs.at(widx)));
    }

    /**
     * @brief Get the Nlog Prop object
     * 获取两个词联合规律负对数
     * @param word
     * @param next
     * @return double
     */
    double getNlogProp(const std::string& word, const std::string& next) const {
        double a = 0.0, b = 0.0, n_ij = 0.0;
        int a_widx = getWordKey(word), b_widx = getWordKey(next);
        if (a_widx < 0 || b_widx < 0) {
            if (a_widx < 0) {
                a = avg_single_freq / 2;
            }
            if (b_widx < 0) {
                b = avg_single_freq / 2;
            }
            n_ij = std::min(std::min(a, b), double(avg_union_freq)) / 2.0;
        } else {
            a = freqs.at(a_widx), b = freqs.at(b_widx);
            bigram_key key(a_widx, b_widx);
            auto it = bigrams.find(key);
            if (it == bigrams.end()) {
                n_ij = std::min(std::min(a, b), double(avg_union_freq)) / 2.0;
            } else {
                double m = std::min(a, b);
                n_ij     = std::min(m, double(it->second.count)) * 0.85 + 0.15 * m;
            }
        }
        return log((1.0 + a) / (1.0 + n_ij));
    }

    /**
     * @brief read table file
     *
     * @param ifile
     * @return int
     */
    int readTable(const std::string& ifile) {
        zipfile::ZipFileReader zfile(ifile);
        std::istream* zipIn = zfile.Get_File("table.pb");
        if (!zipIn) {
            std::cerr << "ERROR: load table file failed:" << ifile << std::endl;
            return EXIT_FAILURE;
        }

        darts::BigramDat dat;
        if (!dat.ParseFromIstream(zipIn)) {
            std::cerr << "ERROR: Failed to read table file:" << ifile << std::endl;
            delete zipIn;
            return EXIT_FAILURE;
        }
        delete zipIn;

        this->avg_single_freq = dat.avg_single_freq();
        this->avg_union_freq  = dat.avg_union_freq();
        this->max_single_freq = dat.max_single_freq();

        auto freq = dat.freq();
        this->freqs.insert(freq.begin(), freq.end());

        auto size = dat.table_size();
        for (size_t i = 0; i < size; i++) {
            auto tuple                                      = dat.table(i);
            this->bigrams[bigram_key(tuple.x(), tuple.y())] = bigram_data(tuple.freq());
        }
        return EXIT_SUCCESS;
    }

    /**
     * @brief write table file
     *
     * @param outfile
     * @return int
     */
    int writeTable(const std::string& outfile) const {
        std::fstream f_out(outfile, std::ios::out | std::ios::trunc | std::ios::binary);
        if (!f_out.is_open()) {
            std::cerr << "ERROE: write table file failed:" << outfile << std::endl;
            return EXIT_FAILURE;
        }
        darts::BigramDat dat;
        dat.set_avg_single_freq(this->avg_single_freq);
        dat.set_avg_union_freq(this->avg_union_freq);
        dat.set_max_single_freq(this->max_single_freq);

        auto freq = dat.mutable_freq();
        freq->insert(this->freqs.begin(), this->freqs.end());
        for (auto& kv : this->bigrams) {
            auto table = dat.add_table();
            table->set_x(kv.first.i);
            table->set_y(kv.first.j);
            table->set_freq(kv.second.count);
        }

        zipfile::ZipFileWriter zfile(outfile);
        std::ostream* zipOut = zfile.Add_File("table.pb");
        if (!dat.SerializePartialToOstream(zipOut)) {
            std::cerr << "ERROR: Failed to write table file: " << outfile << std::endl;
            delete zipOut;
            return EXIT_FAILURE;
        }
        delete zipOut;
        return EXIT_SUCCESS;
    }

   public:
    /**
     * @brief Get the Word Freq object
     *
     * @param word
     * @return size_t
     */
    int getWordKey(const std::string& word) const {
        auto widx = idx.exactMatchSearch<int>(word.c_str(), word.length());
        if (widx == cedar::da<int>::CEDAR_NO_VALUE) {
            return -1;
        }
        return widx;
    }

    /**
     * @brief
     *
     * @param single_freq_dict 单个词频的数据
     * @param union_freq_dict   联合词频数据
     * @param outdir 输出目录
     * @return int
     */
    static int buildDict(const std::string& single_freq_dict, const std::string& union_freq_dict,
                         const std::string& outdir) {
        // create dir
        if (createDirectory(outdir)) {
            std::cerr << "ERROR:create out dir error: " << outdir << std::endl;
            return EXIT_FAILURE;
        }
        BigramDict biggram_dict;
        // get output file
        namespace fs = std::filesystem;
        fs::path widx_path(outdir);
        widx_path.append(KEY_IDX_FILE);
        fs::path table_path(outdir);
        table_path.append(TABLE_FILE);

        // read freq data
        std::ifstream idx_in(single_freq_dict.c_str());
        if (!idx_in.is_open()) {
            std::cerr << "ERROR: open data " << single_freq_dict << " file failed " << std::endl;
            return EXIT_FAILURE;
        }
        uint64_t freq_sum = 0, max_freq = 0, union_freq_sum = 0;
        std::string line;
        size_t n = 0;
        std::vector<std::string> coloums;
        while (std::getline(idx_in, line)) {
            darts::trim(line);
            if (line.empty()) {
                continue;
            }
            coloums.clear();
            split(line, "\t", coloums);
            if (coloums.size() < 2) {
                std::cerr << "WARN:bad line for freq [" << line << "] coloum size:" << coloums.size() << std::endl;
                continue;
            }
            std::string word = trim(coloums[0]);
            size_t freq      = atol(coloums[1].c_str());
            if (freq < 1 || word.empty()) {
                std::cerr << "WARN:bad line freq for freq: " << line << std::endl;
                continue;
            }
            freq_sum += freq;
            if (freq > max_freq) max_freq = freq;
            biggram_dict.idx.update(word.c_str(), word.length(), n++);
            biggram_dict.freqs[n - 1] = freq;
        }
        idx_in.close();

        // read table file
        std::vector<std::string> tmpkey;
        std::ifstream table_in(union_freq_dict.c_str());
        if (!table_in.is_open()) {
            std::cerr << "ERROR: open data " << union_freq_dict << " file failed " << std::endl;
            return EXIT_FAILURE;
        }
        size_t unum = 0;
        while (std::getline(table_in, line)) {
            darts::trim(line);
            if (line.empty()) {
                continue;
            }
            coloums.clear();
            split(line, "\t", coloums);
            if (coloums.size() != 2) {
                std::cerr << "WARN:bad line for table [" << line << "] coloum size:" << coloums.size() << std::endl;
                continue;
            }
            std::string word = coloums[0];
            size_t freq      = atol(coloums[1].c_str());
            if (freq < 1) {
                std::cerr << "WARN:bad line freq for table: " << line << std::endl;
                continue;
            }
            tmpkey.clear();
            split(word, "-", tmpkey);
            if (tmpkey.size() != 2) {
                std::cerr << "WARN:bad line words for table: " << line << std::endl;
                continue;
            }
            trim(tmpkey[0]);
            trim(tmpkey[1]);
            if (tmpkey[0].empty() || tmpkey[1].empty()) {
                std::cerr << "WARN:bad line words for table: " << line << std::endl;
                continue;
            }
            auto pidx = biggram_dict.getWordKey(tmpkey[0]);
            auto nidx = biggram_dict.getWordKey(tmpkey[1]);
            if (pidx < 0 || nidx < 0) {
                std::cerr << "WARN:bad line words for table,words not in freq txt: "
                          << "key1:" << tmpkey[0] << "," << pidx << " key2:" << tmpkey[1] << "," << nidx << "|" << line
                          << std::endl;
                continue;
            }
            biggram_dict.bigrams[bigram_key(pidx, nidx)] = bigram_data(freq);
            union_freq_sum += freq;
            unum++;
        }
        table_in.close();
        // set static info
        biggram_dict.avg_single_freq = freq_sum / (1 + n);
        biggram_dict.avg_union_freq  = union_freq_sum / (1 + unum);
        biggram_dict.max_single_freq = max_freq;

        // save data
        if (biggram_dict.idx.save(widx_path.string().c_str())) {
            std::cerr << "ERROR: cannot save trie: " << widx_path.string() << std::endl;
            return EXIT_FAILURE;
        }
        return biggram_dict.writeTable(table_path.string());
    }

    /**
     * @brief load the dictionary
     *
     * @param dictionary_dir
     * @return int
     */
    int loadDict(const std::string& dictionary_dir) {
        namespace fs = std::filesystem;
        fs::path widx_path(dictionary_dir);
        widx_path.append(KEY_IDX_FILE);
        fs::path table_path(dictionary_dir);
        table_path.append(TABLE_FILE);
        if (idx.open(widx_path.string().c_str())) {
            std::cerr << "ERROR:cannot open word idx: " << widx_path.string() << std::endl;
            return EXIT_FAILURE;
        }
        return this->readTable(table_path.string());
    }

    double wordDist(const char* pre, const char* next) const {
        if (pre == NULL || next == NULL) {
            if (pre) {
                return getSingleNlogProp(pre);
            }
            if (next) {
                return getSingleNlogProp(next);
            }
            return 0.0;
        }
        return getNlogProp(pre, next);
    }

    ~BigramDict() {
        idx.clear();
        bigrams.clear();
        freqs.clear();
    }
};
const char* BigramDict::KEY_IDX_FILE = "words.da";
const char* BigramDict::TABLE_FILE   = "table.pb.gz";
}  // namespace darts

#endif  // SRC_UTILS_BIGGRAM_HPP_
