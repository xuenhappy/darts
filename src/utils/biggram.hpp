/*
 * File: biggram.hpp
 * Project: utils
 * File Created: Sunday, 20th November 2022 3:27:45 pm
 * Author: dell (nanhangxuen@163.com)
 * -----
 * Last Modified: Sunday, 20th November 2022 3:31:20 pm
 * Modified By: dell (nanhangxuen@163.com)
 * -----
 * Copyright 2021 - 2022 XuEn
 */
#ifndef SRC_UTILS_BIGGRAM_HPP_
#define SRC_UTILS_BIGGRAM_HPP_

#include <cstdlib>
#pragma once
#include <darts.pb.h>
#include <math.h>
#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>
#include "./cedar.hpp"
#include "./codecvt.hpp"
#include "./filetool.hpp"
#include "./strtool.hpp"
#include "./zipfile.hpp"

namespace darts {

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
    std::unordered_map<uint64_t, bigram_data> bigrams;
    std::vector<size_t> freqs;
    std::vector<std::string> words;
    std::vector<uint32_t> word_first_char;
    std::vector<uint32_t> word_last_char;
    std::vector<uint64_t> history_total;
    std::vector<uint32_t> history_types;
    std::vector<uint32_t> continuation_types;
    std::unordered_map<uint32_t, uint64_t> char_freqs;
    std::unordered_map<uint64_t, uint64_t> char_bigrams;
    std::unordered_map<uint32_t, uint64_t> char_history_total;
    std::unordered_map<uint32_t, uint32_t> char_history_types;
    std::unordered_map<uint32_t, uint32_t> char_continuation_types;

    size_t avg_single_freq = 0;
    size_t max_single_freq = 0;
    size_t avg_union_freq  = 0;
    uint64_t total_unigram_freq = 0;
    uint64_t total_bigram_types = 0;
    uint64_t total_char_freq = 0;
    uint64_t total_char_bigram_types = 0;
    double discount = 0.5;
    double char_discount = 0.5;
    bool smoothed = false;

    static uint64_t makeBigramKey(uint32_t first, uint32_t second) {
        return (static_cast<uint64_t>(first) << 32) | second;
    }

    static double safeNll(double probability) {
        return -std::log(std::max(probability, 1e-15));
    }

    double unigramProbability(int word_idx) const {
        const double denominator = static_cast<double>(total_unigram_freq) + freqs.size();
        if (word_idx < 0 || static_cast<size_t>(word_idx) >= freqs.size() || denominator <= 0.0)
            return 1.0 / std::max(1.0, denominator);
        return (freqs[word_idx] + 1.0) / denominator;
    }

    double charContinuation(uint32_t codepoint) const {
        auto it = char_continuation_types.find(codepoint);
        if (it == char_continuation_types.end() || total_char_bigram_types == 0)
            return 1.0 / std::max<size_t>(1, char_freqs.size());
        return static_cast<double>(it->second) / total_char_bigram_types;
    }

    double charConditional(uint32_t left, uint32_t right) const {
        auto total = char_history_total.find(left);
        if (total == char_history_total.end() || total->second == 0) return charContinuation(right);
        const auto count = char_bigrams.find(makeBigramKey(left, right));
        const double observed = count == char_bigrams.end() ? 0.0 :
            std::max(0.0, static_cast<double>(count->second) - char_discount) / total->second;
        const auto types = char_history_types.find(left);
        const double gamma = char_discount * (types == char_history_types.end() ? 0 : types->second) / total->second;
        return observed + gamma * charContinuation(right);
    }

    static bool isAsciiLetter(uint32_t codepoint) {
        return (codepoint >= 'a' && codepoint <= 'z') || (codepoint >= 'A' && codepoint <= 'Z');
    }

    double oovProbability(const std::string& word) const {
        if (word.empty()) return 1e-15;
        const auto chars = to_utf32(word);
        if (chars.empty()) return 1e-15;
        bool all_digits = true;
        bool all_letters = true;
        for (uint32_t codepoint : chars) {
            all_digits = all_digits && codepoint >= '0' && codepoint <= '9';
            all_letters = all_letters && isAsciiLetter(codepoint);
        }
        if (all_digits) {
            // Numeric values are open-class; model their length rather than identity.
            return std::max(1e-15, 0.02 * std::pow(0.55, static_cast<double>(chars.size() - 1)));
        }
        double probability = charContinuation(chars.front());
        for (size_t i = 1; i < chars.size(); ++i) probability *= charConditional(chars[i - 1], chars[i]);
        const double length_decay = all_letters ? 0.92 : 0.80;
        probability *= std::pow(length_decay, static_cast<double>(chars.size() - 1));
        return std::max(probability, 1e-15);
    }

    double continuationProbability(int word_idx, const std::string* word = nullptr) const {
        if (word_idx >= 0 && static_cast<size_t>(word_idx) < continuation_types.size() &&
            total_bigram_types > 0 && continuation_types[word_idx] > 0)
            return static_cast<double>(continuation_types[word_idx]) / total_bigram_types;
        return word ? oovProbability(*word) : unigramProbability(word_idx);
    }

    double knProbability(int left, int right, const std::string* left_word,
                         const std::string* right_word) const {
        if (left < 0) return right >= 0 ? unigramProbability(right) :
            (right_word ? oovProbability(*right_word) : unigramProbability(-1));
        if (static_cast<size_t>(left) >= history_total.size() || history_total[left] == 0)
            return right >= 0 ? unigramProbability(right) :
                (right_word ? oovProbability(*right_word) : unigramProbability(-1));

        const auto found = right >= 0 ? bigrams.find(makeBigramKey(left, right)) : bigrams.end();
        const double observed = found == bigrams.end() ? 0.0 :
            std::max(0.0, static_cast<double>(found->second.count) - discount) / history_total[left];
        double backoff = continuationProbability(right, right_word);
        if (found == bigrams.end()) {
            uint32_t left_boundary = 0, right_boundary = 0;
            if (left >= 0 && static_cast<size_t>(left) < word_last_char.size()) {
                left_boundary = word_last_char[left];
            } else if (left_word && !left_word->empty()) {
                const auto chars = to_utf32(*left_word);
                if (!chars.empty()) left_boundary = chars.back();
            }
            if (right >= 0 && static_cast<size_t>(right) < word_first_char.size()) {
                right_boundary = word_first_char[right];
            } else if (right_word && !right_word->empty()) {
                const auto chars = to_utf32(*right_word);
                if (!chars.empty()) right_boundary = chars.front();
            }
            if (left_boundary && right_boundary)
                backoff = 0.8 * backoff + 0.2 * charConditional(left_boundary, right_boundary);
        }
        const double gamma = discount * history_types[left] / history_total[left];
        return observed + gamma * backoff;
    }

    double getSingleNlogProp(int widx) const {
        if (smoothed) return safeNll(unigramProbability(widx));
        if (widx < 0) return log((1000.0 + max_single_freq) / (1.0 + avg_single_freq * 1.0 / 2));
        return log((1000.0 + max_single_freq) / (1.0 + freqs.at(widx)));
    }

    /**
     * @brief Get the Single Nlog Prop object
     * 获取某个字独立出现的词频负对数
     * @param word
     * @return double
     */
    double getSingleNlogProp(const std::string& word) const {
        const int key = getWordKey(word);
        if (smoothed && key < 0) return safeNll(oovProbability(word));
        return getSingleNlogProp(key);
    }
    double getNlogProp(int a_widx, int b_widx) const {
        if (smoothed) return safeNll(knProbability(a_widx, b_widx, nullptr, nullptr));
        double a = 0.0, b = 0.0, n_ij = 0.0;
        if (a_widx < 0 || b_widx < 0) {
            if (a_widx < 0) {
                a = avg_single_freq * 1.0 / 2;
            }
            if (b_widx < 0) {
                b = avg_single_freq * 1.0 / 2;
            }
            n_ij = std::min(std::min(a, b), double(avg_union_freq)) / 2.0;
        } else {
            a = freqs.at(a_widx), b = freqs.at(b_widx);
            uint64_t key = makeBigramKey(a_widx, b_widx);
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
     * @brief Get the Nlog Prop object
     * 获取两个词联合规律负对数
     * @param word
     * @param next
     * @return double
     */
    double getNlogProp(const std::string& word, const std::string& next) const {
        int a_widx = getWordKey(word), b_widx = getWordKey(next);
        if (smoothed) return safeNll(knProbability(a_widx, b_widx, &word, &next));
        return getNlogProp(a_widx, b_widx);
    }

    /**
     * @brief read table file
     *
     * @param ifile
     * @return int
     */
    int readTable(std::istream& is) {
        darts::BigramDat dat;
        if (!dat.ParseFromIstream(&is)) return EXIT_FAILURE;
        this->avg_single_freq = dat.avg_single_freq();
        this->avg_union_freq  = dat.avg_union_freq();
        this->max_single_freq = dat.max_single_freq();
        smoothed = dat.format_version() >= 2 && dat.history_total_size() > 0;

        const auto& freq = dat.freq();
        size_t max_index = 0;
        for (const auto& entry : freq) max_index = std::max(max_index, static_cast<size_t>(entry.first));
        this->freqs.assign(freq.empty() ? 0 : max_index + 1, 0);
        for (const auto& entry : freq) this->freqs[entry.first] = entry.second;
        words.assign(dat.words().begin(), dat.words().end());
        word_first_char.reserve(words.size());
        word_last_char.reserve(words.size());
        for (const auto& word : words) {
            const auto chars = to_utf32(word);
            word_first_char.push_back(chars.empty() ? 0 : chars.front());
            word_last_char.push_back(chars.empty() ? 0 : chars.back());
        }
        history_total.assign(dat.history_total().begin(), dat.history_total().end());
        history_types.assign(dat.history_types().begin(), dat.history_types().end());
        continuation_types.assign(dat.continuation_types().begin(), dat.continuation_types().end());
        discount = dat.discount() > 0.0 ? dat.discount() : 0.5;
        total_unigram_freq = dat.total_unigram_freq();
        total_bigram_types = dat.total_bigram_types();
        char_discount = dat.char_discount() > 0.0 ? dat.char_discount() : 0.5;
        total_char_freq = dat.total_char_freq();
        total_char_bigram_types = dat.total_char_bigram_types();
        for (const auto& item : dat.chars()) {
            char_freqs[item.codepoint()] = item.freq();
            char_history_types[item.codepoint()] = item.history_types();
            char_continuation_types[item.codepoint()] = item.continuation_types();
        }
        for (const auto& item : dat.char_table()) {
            char_bigrams[makeBigramKey(item.x(), item.y())] = item.freq();
            char_history_total[item.x()] += item.freq();
        }

        auto size = dat.table_size();
        this->bigrams.reserve(size);
        for (size_t i = 0; i < size; i++) {
            auto tuple                                      = dat.table(i);
            this->bigrams[makeBigramKey(tuple.x(), tuple.y())] = bigram_data(tuple.freq());
        }
        return EXIT_SUCCESS;
    }

    /**
     * @brief write table file
     *
     * @param outfile
     * @return int
     */
    int writeTable(std::ostream& os) const {
        darts::BigramDat dat;
        dat.set_avg_single_freq(this->avg_single_freq);
        dat.set_avg_union_freq(this->avg_union_freq);
        dat.set_max_single_freq(this->max_single_freq);
        dat.set_format_version(2);
        dat.set_discount(discount);
        dat.set_total_unigram_freq(total_unigram_freq);
        dat.set_total_bigram_types(total_bigram_types);
        dat.set_char_discount(char_discount);
        dat.set_total_char_freq(total_char_freq);
        dat.set_total_char_bigram_types(total_char_bigram_types);
        for (const auto& word : words) dat.add_words(word);
        for (auto value : history_total) dat.add_history_total(value);
        for (auto value : history_types) dat.add_history_types(value);
        for (auto value : continuation_types) dat.add_continuation_types(value);
        for (const auto& item : char_freqs) {
            auto* value = dat.add_chars();
            value->set_codepoint(item.first);
            value->set_freq(item.second);
            value->set_history_types(char_history_types.count(item.first) ? char_history_types.at(item.first) : 0);
            value->set_continuation_types(
                char_continuation_types.count(item.first) ? char_continuation_types.at(item.first) : 0);
        }
        for (const auto& item : char_bigrams) {
            auto* value = dat.add_char_table();
            value->set_x(item.first >> 32);
            value->set_y(item.first & 0xffffffffu);
            value->set_freq(item.second);
        }

        auto freq = dat.mutable_freq();
        for (size_t i = 0; i < this->freqs.size(); ++i) (*freq)[i] = this->freqs[i];
        for (auto& kv : this->bigrams) {
            auto table = dat.add_table();
            table->set_x(kv.first >> 32);
            table->set_y(kv.first & 0xffffffffu);
            table->set_freq(kv.second.count);
        }
        if (!dat.SerializePartialToOstream(&os)) return EXIT_FAILURE;
        os.flush();
        return EXIT_SUCCESS;
    }

   public:
    BigramDict() {}
    /**
     * @brief
     *
     * @param word
     * @param hit (start,len,idx)
     */
    void matchKey(const std::string& word, std::function<void(int, int, int)> hit) const {
        const char* key = word.c_str();
        size_t len      = word.size();

        for (size_t s = 0; s < len; ++s) {
            size_t didx = 0;
            for (size_t pos = s; pos < len;) {
                auto widx = idx.traverse(key, didx, pos, pos + 1);
                if (widx == cedar::da<int>::CEDAR_NO_VALUE) continue;
                if (widx == cedar::da<int>::CEDAR_NO_PATH) break;
                hit(s, pos - s, widx);
            }
        }
    }

    /**
     * @brief Get the Word Freq object
     *
     * @param word
     * @return size_t
     */
    int getWordKey(const std::string& word) const {
        auto widx = idx.exactMatchSearch<int>(word.c_str(), word.length());
        return widx == cedar::da<int>::CEDAR_NO_VALUE ? -1 : widx;
    }

    /**
     * @brief load the dictionary
     *
     * @param dictionary_dir
     * @return int
     */
    int loadDict(const std::string& dict_file) {
        zipfile::ZipFileReader zfile(dict_file);
        std::istream* keyidx_zipIn = zfile.Get_File(KEY_IDX_FILE);
        if (!keyidx_zipIn) {
            std::cerr << "ERROR:load word idx file failed!" << std::endl;
            return EXIT_FAILURE;
        }
        if (idx.open(*keyidx_zipIn)) {
            delete keyidx_zipIn;
            std::cerr << "ERROR:read word idx dict failed! " << std::endl;
            return EXIT_FAILURE;
        }
        delete keyidx_zipIn;
        std::istream* table_zipIn = zfile.Get_File(TABLE_FILE);
        if (!table_zipIn) {
            std::cerr << "ERROR:load table file failed:" << KEY_IDX_FILE << std::endl;
            return EXIT_FAILURE;
        }
        if (readTable(*table_zipIn)) {
            std::cerr << "ERROR:read table file failed:" << KEY_IDX_FILE << std::endl;
            delete table_zipIn;
            return EXIT_FAILURE;
        }
        delete table_zipIn;
        return EXIT_SUCCESS;
    }

    /**
     * @brief
     *
     * @param single_freq_dict 单个词频的数据
     * @param union_freq_dict   联合词频数据
     * @return int
     */
    int loadDictFromTxt(const std::string& single_freq_dict, const std::string& union_freq_dict) {
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
            this->idx.update(word.c_str(), word.length(), n++);
            this->freqs.push_back(freq);
            this->words.push_back(word);
            total_unigram_freq += freq;
            const auto chars = to_utf32(word);
            word_first_char.push_back(chars.empty() ? 0 : chars.front());
            word_last_char.push_back(chars.empty() ? 0 : chars.back());
            for (uint32_t codepoint : chars) {
                char_freqs[codepoint] += freq;
                total_char_freq += freq;
            }
            for (size_t i = 1; i < chars.size(); ++i)
                char_bigrams[makeBigramKey(chars[i - 1], chars[i])] += freq;
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
            auto pidx = this->getWordKey(tmpkey[0]);
            auto nidx = this->getWordKey(tmpkey[1]);
            if (pidx < 0 || nidx < 0) {
                std::cerr << "WARN:bad line words for table,words not in freq txt: "
                          << "key1:" << tmpkey[0] << "," << pidx << " key2:" << tmpkey[1] << "," << nidx << "|" << line
                          << std::endl;
                continue;
            }
            this->bigrams[makeBigramKey(pidx, nidx)] = bigram_data(freq);
            union_freq_sum += freq;
            unum++;
        }
        table_in.close();
        history_total.assign(freqs.size(), 0);
        history_types.assign(freqs.size(), 0);
        continuation_types.assign(freqs.size(), 0);
        size_t count_of_one = 0, count_of_two = 0;
        for (const auto& item : bigrams) {
            const uint32_t left = item.first >> 32;
            const uint32_t right = item.first & 0xffffffffu;
            history_total[left] += item.second.count;
            history_types[left]++;
            continuation_types[right]++;
            if (item.second.count == 1) count_of_one++;
            else if (item.second.count == 2) count_of_two++;
        }
        total_bigram_types = bigrams.size();
        discount = count_of_one + 2 * count_of_two > 0 ?
            static_cast<double>(count_of_one) / (count_of_one + 2.0 * count_of_two) : 0.5;
        size_t char_count_one = 0, char_count_two = 0;
        for (const auto& item : char_bigrams) {
            const uint32_t left = item.first >> 32;
            const uint32_t right = item.first & 0xffffffffu;
            char_history_total[left] += item.second;
            char_history_types[left]++;
            char_continuation_types[right]++;
            if (item.second == 1) char_count_one++;
            else if (item.second == 2) char_count_two++;
        }
        total_char_bigram_types = char_bigrams.size();
        char_discount = char_count_one + 2 * char_count_two > 0 ?
            static_cast<double>(char_count_one) / (char_count_one + 2.0 * char_count_two) : 0.5;
        smoothed = true;
        // set static info
        this->avg_single_freq = freq_sum / (1 + n);
        this->avg_union_freq  = union_freq_sum / (1 + unum);
        this->max_single_freq = max_freq;
        return EXIT_SUCCESS;
    }

    int saveDict(const std::string& outfile) {
        // create file
        zipfile::ZipFileWriter zfile(outfile);
        std::ostream* keyidx_zipOut = zfile.Add_File(KEY_IDX_FILE);
        if (!keyidx_zipOut) {
            std::cout << "create idx zipfile failed!" << std::endl;
            return EXIT_FAILURE;
        }
        if (idx.save(*keyidx_zipOut)) {
            std::cout << "save idx failed" << std::endl;
            delete keyidx_zipOut;
            return EXIT_FAILURE;
        }
        delete keyidx_zipOut;

        std::ostream* table_zipOut = zfile.Add_File(TABLE_FILE);
        if (!table_zipOut) {
            std::cout << "craete table failed!" << std::endl;
            return EXIT_FAILURE;
        }
        if (writeTable(*table_zipOut)) {
            std::cout << "write table failed!" << std::endl;
            delete table_zipOut;
            return EXIT_FAILURE;
        }
        delete table_zipOut;
        return EXIT_SUCCESS;
    }

    double wordDist(const char* pre, const char* next) const {
        if (pre == nullptr || next == nullptr) {
            if (pre) return getSingleNlogProp(pre);
            if (next) return getSingleNlogProp(next);
            return 0.0;
        }
        return getNlogProp(pre, next);
    }

    double wordDist(const std::string* pre, const std::string* next) const {
        if (!pre || !next) {
            if (pre) return getSingleNlogProp(*pre);
            if (next) return getSingleNlogProp(*next);
            return 0.0;
        }
        return getNlogProp(*pre, *next);
    }

    double wordDist(int pre_idx, int next_idx) const {
        if (pre_idx < 0 || next_idx < 0) {
            if (pre_idx >= 0) return getSingleNlogProp(pre_idx);
            if (next_idx >= 0) return getSingleNlogProp(next_idx);
            return getSingleNlogProp(-1);
        }
        return getNlogProp(pre_idx, next_idx);
    }

    double wordDist(int pre_idx, int next_idx, const std::string* pre, const std::string* next) const {
        if (!smoothed) return wordDist(pre_idx, next_idx);
        if (pre_idx < 0 && pre == nullptr) {
            if (next_idx < 0 && next == nullptr) return 0.0;
            return safeNll(next_idx >= 0 ? unigramProbability(next_idx) : oovProbability(*next));
        }
        if (next_idx < 0 && next == nullptr) {
            return safeNll(pre_idx >= 0 ? unigramProbability(pre_idx) : oovProbability(*pre));
        }
        return safeNll(knProbability(pre_idx, next_idx, pre, next));
    }

    ~BigramDict() {
        idx.clear();
        bigrams.clear();
        freqs.clear();
        words.clear();
        word_first_char.clear();
        word_last_char.clear();
        history_total.clear();
        history_types.clear();
        continuation_types.clear();
        char_freqs.clear();
        char_bigrams.clear();
    }
};
const char* BigramDict::KEY_IDX_FILE = "words.da";
const char* BigramDict::TABLE_FILE   = "table.pb";
}  // namespace darts

#endif  // SRC_UTILS_BIGGRAM_HPP_
