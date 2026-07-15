/**
 * @file dregex.hpp
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2021-12-25
 *
 * @copyright Copyright (c) 2021
 *
 */

#ifndef SRC_UTILS_DREGEX_HPP_
#define SRC_UTILS_DREGEX_HPP_

#include <darts.pb.h>
#include <cctype>
#include <fstream>
#include <map>
#include <queue>
#include <string>
#include <utility>
#include <unordered_map>
#include <vector>
#include "zipfile.hpp"

namespace dregex {

class StringIter {
   public:
    virtual void walks(std::function<bool(const std::string& iterm, size_t postion)> hit) const = 0;
    virtual ~StringIter() {}
};

class Trie {
   public:
    std::vector<std::string> Labels;
    std::vector<int64_t> Check, Base, Fail, L;
    std::vector<std::vector<int64_t>*> V, OutPut;
    size_t MaxLen;
    std::unordered_map<std::string, int> CodeMap;

    Trie() : MaxLen(0) {}

   private:
    // trans
    int64_t transitionWithRoot(int64_t nodePos, int64_t c) const {
        int64_t b = 0;
        if (nodePos < this->Base.size()) b = this->Base[nodePos];
        auto p = b + c + 1;
        auto x = 0;
        if (p < this->Check.size()) x = this->Check[p];
        if (b != x) {
            if (nodePos == 0) return 0;
            return -1;
        }
        return p;
    }

    // get state transpose
    int64_t getstate(int64_t currentState, int64_t character) const {
        auto newCurrentState = this->transitionWithRoot(currentState, character);
        while (newCurrentState == -1) {
            currentState    = this->Fail[currentState];
            newCurrentState = this->transitionWithRoot(currentState, character);
        }
        return newCurrentState;
    }

    int writePb(std::ostream& out) const {
        darts::DRegexDat dat;
        dat.set_maxlen(this->MaxLen);
        dat.set_formatversion(2);
        dat.mutable_labels()->Add(this->Labels.begin(), this->Labels.end());
        int64_t previous = 0;
        for (auto value : this->Check) {
            dat.add_checkcompact(static_cast<int32_t>(value - previous));
            previous = value;
        }
        previous = 0;
        for (auto value : this->Base) {
            dat.add_basecompact(static_cast<int32_t>(value - previous));
            previous = value;
        }
        previous = 0;
        for (auto value : this->Fail) {
            dat.add_failcompact(static_cast<int32_t>(value - previous));
            previous = value;
        }
        for (auto value : this->L) dat.add_lengthcompact(static_cast<uint32_t>(value));

        for (const auto* values : this->V) {
            auto* item = dat.add_v();
            if (values) item->mutable_item()->Add(values->begin(), values->end());
        }
        for (const auto* outputs : this->OutPut) {
            auto* item = dat.add_output();
            if (outputs) item->mutable_item()->Add(outputs->begin(), outputs->end());
        }

        auto cmap = dat.mutable_codemap();
        cmap->insert(this->CodeMap.begin(), this->CodeMap.end());

        if (!dat.SerializePartialToOstream(&out)) {
            std::cerr << "ERROR: Failed to write Trie" << std::endl;
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }

    int loadPb(std::istream& in) {
        darts::DRegexDat dat;
        if (!dat.ParseFromIstream(&in)) {
            std::cerr << "ERROR: Failed to read Trie" << std::endl;
            return EXIT_FAILURE;
        }
        this->MaxLen = dat.maxlen();
        auto& labels = dat.labels();
        this->Labels.insert(this->Labels.end(), labels.begin(), labels.end());
        if (dat.checkcompact_size() > 0) {
            int64_t value = 0;
            this->Check.reserve(dat.checkcompact_size());
            for (auto delta : dat.checkcompact()) {
                value += delta;
                this->Check.push_back(value);
            }
            value = 0;
            this->Base.reserve(dat.basecompact_size());
            for (auto delta : dat.basecompact()) {
                value += delta;
                this->Base.push_back(value);
            }
            value = 0;
            this->Fail.reserve(dat.failcompact_size());
            for (auto delta : dat.failcompact()) {
                value += delta;
                this->Fail.push_back(value);
            }
            this->L.assign(dat.lengthcompact().begin(), dat.lengthcompact().end());

            this->V.assign(dat.v_size(), nullptr);
            for (size_t i = 0; i < this->V.size(); ++i) {
                const auto& values = dat.v(i).item();
                if (!values.empty()) this->V[i] = new std::vector<int64_t>(values.begin(), values.end());
            }
            this->OutPut.assign(dat.output_size(), nullptr);
            for (size_t i = 0; i < this->OutPut.size(); ++i) {
                const auto& outputs = dat.output(i).item();
                if (!outputs.empty()) this->OutPut[i] = new std::vector<int64_t>(outputs.begin(), outputs.end());
            }
        } else {
            this->Check.assign(dat.check().begin(), dat.check().end());
            this->Base.assign(dat.base().begin(), dat.base().end());
            this->Fail.assign(dat.fail().begin(), dat.fail().end());
            this->L.assign(dat.l().begin(), dat.l().end());
            this->V.assign(dat.v_size(), nullptr);
            for (size_t i = 0; i < this->V.size(); ++i) {
                const auto& values = dat.v(i).item();
                if (!values.empty()) this->V[i] = new std::vector<int64_t>(values.begin(), values.end());
            }
            this->OutPut.assign(dat.output_size(), nullptr);
            for (size_t i = 0; i < this->OutPut.size(); ++i) {
                const auto& outputs = dat.output(i).item();
                if (!outputs.empty()) this->OutPut[i] = new std::vector<int64_t>(outputs.begin(), outputs.end());
            }
        }
        auto& cmap = dat.codemap();
        this->CodeMap.reserve(cmap.size());
        this->CodeMap.insert(cmap.begin(), cmap.end());
        return EXIT_SUCCESS;
    }

   public:
    ~Trie() {
        this->Check.clear();
        this->Base.clear();
        this->Fail.clear();
        this->L.clear();
        this->CodeMap.clear();
        for (auto e : this->V) {
            delete e;
        }
        for (auto e : this->OutPut) {
            delete e;
        }
        this->V.clear();
        this->OutPut.clear();
    }
    /**
     * @brief Get the Label object
     *
     * @param label_idx
     * @return const std::string&
     */
    const char* getLabel(size_t label_idx) const {
        if (label_idx < this->Labels.size()) return this->Labels[label_idx].c_str();
        return "";
    }
    /**
     * @brief Get the ori word code in this trie
     *
     * @param word
     * @return int64_t
     */
    int64_t getCode(const std::string& word) const {
        auto it = this->CodeMap.find(word);
        if (it == this->CodeMap.end()) return this->CodeMap.size() + 1;
        return it->second;
    }

    int64_t getCodeCaseFolded(const std::string& word) const {
        bool has_upper = false;
        for (unsigned char value : word) {
            if (value >= 'A' && value <= 'Z') {
                has_upper = true;
                break;
            }
        }
        if (!has_upper) return getCode(word);
        std::string folded(word);
        for (char& value : folded) {
            auto byte = static_cast<unsigned char>(value);
            if (byte >= 'A' && byte <= 'Z') value = static_cast<char>(byte + ('a' - 'A'));
        }
        return getCode(folded);
    }

    template <typename TokenAt, typename Hit>
    void parseContiguous(size_t count, TokenAt&& token_at, Hit&& hit) const {
        if (this->MaxLen < 1 || this->V.empty()) return;
        int64_t current_state = 0;
        for (size_t position = 0; position < count; ++position) {
            current_state = getstate(current_state, getCodeCaseFolded(token_at(position)));
            if (current_state < 0 || static_cast<size_t>(current_state) >= this->OutPut.size()) continue;
            const auto* outputs = this->OutPut[current_state];
            if (!outputs) continue;
            for (auto keyword : *outputs) {
                if (keyword < 0 || static_cast<size_t>(keyword) >= this->L.size() ||
                    static_cast<size_t>(keyword) >= this->V.size())
                    continue;
                const auto length = static_cast<size_t>(this->L[keyword]);
                if (length <= position + 1 && hit(position + 1 - length, position + 1, this->V[keyword])) return;
            }
        }
    }
    /**
     * @brief match a gvie word list
     *
     * @param text src word list
     * @param hit match call back function
     */
    void parse(StringIter& text, std::function<bool(size_t, size_t, const std::vector<int64_t>*)> hit) const {
        if (this->MaxLen < 1 || this->V.empty()) {
            std::cerr << "ERROR: parse on empty trie!" << std::endl;
            return;
        }
        auto currentState = 0, indexBufferPos = 0;
        std::vector<int64_t> indexBufer;
        indexBufer.assign(this->MaxLen, 0);
        text.walks([&](const std::string& seq, size_t position) -> bool {
            indexBufer[indexBufferPos % this->MaxLen] = position;
            indexBufferPos++;
            currentState = this->getstate(currentState, this->getCode(seq));
            if (currentState >= this->OutPut.size()) return false;
            auto hitArray = this->OutPut[currentState];
            if (!hitArray) return false;
            for (auto h : *hitArray) {
                auto preIndex = (indexBufferPos - this->L[h]) % this->MaxLen;
                if (hit(indexBufer[preIndex], position + 1, this->V[h])) return true;
            }
            return false;
        });
    }

    /**
     * @brief load data from a pb file
     *
     * @param path
     */
    int loadPb(const std::string& path) {
        zipfile::ZipFileReader zipf(path);
        std::istream* zstream = zipf.Get_File("dregex.pb");
        if (!zstream) {
            std::cerr << "ERROE: read trie file failed:" << path << std::endl;
            return EXIT_FAILURE;
        }
        auto ret = this->loadPb(*zstream);
        delete zstream;
        return ret;
    }
    /**
     * @brief write this conf into a file
     *
     * @param path
     */
    int writePb(const std::string& path) const {
        zipfile::ZipFileWriter zipf(path);
        std::ostream* zstream = zipf.Add_File("dregex.pb", true);
        if (!zstream) {
            std::cerr << "ERROE: write trie file failed:" << path << std::endl;
            return EXIT_FAILURE;
        }
        auto ret = this->writePb(*zstream);
        delete zstream;
        return ret;
    }
};

}  // namespace dregex
#endif  // SRC_UTILS_DREGEX_HPP_
