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
#include <fstream>
#include <map>
#include <queue>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "str_utils.hpp"
#include "zipfile.hpp"

namespace darts {

class Trie {
   public:
    std::vector<std::string> Labels;
    std::vector<int64_t> Check, Base, Fail, L;
    std::vector<std::set<int64_t>*> V, OutPut;
    size_t MaxLen;
    std::map<std::string, int> CodeMap;

    Trie() : MaxLen(0) {}

   private:
    // trans
    int64_t transitionWithRoot(int64_t nodePos, int64_t c) const {
        int64_t b = 0;
        if (nodePos < this->Base.size()) {
            b = this->Base[nodePos];
        }
        auto p = b + c + 1;
        auto x = 0;
        if (p < this->Check.size()) {
            x = this->Check[p];
        }
        if (b != x) {
            if (nodePos == 0) {
                return 0;
            }
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
        dat.mutable_labels()->Add(this->Labels.begin(), this->Labels.end());
        dat.mutable_check()->Add(this->Check.begin(), this->Check.end());
        dat.mutable_base()->Add(this->Base.begin(), this->Base.end());
        dat.mutable_fail()->Add(this->Fail.begin(), this->Fail.end());
        dat.mutable_l()->Add(this->L.begin(), this->L.end());

        for (auto s : this->V) {
            if (!s) {
                dat.add_v();
                continue;
            }
            dat.add_v()->mutable_item()->Add(s->begin(), s->end());
        }

        for (auto s : this->OutPut) {
            if (!s) {
                dat.add_output();
                continue;
            }
            dat.add_output()->mutable_item()->Add(s->begin(), s->end());
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
        auto& check = dat.check();
        this->Check.insert(this->Check.end(), check.begin(), check.end());

        auto& base = dat.base();
        this->Base.insert(this->Base.end(), base.begin(), base.end());

        auto& fail = dat.fail();
        this->Fail.insert(this->Fail.end(), fail.begin(), fail.end());

        auto& l = dat.l();
        this->L.insert(this->L.end(), l.begin(), l.end());
        auto size = dat.v_size();
        this->V.assign(size, NULL);
        for (size_t i = 0; i < size; i++) {
            if (dat.v(i).item_size() < 1) {
                this->V[i] = NULL;
                continue;
            }
            auto& vlist = dat.v(i).item();
            this->V[i]  = new std::set<int64_t>(vlist.begin(), vlist.end());
        }
        size = dat.output_size();
        this->OutPut.assign(size, NULL);
        for (size_t i = 0; i < size; i++) {
            if (dat.output(i).item_size() < 1) {
                this->OutPut[i] = NULL;
                continue;
            }
            auto& vlist     = dat.output(i).item();
            this->OutPut[i] = new std::set<int64_t>(vlist.begin(), vlist.end());
        }
        auto& cmap = dat.codemap();
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
        if (label_idx < this->Labels.size()) {
            return this->Labels[label_idx].c_str();
        }
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
        if (it == this->CodeMap.end()) {
            return this->CodeMap.size() + 1;
        }
        return it->second;
    }
    /**
     * @brief match a gvie word list
     *
     * @param text src word list
     * @param hit match call back function
     */
    void parse(StringIter& text, std::function<bool(size_t, size_t, const std::set<int64_t>*)> hit) const {
        if (this->MaxLen < 1 || this->V.empty()) {
            std::cerr << "ERROR: parse on empty trie!" << std::endl;
            return;
        }
        auto currentState = 0, indexBufferPos = 0;
        std::vector<int64_t> indexBufer;
        indexBufer.assign(this->MaxLen, 0);
        text.iter([&](const std::string& seq, size_t position) -> bool {
            indexBufer[indexBufferPos % this->MaxLen] = position;
            indexBufferPos++;
            currentState = this->getstate(currentState, this->getCode(seq));
            if (currentState >= this->OutPut.size()) return false;
            auto hitArray = this->OutPut[currentState];
            if (!hitArray) return false;
            for (auto h : *hitArray) {
                auto preIndex = (indexBufferPos - this->L[h]) % this->MaxLen;
                if (hit(indexBufer[preIndex], position + 1, this->V[h])) {
                    return true;
                }
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
        std::ostream* zstream = zipf.Add_File("dregex.pb");
        if (!zstream) {
            std::cerr << "ERROE: write trie file failed:" << path << std::endl;
            return EXIT_FAILURE;
        }
        auto ret = this->writePb(*zstream);
        delete zstream;
        return ret;
    }
};

}  // namespace darts
#endif  // SRC_UTILS_DREGEX_HPP_
