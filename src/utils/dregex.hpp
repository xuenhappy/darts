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

#include <map>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "./darts.pb.h"
#include "utf8.hpp"
namespace darts {

class StringIter {
   public:
    /**
     * @brief iter string
     *
     * @param hit
     */
    virtual void iter(std::function<bool(const std::string &, size_t)> hit) = 0;
    virtual ~StringIter() {}
};


class StringIterPairs {
    virtual void iter(std::function<void(StringIter &, int64_t)> hit) = 0;
    virtual ~StringIterPairs() {}
};

class UTF8StrIterator : public StringIter {
   private:
    const char32_t *str;
    size_t len;

   public:
    explicit UTF8StrIterator(const char32_t *str, size_t len) {
        this->str = str;
        this->len = len;
    }
    void iter(std::function<bool(const std::string &, size_t)> hit) {
        if (!this->str) return;
        std::string tmp;
        for (size_t i = 0; i < this->len; ++i) {
            tmp = unicode_to_utf8(this->str[i]);
            if (hit(tmp, i)) {
                return;
            }
        }
    }
};


class Trie {
   public:
    std::vector<int64_t> Check, Base, Fail, L;
    std::vector<std::vector<int64_t> *> V, OutPut;
    size_t MaxLen;
    std::map<std::string, int> CodeMap;

    Trie() : MaxLen(0) {}

   private:
    int64_t getcode(const std::string &word) const {
        auto it = this->CodeMap.find(word);
        if (it == this->CodeMap.end()) {
            return this->CodeMap.size() + 1;
        }
        return it->second;
    }

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
            currentState = this->Fail[currentState];
            newCurrentState = this->transitionWithRoot(currentState, character);
        }
        return newCurrentState;
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
    // ParseText parse a text list hit ( [start,end),tagidx)
    void parse(StringIter &text, std::function<bool(size_t, size_t, const std::vector<int64_t> *)> hit) {
        auto currentState = 0, indexBufferPos = 0;
        std::vector<int64_t> indexBufer;
        indexBufer.assign(this->MaxLen, 0);
        text.iter([&](const std::string &seq, size_t position) -> bool {
            indexBufer[indexBufferPos % this->MaxLen] = position;
            indexBufferPos++;
            currentState = this->getstate(currentState, this->getcode(seq));
            auto hitArray = this->OutPut[currentState];
            for (auto h : *hitArray) {
                auto preIndex = (indexBufferPos - this->L[h]) % this->MaxLen;
                if (hit(indexBufer[preIndex], position + 1, this->V[h])) {
                    return true;
                }
            }
            return false;
        });
    }
    friend std::ostream &operator<<(std::ostream &out, const Trie &my) {
        darts::DRegexDat dat;
        dat.set_maxlen(my.MaxLen);
        dat.mutable_check()->Add(my.Check.begin(), my.Check.end());
        dat.mutable_base()->Add(my.Base.begin(), my.Base.end());
        dat.mutable_fail()->Add(my.Fail.begin(), my.Fail.end());
        dat.mutable_l()->Add(my.L.begin(), my.L.end());

        for (auto s : my.V) {
            dat.add_v()->mutable_item()->Add(s->begin(), s->end());
        }
        for (auto s : my.OutPut) {
            dat.add_output()->mutable_item()->Add(s->begin(), s->end());
        }

        auto cmap = dat.mutable_codemap();
        cmap->insert(my.CodeMap.begin(), my.CodeMap.end());

        if (!dat.SerializePartialToOstream(&out)) {
            std::cerr << "Failed to write Trie" << std::endl;
        }
        return out;
    }

    friend std::istream &operator>>(std::istream &in, Trie &my) {
        darts::DRegexDat dat;
        if (!dat.ParseFromIstream(&in)) {
            std::cerr << "Failed to read Trie" << std::endl;
        }
        my.MaxLen = dat.maxlen();

        auto check = dat.check();
        my.Check.insert(my.Check.end(), check.begin(), check.end());

        auto base = dat.base();
        my.Base.insert(my.Base.end(), base.begin(), base.end());

        auto fail = dat.fail();
        my.Fail.insert(my.Fail.end(), fail.begin(), fail.end());

        auto l = dat.l();
        my.L.insert(my.L.end(), l.begin(), l.end());

        auto size = dat.v_size();
        my.V.assign(size, NULL);
        for (size_t i = 0; i < size; i++) {
            auto vlist = dat.v(i).item();
            my.V[i] = new std::vector<int64_t>(vlist.begin(), vlist.end());
        }

        size = dat.output_size();
        my.OutPut.assign(size, NULL);
        for (size_t i = 0; i < size; i++) {
            auto vlist = dat.output(i).item();
            my.OutPut[i] = new std::vector<int64_t>(vlist.begin(), vlist.end());
        }
        auto cmap = dat.codemap();
        my.CodeMap.insert(cmap.begin(), cmap.end());
        return in;
    }

    /**
     * @brief load data from a pb file
     *
     * @param path
     */
    void loadPb(const std::string &path) {
        std::ifstream f_in(path.c_str(), std::ios::in | std::ios::binary);
        if (!f_in.is_open()) {
            std::cerr << "load trie file failed:" << path << std::endl;
            return;
        }
        f_in >> *this;
        f_in.close();
    }
    /**
     * @brief write this conf into a file
     *
     * @param path
     */
    void writePb(const std::string &path) const {
        std::fstream f_out(path, std::ios::out | std::ios::trunc | std::ios::binary);
        if (!f_out.is_open()) {
            std::cerr << "write trie file failed:" << path << std::endl;
            return;
        }
        f_out << *this;
        f_out.close();
    }
};

/**
 * @brief compile a trie
 *
 * @param pairs
 * @param trie
 */
void compile(StringIterPairs &pairs, Trie &trie);

}  // namespace darts
#endif  // SRC_UTILS_DREGEX_HPP_
