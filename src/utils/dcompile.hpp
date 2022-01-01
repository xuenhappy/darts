/*
 * File: dregex.cpp
 * Project: utils
 * File Created: Thursday, 30th December 2021 1:53:52 pm
 * Author: Xu En (xuen@mokar.com)
 * -----
 * Last Modified: Thursday, 30th December 2021 1:54:03 pm
 * Modified By: Xu En (xuen@mokahr.com)
 * -----
 * Copyright 2021 - 2021 Your Company, Moka
 */
#ifndef SRC_UTILS_DCOMPILE_HPP_
#define SRC_UTILS_DCOMPILE_HPP_


#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "chspliter.hpp"
#include "dregex.hpp"
#include "norm_chr.hpp"
#include "str_utils.hpp"
#include "utf8.hpp"
namespace dregex {
struct State {
    int depth;                           // the string length
    State *failure;                      // match failed use
    std::vector<int64_t> emits;          // emits
    std::map<int64_t, State *> success;  // go map
    int64_t index;                       // index of the struct
    bool fake;                           // this node is fake
};

State *newState(int depth, bool fake) {
    auto S = new State();
    S->depth = depth;
    S->fake = fake;
    S->failure = NULL;
    S->index = 0;
    return S;
}

void freeState(State *state) {
    if (state == NULL) return;
    for (auto &kv : state->success) {
        freeState(kv.second);
    }
    state->success.clear();
    state->failure = NULL;
    state->emits.clear();
    delete state;
}

// insert sort a sort to keep it order
void insertSorted(std::vector<int64_t> &s, int64_t e) { s.insert(std::upper_bound(s.begin(), s.end(), e), e); }

// add emit
void addEmit(State *s, int64_t keyword) {
    if (keyword != INT64_MIN) {
        insertSorted(s->emits, keyword);
    }
}

// get l code
int64_t getMaxValueID(State *s) {
    if (s->emits.empty()) {
        return INT64_MIN;
    }
    return s->emits.back();
}

bool isAcceptable(State *s) { return s->depth > 0 && s->emits.size() > 0; }

void setFailure(State *s, State *failState, std::vector<int64_t> &fail) {
    s->failure = failState;
    fail[s->index] = failState->index;
}

State *nextState(State *s, int64_t character, bool ignoreRootState) {
    auto it = s->success.find(character);
    if (it == s->success.end()) {
        if ((!ignoreRootState) && (s->depth == 0)) {
            return s;
        }
        return NULL;
    }
    return it->second;
}

State *addState(State *s, int64_t character) {
    auto nextS = nextState(s, character, true);
    if (nextS == NULL) {
        nextS = newState(s->depth + 1, false);
        s->success[character] = nextS;
    }
    return nextS;
}

// Builder is inner useed
struct Builder {
    State *rootState;
    darts::Trie *trie;
    /**
     * whether the position has been used
     */
    std::vector<bool> used;
    /**
     * the allocSize of the dynamic array
     */
    size_t allocSize;
    /**
     * the next position to check unused memory
     */
    int64_t nextCheckPos;
    /**
     * the size of the key-pair sets
     */
    size_t size;
};


void zipWeight(Builder *b) {
    size_t msize = b->size;
    b->trie->Base.resize(msize);
    b->trie->Check.resize(msize);
}

void constructFailureStates(Builder *b) {
    b->trie->Fail.resize(b->size + 1, 0);
    b->trie->OutPut.resize(b->size + 1, NULL);
    std::queue<State *> queue;


    for (auto &kv : b->rootState->success) {
        auto depthOneState = kv.second;
        setFailure(depthOneState, b->rootState, b->trie->Fail);
        queue.push(depthOneState);

        if (!depthOneState->emits.empty()) {
            auto dt = new std::vector<int64_t>(depthOneState->emits.begin(), depthOneState->emits.end());
            b->trie->OutPut[depthOneState->index] = dt;
        }
    }
    while (!queue.empty()) {
        auto currentState = queue.front();
        queue.pop();
        for (auto &kv : currentState->success) {
            auto transition = kv.first;
            auto targetState = nextState(currentState, transition, false);
            queue.push(targetState);
            auto traceFailureState = currentState->failure;
            while (!nextState(traceFailureState, transition, false)) {
                traceFailureState = traceFailureState->failure;
            }
            auto newFailureState = nextState(traceFailureState, transition, false);
            setFailure(targetState, newFailureState, b->trie->Fail);
            for (auto e : newFailureState->emits) {
                addEmit(targetState, e);
            }
            auto dt = new std::vector<int64_t>(targetState->emits.begin(), targetState->emits.end());
            b->trie->OutPut[targetState->index] = dt;
        }
    }
}

size_t addAllKeyword(Builder *b, darts::StringIterPairs &kvs) {
    size_t maxCode = 0;
    int64_t index = -1;
    auto t = b->trie;
    kvs.iter([&](darts::StringIter &k, const int64_t *v, size_t vlen) {
        index++;
        size_t lens = 0;
        auto currentState = b->rootState;
        k.iter([&](const std::string &s, size_t) {
            lens++;
            auto code = t->getCode(s);
            t->CodeMap[s] = code;
            currentState = addState(currentState, code);
            return false;
        });
        addEmit(currentState, index);
        t->L.push_back(lens);
        if (lens > t->MaxLen) {
            t->MaxLen = lens;
        }
        maxCode += lens;
        if (v) {
            t->V.push_back(new std::vector<int64_t>(v, v + vlen));
        } else {
            t->V.push_back(new std::vector<int64_t>(0));
        }
    });
    t->MaxLen++;
    return size_t(double(maxCode + t->CodeMap.size()) / 1.5) + 1;
}

// resize data
void resize(Builder *b, size_t newSize) {
    b->trie->Base.resize(newSize, 0);
    b->trie->Check.resize(newSize, 0);
    b->used.resize(newSize, false);
    b->allocSize = newSize;
}

// Pair si tmp used

std::vector<std::pair<int64_t, State *>> *fetch(State *parent) {
    auto siblings = new std::vector<std::pair<int64_t, State *>>();
    siblings->reserve(parent->success.size() + 1);
    if (isAcceptable(parent)) {
        auto fakeNode = newState(-(parent->depth + 1), true);
        addEmit(fakeNode, getMaxValueID(parent));
        siblings->push_back(std::pair<int64_t, State *>(0, fakeNode));
    }
    for (auto &kv : parent->success) {
        siblings->push_back(std::pair<int64_t, State *>(kv.first + 1, kv.second));
    }
    return siblings;
}

void insert(Builder *b, std::queue<std::pair<int64_t, std::vector<std::pair<int64_t, State *>> *>> &queue) {
    auto tCurrent = queue.front();
    queue.pop();
    auto value = tCurrent.first;
    auto siblings = tCurrent.second;


    int64_t begin = 0, nonZeroNum = 0;
    bool first = true;
    int64_t pos = b->nextCheckPos - 1;
    if (pos < siblings->front().first) {
        pos = siblings->front().first;
    }
    if (b->allocSize <= pos) {
        resize(b, pos + 1);
    }
    auto t = b->trie;
    while (true) {
        pos++;
        if (b->allocSize <= pos) {
            resize(b, pos + pos / 2 + 1);
        }
        if (t->Check[pos] != 0) {
            nonZeroNum++;
            continue;
        } else if (first) {
            b->nextCheckPos = pos;
            first = false;
        }

        begin = pos - siblings->front().first;
        if (b->allocSize <= (begin + siblings->back().first)) {
            resize(b, begin + siblings->back().first + 100);
        }
        if (b->used[begin]) {
            continue;
        }
        auto allIszero = true;
        for (auto &kv : *siblings) {
            if (t->Check[begin + kv.first] != 0) {
                allIszero = false;
                break;
            }
        }
        if (allIszero) {
            break;
        }
    }

    if (double(nonZeroNum) / double(pos - b->nextCheckPos + 1) >= 0.95) {
        b->nextCheckPos = pos;
    }
    b->used[begin] = true;
    if (b->size < begin + siblings->back().first + 1) {
        b->size = begin + siblings->back().first + 1;
    }
    for (auto &kv : *siblings) {
        t->Check[begin + kv.first] = begin;
    }
    for (auto &kv : *siblings) {
        auto newSiblings = fetch(kv.second);
        if (newSiblings->empty()) {
            t->Base[begin + kv.first] = -(getMaxValueID(kv.second) + 1);
            delete newSiblings;
        } else {
            queue.push(std::pair<int64_t, std::vector<std::pair<int64_t, State *>> *>(begin + kv.first, newSiblings));
        }
        kv.second->index = begin + kv.first;
    }
    if (value >= 0) {
        t->Base[value] = begin;
    }
    // free memory
    for (auto &item : *siblings) {
        if (item.second->fake) {
            delete item.second;
            item.second = NULL;
        }
    }
    siblings->clear();
    delete siblings;
}

void build(Builder *b, darts::StringIterPairs &kvs) {
    size_t maxCode = addAllKeyword(b, kvs);
    if (maxCode < 2) {
        return;
    }
    // build double array tire base on tire

    resize(b, maxCode + 10);
    b->trie->Base[0] = 1;
    auto siblings = fetch(b->rootState);
    if (!siblings->empty()) {
        std::queue<std::pair<int64_t, std::vector<std::pair<int64_t, State *>> *>> queue;
        queue.push(std::pair<int64_t, std::vector<std::pair<int64_t, State *>> *>(-1, siblings));
        while (!queue.empty()) {
            insert(b, queue);
        }
    } else {
        delete siblings;
    }
    // build failure table and merge output table
    constructFailureStates(b);
    zipWeight(b);
}

/**
 * @brief complile a give trie
 *
 * @param pairs
 * @param trie
 */
void compile(darts::StringIterPairs &pairs, darts::Trie &trie) {
    Builder builder;
    builder.size = 0;
    builder.allocSize = 0;
    builder.nextCheckPos = 0;
    builder.rootState = newState(0, false);
    builder.trie = &trie;
    build(&builder, pairs);
    // clear mem
    builder.trie = NULL;
    freeState(builder.rootState);
    builder.rootState = NULL;
}

class U32StrIterator : public darts::StringIter {
   private:
    const char32_t *str;
    size_t len;

   public:
    U32StrIterator(const char32_t *str, size_t len) {
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


class WordStrIterator : public darts::StringIter {
   private:
    const char *str;
    const std::set<WordType> *skiptypes;

   public:
    WordStrIterator(const char *str, const std::set<WordType> *skiptypes) {
        this->str = str;
        this->skiptypes = skiptypes;
    }


    void iter(std::function<bool(const std::string &, size_t)> hit) {
        if (!this->str) return;
        std::string tmp;
        int pos = -1;
        atomSplit(this->str, [&](const char *chrs, WordType ttype, size_t s, size_t e) {
            pos++;
            tmp = chrs;
            if (this->skiptypes == NULL || skiptypes->find(ttype) == skiptypes->end()) {
                hit(tmp, pos);
            }
        });
    }
};

/**
 * @brief 标准行文件迭代器
 *
 */
class FileStringPairIter : public darts::StringIterPairs {
   private:
    const std::vector<std::string> *files;
    const std::set<WordType> *skiptypes;
    std::map<std::string, int64_t> *labels;
    std::function<std::string(std::string &, std::vector<int64_t> &, std::map<std::string, int64_t> *)> lineFix;


   public:
    FileStringPairIter(
        const std::vector<std::string> *files, std::map<std::string, int64_t> *labels,
        const std::set<WordType> *skiptypes,
        std::function<std::string(std::string &, std::vector<int64_t> &, std::map<std::string, int64_t> *)> lineFix) {
        this->files = files;
        this->labels = labels;
        this->skiptypes = skiptypes;
        this->lineFix = lineFix;
        if (!lineFix) {
            this->lineFix = [&](std::string &line, std::vector<int64_t> &lidxes,
                                std::map<std::string, int64_t> *lmap) -> std::string {
                auto pos = line.find_first_of(',');
                if (pos == std::string::npos || pos < 1 || pos >= line.size() - 1) {
                    std::cerr << "WARN: bad line:[" << line << "]" << std::endl;
                    return "";
                }
                auto key = line.substr(0, pos);
                darts::trim(key);
                auto strs = line.substr(pos + 1);
                darts::trim(strs);
                if (key.empty() || strs.empty() || key.front() != '<' || key.back() != '>') {
                    std::cerr << "WARN: bad line:[" << line << "]" << std::endl;
                    return "";
                }
                darts::tolower(strs);
                key = key.substr(1, key.size() - 2);
                darts::toupper(key);
                // set keystr
                auto it = lmap->find(key);
                if (it != lmap->end()) {
                    lidxes.push_back(it->second);
                } else {
                    auto idx = labels->size();
                    (*lmap)[key] = idx;
                    lidxes.push_back(idx);
                }
                return strs;
            };
        }
    }
    virtual void iter(std::function<void(darts::StringIter &, const int64_t *, size_t)> hit) {
        std::vector<int64_t> lidxes;
        for (auto &fpath : *this->files) {
            std::ifstream f_in(fpath);
            if (!f_in.is_open()) {
                std::cerr << "ERROR: read open file error," << fpath << std::endl;
                break;
            }

            std::string line;
            while (std::getline(f_in, line)) {
                line = normalizeStr(line);
                darts::trim(line);
                if (line.empty()) {
                    continue;
                }
                lidxes.clear();
                auto key = this->lineFix(line, lidxes, this->labels);
                if (key.empty()) continue;
                WordStrIterator atom(key.c_str(), this->skiptypes);
                if (lidxes.empty()) {
                    hit(atom, NULL, 0);
                } else {
                    hit(atom, &lidxes[0], lidxes.size());
                }
            }
            f_in.close();
        }
    }
};

/**
 * @brief 编译一些词典文件
 *
 * @param paths 需要被编译的原始文件，文件内容必须符合特定规范 ，单行内容形如: "<LABEL>,清华大学"
 * @param pbfile pb文件输出的地方
 * @param skiptypes 编译过程需要跳过的一些词
 * @param lineFix 行处理函数
 */
void compileStringDict(
    const std::vector<std::string> &paths, const std::string &pbfile, const std::set<WordType> *skiptypes = NULL,
    std::function<std::string(std::string &, std::vector<int64_t> &, std::map<std::string, int64_t> *)> lineFix =
        NULL) {
    darts::Trie trie;
    std::map<std::string, int64_t> labels;
    // build trie
    FileStringPairIter pairs(&paths, &labels, skiptypes, lineFix);
    compile(pairs, trie);
    // set labels
    trie.Labels.resize(labels.size());
    for (auto &kv : labels) {
        trie.Labels[kv.second] = kv.first;
    }
    trie.writePb(pbfile);
}

}  // namespace dregex
#endif  // SRC_UTILS_DCOMPILE_HPP_
