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
#include "dregex.hpp"

namespace dregex {
struct State {
    int depth;                          // the string length
    State* failure;                     // match failed use
    std::vector<int64_t> emits;         // emits
    std::map<int64_t, State*> success;  // go map
    int64_t index;                      // index of the struct
    bool fake;                          // this node is fake
};

inline State* newState(int depth, bool fake) {
    auto S     = new State();
    S->depth   = depth;
    S->fake    = fake;
    S->failure = NULL;
    S->index   = 0;
    return S;
}

inline void freeState(State* state) {
    if (state == NULL) return;
    for (auto& kv : state->success) {
        freeState(kv.second);
    }
    state->success.clear();
    state->failure = NULL;
    state->emits.clear();
    delete state;
}

// insert sort a sort to keep it order
inline void insertSorted(std::vector<int64_t>& s, int64_t e) { s.insert(std::upper_bound(s.begin(), s.end(), e), e); }

// add emit
inline void addEmit(State* s, int64_t keyword) {
    if (keyword != INT64_MIN) insertSorted(s->emits, keyword);
}

// get l code
inline int64_t getMaxValueID(State* s) {
    if (s->emits.empty()) return INT64_MIN;
    return s->emits.back();
}

inline bool isAcceptable(State* s) { return s->depth > 0 && s->emits.size() > 0; }

inline void setFailure(State* s, State* failState, std::vector<int64_t>& fail) {
    s->failure     = failState;
    fail[s->index] = failState->index;
}

inline State* nextState(State* s, int64_t character, bool ignoreRootState) {
    auto it = s->success.find(character);
    if (it == s->success.end()) {
        if ((!ignoreRootState) && (s->depth == 0)) return s;
        return NULL;
    }
    return it->second;
}

inline State* addState(State* s, int64_t character) {
    auto nextS = nextState(s, character, true);
    if (nextS == NULL) {
        nextS                 = newState(s->depth + 1, false);
        s->success[character] = nextS;
    }
    return nextS;
}

class KvPairsIter {
   public:
    /**
     * @brief iter kv pair
     *
     * @param hit
     * @return int if sucess 0 else 1
     */
    virtual int iter(std::function<void(const StringIter&, const char** lables, size_t label_size)> hit) const = 0;
    virtual ~KvPairsIter() {}
};

// Builder is inner useed
class Builder {
   private:
    // cache labels idx
    std::map<std::string, int64_t> labels;
    // root
    State* rootState;
    Trie* trie;
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

   private:
    void zipWeight() {
        size_t msize = this->size;
        this->trie->Base.resize(msize);
        this->trie->Check.resize(msize);
    }

    void constructFailureStates() {
        this->trie->Fail.resize(this->size + 1, 0);
        this->trie->OutPut.resize(this->size + 1, NULL);
        std::queue<State*> queue;

        for (auto& kv : this->rootState->success) {
            auto depthOneState = kv.second;
            setFailure(depthOneState, this->rootState, this->trie->Fail);
            queue.push(depthOneState);

            if (!depthOneState->emits.empty()) {
                auto dt = new std::set<int64_t>(depthOneState->emits.begin(), depthOneState->emits.end());
                this->trie->OutPut[depthOneState->index] = dt;
            }
        }
        while (!queue.empty()) {
            auto currentState = queue.front();
            queue.pop();

            for (auto& kv : currentState->success) {
                auto transition  = kv.first;
                auto targetState = nextState(currentState, transition, false);
                queue.push(targetState);
                auto traceFailureState = currentState->failure;
                while (!nextState(traceFailureState, transition, false)) {
                    traceFailureState = traceFailureState->failure;
                }
                auto newFailureState = nextState(traceFailureState, transition, false);
                setFailure(targetState, newFailureState, this->trie->Fail);
                for (auto e : newFailureState->emits) {
                    addEmit(targetState, e);
                }
                auto dt = new std::set<int64_t>(targetState->emits.begin(), targetState->emits.end());
                this->trie->OutPut[targetState->index] = dt;
            }
        }
    }
    size_t addAllKeyword(const KvPairsIter& kvs, int* ret) {
        size_t maxCode = 0;
        int64_t index  = -1;

        auto t = this->trie;
        *ret   = kvs.iter([&](const StringIter& k, const char** lables, size_t label_size) {
            index++;
            size_t lens       = 0;
            auto currentState = this->rootState;
            k.walks([&](const std::string& s, size_t) {
                lens++;
                auto code     = t->getCode(s);
                t->CodeMap[s] = code;
                currentState  = addState(currentState, code);
                return false;
            });
            addEmit(currentState, index);
            t->L.push_back(lens);
            if (lens > t->MaxLen) t->MaxLen = lens;

            maxCode += lens;
            auto lables_idx = new std::set<int64_t>();
            for (size_t i = 0; i < label_size; ++i) {
                auto vit = this->labels.find(lables[i]);
                if (vit == this->labels.end()) {
                    auto idx = this->labels.size();
                    lables_idx->insert(idx);
                    this->labels[lables[i]] = idx;
                } else {
                    lables_idx->insert(vit->second);
                }
            }
            t->V.push_back(lables_idx);
        });
        t->MaxLen++;
        return size_t(double(maxCode + t->CodeMap.size()) / 1.5) + 1;
    }

    // resize data
    void resize(size_t newSize) {
        this->trie->Base.resize(newSize, 0);
        this->trie->Check.resize(newSize, 0);
        this->used.resize(newSize, false);
        this->allocSize = newSize;
    }

    std::vector<std::pair<int64_t, State*>>* fetch(State* parent) {
        auto siblings = new std::vector<std::pair<int64_t, State*>>();
        siblings->reserve(parent->success.size() + 1);
        if (isAcceptable(parent)) {
            auto fakeNode = newState(-(parent->depth + 1), true);
            addEmit(fakeNode, getMaxValueID(parent));
            siblings->push_back(std::pair<int64_t, State*>(0, fakeNode));
        }
        for (auto& kv : parent->success) {
            siblings->push_back(std::pair<int64_t, State*>(kv.first + 1, kv.second));
        }
        return siblings;
    }

    void insert(std::queue<std::pair<int64_t, std::vector<std::pair<int64_t, State*>>*>>& queue) {
        auto tCurrent = queue.front();
        queue.pop();
        auto value    = tCurrent.first;
        auto siblings = tCurrent.second;

        int64_t begin = 0, nonZeroNum = 0;
        bool first  = true;
        int64_t pos = this->nextCheckPos - 1;
        if (pos < siblings->front().first) {
            pos = siblings->front().first;
        }
        if (this->allocSize <= pos) {
            resize(pos + 1);
        }
        auto t = this->trie;
        while (true) {
            pos++;
            if (this->allocSize <= pos) {
                resize(pos + pos / 2 + 1);
            }
            if (t->Check[pos] != 0) {
                nonZeroNum++;
                continue;
            } else if (first) {
                this->nextCheckPos = pos;
                first              = false;
            }

            begin = pos - siblings->front().first;
            if (this->allocSize <= (begin + siblings->back().first)) {
                resize(begin + siblings->back().first + 100);
            }
            if (this->used[begin]) {
                continue;
            }
            auto allIszero = true;
            for (auto& kv : *siblings) {
                if (t->Check[begin + kv.first] != 0) {
                    allIszero = false;
                    break;
                }
            }
            if (allIszero) {
                break;
            }
        }

        if (double(nonZeroNum) / double(pos - this->nextCheckPos + 1) >= 0.95) {
            this->nextCheckPos = pos;
        }
        this->used[begin] = true;
        if (this->size < begin + siblings->back().first + 1) {
            this->size = begin + siblings->back().first + 1;
        }
        for (auto& kv : *siblings) {
            t->Check[begin + kv.first] = begin;
        }
        for (auto& kv : *siblings) {
            auto newSiblings = fetch(kv.second);
            if (newSiblings->empty()) {
                t->Base[begin + kv.first] = -(getMaxValueID(kv.second) + 1);
                delete newSiblings;
            } else {
                queue.push(std::pair<int64_t, std::vector<std::pair<int64_t, State*>>*>(begin + kv.first, newSiblings));
            }
            kv.second->index = begin + kv.first;
        }
        if (value >= 0) {
            t->Base[value] = begin;
        }
        // free memory
        for (auto& item : *siblings) {
            if (item.second->fake) {
                delete item.second;
                item.second = NULL;
            }
        }
        siblings->clear();
        delete siblings;
    }

   public:
    Builder(Trie& trie) : allocSize(0), nextCheckPos(0), size(0) {
        rootState  = newState(0, false);
        this->trie = &trie;
    }
    ~Builder() {
        trie = NULL;
        freeState(rootState);
        rootState = NULL;
        labels.clear();
    }

    int build(const KvPairsIter& kvs) {
        int ret;
        std::cout << "buiding code..." << std::endl;
        size_t maxCode = addAllKeyword(kvs, &ret);
        std::cout << "finish build code maxcode=" << maxCode << std::endl;
        if (ret) {
            std::cerr << "ERROR: build trie failed!" << std::endl;
            return EXIT_FAILURE;
        }
        if (maxCode < 2) {
            std::cerr << "WARN: nothing to compile!" << std::endl;
            return EXIT_SUCCESS;
        }
        // build double array tire base on tire
        std::cout << "buiding state..." << std::endl;
        resize(maxCode + 10);
        this->trie->Base[0] = 1;
        auto siblings       = fetch(this->rootState);
        if (!siblings->empty()) {
            std::queue<std::pair<int64_t, std::vector<std::pair<int64_t, State*>>*>> queue;
            queue.push(std::pair<int64_t, std::vector<std::pair<int64_t, State*>>*>(-1, siblings));
            while (!queue.empty()) insert(queue);

        } else {
            delete siblings;
        }
        std::cout << "construct failure states..." << std::endl;
        // build failure table and merge output table
        constructFailureStates();
        std::cout << "buiding darray..." << std::endl;
        zipWeight();
        // set labels
        auto& lables = this->trie->Labels;
        lables.resize(this->labels.size());
        for (auto& kv : this->labels) lables[kv.second] = kv.first;
        std::cout << "finish build darray!!" << std::endl;

        return EXIT_SUCCESS;
    }
};

/**
 * @brief complile a give trie
 *
 * @param pairs
 * @param trie
 */
inline int compile(const KvPairsIter& pairs, Trie& trie) {
    Builder builder(trie);
    return builder.build(pairs);
}

/**
 * @brief compile data into file
 *
 * @param pairs
 * @param pbfile
 * @return int
 */
inline int compile(const KvPairsIter& pairs, const std::string& pbfile) {
    Trie trie;
    if (compile(pairs, trie)) return EXIT_FAILURE;
    return trie.writePb(pbfile);
}

}  // namespace dregex
#endif  // SRC_UTILS_DCOMPILE_HPP_
