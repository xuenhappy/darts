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
#ifndef __DCOMPILE__H__
#define __DCOMPILE__H__


#include <iostream>
#include <map>
#include <queue>
#include <sstream>
#include <vector>

#include "dregex.hpp"


struct State {
    int depth;                           // the string length
    State *failure;                      // match failed use
    std::vector<int64_t> emits;          // emits
    std::map<int64_t, State *> success;  // go map
    int64_t index;                       // index of the struct
};

State *newState(int depth) {
    auto S = new State();
    S->depth = depth;
    return S;
}

void freeState(State *state) {
    if (state == NULL) return;
    for (auto kv : state->success) {
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
int64_t getMaxValueIDgo(State *s) {
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
    if ((!ignoreRootState) && (it == s->success.end()) && (s->depth == 0)) {
        return s;
    }
    if (it == s->success.end()) {
        return NULL;
    }
    return it->second;
}

State *addState(State *s, int64_t character) {
    auto nextS = nextState(s, character, true);
    if (nextS == NULL) {
        nextS = newState(s->depth + 1);
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
    b->trie->Fail.assign(b->size + 1, 0);
    b->trie->OutPut.assign(b->size + 1, NULL);
    std::queue<State *> queue;


    for (auto kv : b->rootState->success) {
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
        for (auto kv : currentState->success) {
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
            auto code = t->getcode(s);
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
    b->trie->Base.resize(newSize);
    b->trie->Check.resize(newSize);
    b->used.resize(newSize);
    b->allocSize = newSize;
}

// Pair si tmp used

std::vector<std::pair<int64_t, State *>> *fetch(State *parent) {
    auto siblings = new std::vector<std::pair<int64_t, State *>>();
    siblings->reserve(parent->success.size() + 1);
    if (isAcceptable(parent)) {
        auto fakeNode = newState(-(parent->depth + 1));
        addEmit(fakeNode, getMaxValueIDgo(parent));
        siblings->push_back(std::make_pair<>(0, fakeNode));
    }
    for (auto kv : parent->success) {
        siblings->push_back(std::make_pair<>(kv.first + 1, kv.second));
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
            resize(b, pos + 1);
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
        for (auto kv : *siblings) {
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
    for (auto kv : *siblings) {
        t->Check[begin + kv.first] = begin;
    }
    for (auto kv : *siblings) {
        auto newSiblings = fetch(kv.second);
        if (newSiblings->empty()) {
            t->Base[begin + kv.first] = -(getMaxValueIDgo(kv.second) + 1);
            delete newSiblings;
        } else {
            queue.push(std::make_pair<>(begin + kv.first, newSiblings));
        }
        kv.second->index = begin + kv.first;
    }
    if (value >= 0) {
        t->Base[value] = begin;
    }
    siblings->clear();
    delete siblings;
}

void build(Builder *b, darts::StringIterPairs &kvs) {
    size_t maxCode = addAllKeyword(b, kvs);
    // build double array tire base on tire
    resize(b, maxCode + 10);
    b->trie->Base[0] = 1;
    auto siblings = fetch(b->rootState);
    if (!siblings->empty()) {
        std::queue<std::pair<int64_t, std::vector<std::pair<int64_t, State *>> *>> queue;
        queue.push(std::make_pair<>(-1, siblings));
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


void compile(darts::StringIterPairs &pairs, darts::Trie &trie) {
    Builder builder;
    builder.rootState = newState(0);
    builder.trie = &trie;
    build(&builder, pairs);
    // clear mem
    builder.trie = NULL;
    freeState(builder.rootState);
    builder.rootState = NULL;
}


#endif  //!__DCOMPILE__H__