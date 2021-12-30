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

#include "dregex.hpp"

#include <map>
#include <queue>
#include <vector>


struct State {
    int depth;                           // the string length
    State *failure;                      // match failed use
    std::vector<int64_t> emits;          // emits
    std::map<int64_t, State *> success;  // go map
    int64_t index;                       // index of the struct
};

State *newState(depth int) {
    auto S = new State();
    S->depth = depth;
    return S
}

void freeState(State *state) {
    if (state == NULL) return;
    for (auto kv : state->success) {
        freeState(kv->second);
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

void setFailure(State *s, State failState, std::vector<int64_t> &fail) {
    s->failure = failState;
    fail[s->index] = failState->index;
}

State *nextState(State *s, int64_t character, bool ignoreRootState bool) {
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
    Trie *trie;
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
}


void zipWeight(Builder* b) {
    size_t msize = b->size;
    b->trie->Base.resize(msize);
    b.trie->Check.resize(msize);
}

void constructFailureStates(Builder *b) {
    b->trie->Fail.assign(b->trie->size + 1, 0);
    b->trie->OutPut.assign(b->size + 1, NULL);
    std::queue<State *> queue;


    for (auto kv : b->rootState->success) {
        auto depthOneState = kv->second;
        setFailure(depthOneState, b->rootState, b->trie.Fail);
        queue.push(depthOneState);

        if (!depthOneState->emits.empty()) {
            auto dt = new std::vector<int64_t>(depthOneState->emits.begin(), depthOneState->emits.end());
            b->trie->OutPut[depthOneState.index] = dt;
        }
    }
    while (!queue.empty()) {
        auto currentState = queue.front();
        queue.pop();
        for (auto kv : currentState->success) {
            auto transition = kv->first;
            auto targetState = nextState(currentState, transition, false);
            queue.push(targetState);
            auto traceFailureState = currentState->failure;
            while (!nextState(traceFailureState, transition, false)) {
                traceFailureState = traceFailureState->failure;
            }
            auto newFailureState = nextState(traceFailureState, transition, false);
            setFailure(targetState, newFailureState, b.trie.Fail);
            for (auto e : newFailureState->emits) {
                addEmit(targetState, e);
            }
            auto dt = new std::vector<int64_t>(targetState->emits.begin(), targetState->emits.end());
            b->trie->OutPut[targetState.index] = dt;
        }
    }
}

size_t addAllKeyword(Builder *b, StringIterPairs &kvs) {
    siez_t maxCode = 0;
    int64_t index = -1;
    Trie *t = b.trie;
    kvs.iter([&](StringIter &k, const int64_t *v, size_t vlen) {
        index++;
        size_t lens = 0;
        auto currentState = b->rootState;
        k([&](const std::string s &, size_t) {
            lens++;
            code = t->getcode(s);
            t->CodeMap[s] = code;
            currentState = addState(currentState, code);
            return false;
        });
        addEmit(currentState, index);
        t->L.push_back(lens);
        if (lens > t->MaxLen) {
            t.MaxLen = lens;
        }
        maxCode += lens;
        if (v) {
            for (size_t l = 0; l < vlen; l++) {
                t->V.push_back(v[l]);
            }
        }
    });
    t->MaxLen++;
    return size_t(float64(maxCode + len(t->CodeMap)) / 1.5) + 1;
}

// resize data
void resize(Builder *b, size_t newSize) {
    b->trie->Base.resize(newSize);
    b->trie->Check.resize(newSize);
    b->used.resize(newSize);
    b->allocSize = newSize;
}

// Pair si tmp used

void fetch(parent *State)[] Pair {
    siblings = make([] Pair, 0, parent.success.Size() + 1);
    if (isAcceptable(parent)) {
        fakeNode = newState(-(parent.depth + 1));
        fakeNode.addEmit(parent.getMaxValueIDgo());
        siblings = append(siblings, Pair{0, fakeNode});
    }
    it = parent.success.Iterator();
    for (it.Next()) {
        siblings = append(siblings, Pair{it.Key().(int)+1, it.Value()});
    }
    return siblings;
}

void insert(Builder *b, std::queue<State *> queue) {
    tCurrent = queue.Remove(queue.Front()).(Pair);
    value, siblings = tCurrent.K.(int), tCurrent.V.([] Pair);

    begin, nonZeroNum = 0, 0;
    first = true;
    pos = b.nextCheckPos - 1;
    if (pos < siblings[0].K.(int)) {
        pos = siblings[0].K.(int);
    }
    if (b.allocSize <= pos) {
        b.resize(pos + 1);
    }
    t = b.trie;
    while {
        pos++;
        if (b.allocSize <= pos) {
            b.resize(pos + 1);
        }
        if (t.Check[pos] != 0) {
            nonZeroNum++;
            continue;
        } else if (first) {
            b.nextCheckPos = pos;
            first = false;
        }

        begin = pos - siblings[0].K.(int);
        if (b.allocSize <= (begin + siblings[len(siblings) - 1].K.(int))) {
            b.resize(begin + siblings[len(siblings) - 1].K.(int)+100);
        }
        if (b.used[begin]) {
            continue;
        }
        allIszero = true;
        for (i = 0; i < len(siblings); i++) {
            if (t.Check[begin + siblings[i].K.(int)] != 0) {
                allIszero = false;
                break;
            }
        }
        if (allIszero) {
            break;
        }
    }

    if (float32(nonZeroNum) / float32(pos - b.nextCheckPos + 1) >= 0.95) {
        b.nextCheckPos = pos;
    }
    b.used[begin] = true;
    if (b.size < begin + siblings[len(siblings) - 1].K.(int)+1) {
        b.size = begin + siblings[len(siblings) - 1].K.(int)+1;
    }
    for (i = 0; i < len(siblings); i++) {
        t.Check[begin + siblings[i].K.(int)] = begin;
    }
    for (i = 0; i < len(siblings); i++) {
        kv = siblings[i];
        newSiblings = fetch(kv.V.(*State));
        if (len(newSiblings) < 1) {
            t.Base[begin + kv.K.(int)] = -(kv.V.(*State).getMaxValueIDgo() + 1);
        } else {
            queue.PushBack(Pair{begin + kv.K.(int), newSiblings});
        }
        kv.V.(*State).index = begin + kv.K.(int);
    }
    if (value >= 0) {
        t.Base[value] = begin;
    }
}

void build(Builder *b, StringIterPairs &kvs) {
    size_t maxCode = addAllKeyword(b, kvs);
    // build double array tire base on tire
    resize(b, maxCode + 10);
    b.trie.Base[0] = 1;
    siblings = fetch(b.rootState);
    if (len(siblings) > 0) {
        queue = list.New();
        queue.PushBack(Pair{-1, siblings});
        for (queue.Len() > 0) {
            b.insert(queue);
        }
    }
    // build failure table and merge output table
    constructFailureStates(b);
    zipWeight(b);
}


void compile(StringIterPairs &pairs, Trie &trie) {
    Builder builder;
    builder.rootState = newState(0);
    builder.trie = &trie;
    build(&builder, kvs);
    // clear mem
    builder.trie = NULL;
    freeState(builder.rootState);
    builder.rootState = NULL;
}
