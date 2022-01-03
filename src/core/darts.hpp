/*
 * File: Darts.h
 * Project: core
 * File Created: Saturday, 11th December 2021 6:45:52 pm
 * Author: Xu En (xuen@mokar.com)
 * -----
 * Last Modified: Saturday, 11th December 2021 8:14:04 pm
 * Modified By: Xu En (xuen@mokahr.com)
 * -----
 * Copyright 2021 - 2021 Your Company, Moka
 */
#ifndef SRC_CORE_DARTS_HPP_
#define SRC_CORE_DARTS_HPP_

#include <stdlib.h>

#include <algorithm>
#include <functional>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "../core/wtype.h"
#include "../utils/chspliter.hpp"
#include "../utils/str_utils.hpp"

namespace darts {
class Atom {
   public:
    std::string image;  // image of atom
    uint32_t st;        // start of this atom in str
    uint32_t et;        // end of this atom in str
    std::set<WordType> *tags;

    Atom(const char *image, uint32_t start, uint32_t end) {
        this->image = image;
        this->st = start;
        this->et = end;
        this->tags = NULL;
    }

    Atom(const Atom &atom) {
        this->image = atom.image;
        this->st = atom.st;
        this->et = atom.et;
        if (atom.tags != NULL) {
            this->tags = new std::set<WordType>(*atom.tags);
        }
    }

    bool hasType(WordType type) const {
        if (this->tags != NULL) {
            return this->tags->find(type) != this->tags->end();
        }
        return false;
    }

    bool hasType(std::set<WordType> *otags) const {
        if (otags == NULL || this->tags == NULL) {
            return false;
        }
        for (auto &t : *otags) {
            if (this->tags->find(t) != this->tags->end()) {
                return true;
            }
        }
        return false;
    }

    void addTag(WordType type) {
        if (this->tags == NULL) {
            this->tags = new std::set<WordType>();
        }
        this->tags->insert(type);
    }

    void addTags(std::set<WordType> *otags) {
        if (otags) {
            if (this->tags == NULL) {
                this->tags = new std::set<WordType>();
            }
            this->tags->insert(otags->begin(), otags->end());
        }
    }

    void addTags(std::vector<WordType> *otags) {
        if (otags) {
            if (this->tags == NULL) {
                this->tags = new std::set<WordType>();
            }
            this->tags->insert(otags->begin(), otags->end());
        }
    }

    void joinTags(std::set<WordType> *otags) {
        if (otags) {
            if (!this->tags) {
                this->tags = new std::set<WordType>();
                for (auto tag : *otags) {
                    this->tags->insert(tag);
                }
            } else {
                std::set<WordType> C;
                std::set_intersection(tags->begin(), tags->end(), otags->begin(), otags->end(),
                                      std::inserter(C, C.begin()));
                if (!C.empty()) {
                    this->tags->clear();
                    this->tags->insert(C.begin(), C.end());
                }
            }
        }
    }

    ~Atom() {
        if (tags != NULL) {
            tags->clear();
            delete tags;
            tags = NULL;
        }
    }

    friend std::ostream &operator<<(std::ostream &output, const Atom &D) {
        std::vector<std::string> tags;
        if (D.tags) {
            for (auto t : *D.tags) {
                tags.push_back(wordType2str(t));
            }
        }
        output << "Atom['" << D.image << "',{" << join(tags, ",") << "}]";

        return output;
    }
};

class AtomList {
   public:
    std::vector<std::shared_ptr<Atom>> data;
    std::string content;


    /**
     * @brief Construct a new Atom List object
     *
     * @param str
     */
    explicit AtomList(const std::string &str) {
        data.reserve(str.length() / 3 + 5);
        content = str;
        atomSplit(str.c_str(), [&](const char *astr, WordType ttype, size_t s, size_t e) {
            auto atom = std::make_shared<Atom>(astr, s, e);
            atom->addTag(ttype);
            this->data.push_back(atom);
        });
    }

    /**
     * @brief
     *
     */
    void clear() {
        auto iter = data.begin();
        while (iter != data.end()) {
            iter = data.erase(iter);
        }
        data.clear();
    }

    size_t size() const { return data.size(); }

    ~AtomList() { clear(); }

    /**
     * @brief Get the Atom object
     *
     * @param start
     * @param end
     * @return std::shared_ptr<Atom>
     */
    std::shared_ptr<Atom> at(size_t start, size_t end) {
        if (end > data.size()) {
            end = data.size();
        }
        if (start < 0) {
            start = 0;
        }

        if (start >= end) return nullptr;
        auto st = data[start]->st;
        auto et = data[end - 1]->et;
        auto ret = std::make_shared<Atom>("", st, et);
        for (auto i = start; i < end; ++i) {
            ret->image.append(data[i]->image);
            ret->joinTags(data[i]->tags);
        }
        return ret;
    }

    /**
     * @brief get atom object
     *
     * @param index
     * @return std::shared_ptr<Atom>
     */
    std::shared_ptr<Atom> at(size_t index) {
        if (0 <= index && index < data.size()) {
            return data[index];
        }
        return nullptr;
    }

    friend std::ostream &operator<<(std::ostream &output, const AtomList &D) {
        output << "AtomList[ ";
        for (auto a : D.data) {
            output << *a << ",";
        }
        output << "]";
        return output;
    }
};

class Word {
   public:
    std::shared_ptr<Atom> word;                // the word image
    int st;                                    // the word in atomlist start
    int et;                                    // the words in atom list end
    std::shared_ptr<std::vector<double>> att;  //
    uint16_t feat;                             // this word other type

    Word(std::shared_ptr<Atom> atom, int start, int end) {
        this->word = atom;
        this->st = start;
        this->et = end;
        this->att = nullptr;
        this->feat = 0;
    }

    Word(const Word &wd) {
        this->word = wd.word;
        this->st = wd.st;
        this->et = wd.et;
        this->feat = wd.feat;
        this->att = wd.att;
    }


    ~Word() {
        if (word) {
            word = nullptr;
        }
        if (att) {
            att = nullptr;
        }
        st = et = feat = 0;
    }


    friend std::ostream &operator<<(std::ostream &output, const Word &D) {
        if (D.word) {
            output << "Word[ " << *D.word << "]";
        } else {
            output << "Word[NULL] ";
        }
        return output;
    }

    /**
     * @brief add tags into this words
     *
     * @param tags
     */
    void addTags(std::set<WordType> *tags) { this->word->addTags(tags); }

    void addTag(WordType tag) { this->word->addTag(tag); }

    /**
     * @brief add tags into words
     *
     * @param tags
     */
    void addTags(std::vector<WordType> *tags) { this->word->addTags(tags); }
};

typedef struct _Cursor {
    struct _Cursor *prev;
    struct _Cursor *lack;
    std::shared_ptr<Word> val;
    int idx;
} * Cursor;

/**
 * @brief create a cursor
 *
 * @param word
 * @param pre
 * @param next
 * @return Cursor*
 */
Cursor makeCursor(std::shared_ptr<Word> word, Cursor pre, Cursor next) {
    Cursor cur = new struct _Cursor();
    cur->prev = pre;
    cur->lack = next;
    cur->val = word;
    cur->idx = 0;
    return cur;
}


class CellMap {
   private:
    Cursor head;
    size_t rows, colums, size;

   public:
    Cursor Head() { return this->head; }

    size_t Size() const { return this->size; }

    size_t Row() const { return this->rows; }

    size_t Column() const { return this->colums; }

    CellMap() {
        this->head = makeCursor(std::make_shared<Word>(nullptr, -1, 0), NULL, NULL);
        this->head->idx = -1;
        this->rows = this->colums = this->size = 0;
    }

    ~CellMap() {
        auto node = head;
        while (node) {
            auto next = node->lack;
            if (node->val) {
                node->val = nullptr;
            }
            delete node;
            node = next;
        }
        this->rows = this->colums = this->size = 0;
    }

    /**
     * @brief 创建原始索引
     *
     */
    void makeCurIndex() {
        auto node = this->head;
        node->idx = -1;
        int index = node->idx;
        while (node->lack) {
            node = node->lack;
            index++;
            node->idx = index;
        }
    }


    /**
     * @brief 迭代图
     *
     * @param cur
     * @param row
     * @param dfunc
     */
    void iterRow(Cursor cur, int row, std::function<void(Cursor)> dfunc) {
        if (!cur) {
            cur = this->head;
        }

        if (row < 0) {
            // Iter all if row is negtive
            while (cur->lack) {
                dfunc(cur->lack);
                cur = cur->lack;
            }
            return;
        }


        while (cur->lack) {
            // Iter the give row from start
            auto n = cur->lack;
            if (n->val->st < row) {
                cur = n;
                continue;
            }
            if (n->val->st != row) {
                break;
            }
            dfunc(n);
            cur = n;
        }
    }

    // AddNext do add the cell to give cir next
    Cursor addNext(Cursor cur, std::shared_ptr<Word> cell) {
        if (!cur) {
            cur = this->head;
        }
        if (cell->st > this->rows) {
            this->rows = cell->st;
        }
        if (cell->et > this->colums) {
            this->colums = cell->et;
        }

        while (cur->lack) {
            auto n = cur->lack;
            if (n->val->st < cell->st) {
                cur = n;
                continue;
            }
            if (n->val->st == cell->st) {
                if (n->val->et < cell->et) {
                    cur = n;
                    continue;
                }
                if (n->val->et == cell->et) {
                    n->val->addTags(cell->word->tags);
                    return n;
                }
            }
            auto m = makeCursor(cell, cur, n);
            this->size++;
            cur->lack = m;
            n->prev = m;
            return m;
        }
        cur->lack = makeCursor(cell, cur, NULL);
        this->size++;
        return cur->lack;
    }

    // AddPre add a cell to next
    Cursor addPre(Cursor cur, std::shared_ptr<Word> cell) {
        if (!cur) {
            cur = this->head;
        }
        if (cell->st > this->rows) {
            this->rows = cell->st;
        }
        if (cell->et > this->colums) {
            this->colums = cell->et;
        }
        while (cur->prev != this->head) {
            auto n = cur->prev;
            if (n->val->st > cell->st) {
                cur = n;
                continue;
            }
            if (n->val->st == cell->st) {
                if (n->val->et > cell->et) {
                    cur = n;
                    continue;
                }
                if (n->val->et == cell->et) {
                    n->val->addTags(cell->word->tags);
                    return n;
                }
            }
            auto m = makeCursor(cell, n, cur);
            this->size++;
            cur->prev = m;
            n->lack = m;
            return m;
        }
        cur->prev = makeCursor(cell, this->head, cur);
        this->size++;
        return cur->prev;
    }

    /**
     * @brief add a cell
     *
     * @param cell
     * @param cur
     * @return Cursor*
     */
    Cursor addCell(std::shared_ptr<Word> cell, Cursor cur) {
        if (!cur) {
            return addNext(this->head, cell);
        }
        if (cur->val->st < cell->st) {
            return addNext(cur, cell);
        }
        if ((cur->val->st == cell->st) && (cur->val->et <= cell->et)) {
            return addNext(cur, cell);
        }
        return addPre(cur, cell);
    }
};


}  // namespace darts
#endif  // SRC_CORE_DARTS_HPP_
