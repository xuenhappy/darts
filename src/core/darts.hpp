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

#include <assert.h>
#include <stdlib.h>
#include <algorithm>
#include <functional>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "../utils/chspliter.hpp"
#include "../utils/codecvt.hpp"
#include "../utils/str_utils.hpp"

namespace darts {
class Atom {
   public:
    std::string image;  // image of atom
    uint32_t st;        // start of this atom in str
    uint32_t et;        // end of this atom in str
    std::set<std::string>* labels;

    Atom(const char* image, uint32_t start, uint32_t end) {
        this->image  = image;
        this->st     = start;
        this->et     = end;
        this->labels = NULL;
    }

    Atom(const Atom& atom) {
        this->image = atom.image;
        this->st    = atom.st;
        this->et    = atom.et;
        if (atom.labels) {
            this->labels = new std::set<std::string>(*atom.labels);
        }
    }

    bool hasLabel(std::string label) const {
        if (!labels) return false;
        return labels->find(label) != labels->end();
    }

    bool hasLabel(std::set<std::string>* labels) const {
        if (labels == NULL || this->labels == NULL) {
            return false;
        }
        for (auto& t : *labels) {
            if (this->labels->find(t) != this->labels->end()) {
                return true;
            }
        }
        return false;
    }

    void addLabel(std::string& label) {
        if (!this->labels) {
            this->labels = new std::set<std::string>();
        }
        this->labels->insert(label);
    }

    void addLabels(std::set<std::string>* otags) {
        if (otags) {
            if (!this->labels) {
                this->labels = new std::set<std::string>();
            }
            this->labels->insert(otags->begin(), otags->end());
        }
    }

    void addLabels(std::vector<std::string>* otags) {
        if (otags) {
            if (!this->labels) {
                this->labels = new std::set<std::string>();
            }
            this->labels->insert(otags->begin(), otags->end());
        }
    }

    void joinLabel(std::set<std::string>* otags) {
        if (!otags) return;
        if (!labels || labels->empty()) {
            this->labels = new std::set<std::string>(*otags);
            return;
        }
        for (auto it = labels->begin(); it != labels->end();) {
            if (otags->find(*it) == otags->end()) {
                it = labels->erase(it);
                continue;
            }
            ++it;
        }
    }

    ~Atom() {
        if (labels != NULL) {
            labels->clear();
            delete labels;
            labels = NULL;
        }
    }

    friend std::ostream& operator<<(std::ostream& output, const Atom& D) {
        std::string tags = D.labels == NULL ? "" : join(*D.labels, ",");
        output << "Atom['" << D.image << "':{" << tags << "}]";
        return output;
    }
};

class AtomList {
   public:
    std::vector<std::shared_ptr<Atom>> data;
    std::u32string str;

    /**
     * @brief Construct a new Atom List object
     *
     * @param str
     */
    explicit AtomList(const std::string& str) {
        this->str = to_utf32(str);
        data.reserve(str.length());
        atomSplit(this->str, [&](const char* astr, std::string& ttype, size_t s, size_t e) {
            auto atom = std::make_shared<Atom>(astr, s, e);
            atom->addLabel(ttype);
            this->data.push_back(atom);
        });
    }

    /**
     * @brief Construct a new Atom List object sub
     *
     * @param other
     * @param s
     * @param e
     */
    AtomList(const AtomList& other, int s = 0, int e = -1) {
        if (e < 0) {
            e = other.str.size();
        }
        if (s <= e) {
            return;
        }
        this->str = other.str.substr(0, e - s);
        this->data.insert(data.end(), other.data.begin() + s, other.data.begin() + e);
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
    std::shared_ptr<Atom> at(size_t start, size_t end) const {
        assert(start >= 0 && end < data.size());
        if (start >= end) return nullptr;
        auto st           = data[start]->st;
        auto et           = data[end - 1]->et;
        std::string image = to_utf8(str.substr(st, et - st));
        auto atom         = std::make_shared<Atom>(image.c_str(), st, et);
        for (auto i = start; i < end; ++i) {
            atom->joinLabel(data[i]->labels);
        }
        return atom;
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

    friend std::ostream& operator<<(std::ostream& output, const AtomList& D) {
        output << "AtomList[ ";
        for (auto a : D.data) {
            output << *a << ",";
        }
        output << "]";
        return output;
    }
    // support the for each
    std::vector<std::shared_ptr<Atom>>::const_iterator begin() const { return data.begin(); }
    std::vector<std::shared_ptr<Atom>>::const_iterator end() const { return data.end(); }
};

class Word {
   private:
    void* att;                                  // att adata
    std::function<void(void*)> att_clean_func;  // att clean function

   public:
    std::shared_ptr<Atom> word;  // the word image
    int st;                      // the word in atomlist start
    int et;                      // the words in atom list end
    uint16_t feat;               // this word other type

    /**
     * @brief Construct a new Word object
     *
     * @param atom
     * @param start  start of postion in atom list
     * @param end  end of postion in atom list
     */
    Word(std::shared_ptr<Atom> atom, int start, int end) {
        this->word = atom;
        this->st   = start;
        this->et   = end;
        this->att  = nullptr;
        this->feat = 0;
    }

    Word(const Word& wd) {
        this->word = wd.word;
        this->st   = wd.st;
        this->et   = wd.et;
        this->feat = wd.feat;
        this->att  = wd.att;
    }

    ~Word() {
        if (word) {
            word = nullptr;
        }
        if (att) {
            if (att_clean_func != nullptr) {
                att_clean_func(att);
            }
            att = NULL;
        }
        st = et = feat = 0;
    }
    bool operator==(const Word& D) { return D.st == st && D.et == et && D.feat == feat; }

    friend std::ostream& operator<<(std::ostream& output, const Word& D) {
        if (D.word) {
            output << "Word[ " << *D.word << "]";
        } else {
            output << "Word[NULL] ";
        }
        return output;
    }

    /**
     * @brief Set the Att object
     *
     * @param att
     * @param clean_func
     */
    void setAtt(void* att, std::function<void(void*)> clean_func) {
        if (this->att != NULL && this->att_clean_func != nullptr) {
            this->att_clean_func(this->att);
        }
        this->att            = att;
        this->att_clean_func = clean_func;
    }

    /**
     * @brief Get the Att object
     *
     * @return void*
     */
    void* getAtt() { return this->att; }

    /**
     * @brief add tags into this words
     *
     * @param tags
     */
    void addLabels(std::set<std::string>* tags) {
        if (word != nullptr) word->addLabels(tags);
    }

    void addLabel(std::string& tag) {
        if (word != nullptr) word->addLabel(tag);
    }

    /**
     * @brief add tags into words
     *
     * @param tags
     */
    void addLabels(std::vector<std::string>* tags) {
        if (word != nullptr) word->addLabels(tags);
    }
};

typedef struct _Cursor {
    struct _Cursor* prev;
    struct _Cursor* lack;
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
    cur->prev  = pre;
    cur->lack  = next;
    cur->val   = word;
    cur->idx   = 0;
    return cur;
}

class SegPath {
   private:
    Cursor head;
    size_t rows, colums, size;

   public:
    Cursor Head() { return this->head; }

    size_t Size() const { return this->size; }

    size_t Row() const { return this->rows; }

    size_t Column() const { return this->colums; }

    SegPath() {
        this->head      = makeCursor(std::make_shared<Word>(nullptr, -1, 0), NULL, NULL);
        this->head->idx = -1;
        this->rows = this->colums = this->size = 0;
    }

    ~SegPath() {
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
            auto n    = cur->lack;
            auto nval = n->val;
            if (nval->st < cell->st) {
                cur = n;
                continue;
            }
            if (nval->st == cell->st) {
                if (nval->et < cell->et) {
                    cur = n;
                    continue;
                }
                if (nval->et == cell->et) {
                    if (*nval == *cell) {
                        nval->addLabels(cell->word->labels);
                        return n;
                    }
                    cur = n;
                    continue;
                }
            }
            auto m = makeCursor(cell, cur, n);
            this->size++;
            cur->lack = m;
            n->prev   = m;
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
            auto n    = cur->prev;
            auto nval = n->val;
            if (nval->st > cell->st) {
                cur = n;
                continue;
            }
            if (nval->st == cell->st) {
                if (nval->et > cell->et) {
                    cur = n;
                    continue;
                }
                if (nval->et == cell->et) {
                    if (*nval == *cell) {
                        nval->addLabels(cell->word->labels);
                        return n;
                    }
                    cur = n;
                    continue;
                }
            }
            auto m = makeCursor(cell, n, cur);
            this->size++;
            cur->prev = m;
            n->lack   = m;
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
