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
    std::string char_type;
    bool masked;  // this is for training

    Atom(const char* image, uint32_t start, uint32_t end) : masked(false) {
        this->image = image;
        this->st    = start;
        this->et    = end;
    }
    Atom(const Atom& atom) {
        this->image = atom.image;

        this->st = atom.st;
        this->et = atom.et;

        this->masked    = atom.masked;
        this->char_type = atom.char_type;
    }
    ~Atom() {}
    friend std::ostream& operator<<(std::ostream& output, const Atom& D) {
        output << "Atom['" << D.image << "':{" << D.char_type << "}]";
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
            auto atom       = std::make_shared<Atom>(astr, s, e);
            atom->char_type = ttype;
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
        atom->char_type   = data[start]->char_type;
        return atom;
    }

    std::string subAtom(size_t start, size_t end) const {
        if (start < 0 || end >= data.size() || start >= end) return "";
        auto st = data[start]->st;
        auto et = data[end - 1]->et;
        return to_utf8(str.substr(st, et - st));
    }

    /**
     * @brief get atom object
     *
     * @param index
     * @return std::shared_ptr<Atom>
     */
    std::shared_ptr<Atom> at(size_t index) const {
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
    std::shared_ptr<void> att;  // att data
    std::set<std::string> labels;
    std::string image;

   public:
    int st;    // the word in atomlist start
    int et;    // the words in atom list end
    int feat;  // this word other type

    /**
     * @brief Construct a new Word object
     *
     * @param atom
     * @param start  start of postion in atom list
     * @param end  end of postion in atom list
     */
    Word(const std::string& image, int start, int end) : feat(-1) {
        this->image = image;
        this->st    = start;
        this->et    = end;
        this->att   = nullptr;
    }

    Word(const AtomList& alist, int start, int end) : feat(-1) {
        this->image = alist.subAtom(st, et);
        this->st    = start;
        this->et    = end;
        this->att   = nullptr;
    }

    Word(const Word& wd) {
        this->st   = wd.st;
        this->et   = wd.et;
        this->feat = wd.feat;
        this->att  = wd.att;
        labels.insert(wd.labels.begin(), wd.labels.end());
    }

    ~Word() {
        att = nullptr;
        labels.clear();
        image.clear();
    }
    bool operator==(const Word& D) { return D.st == st && D.et == et && D.feat == feat; }

    const std::string& text() const { return this->image; }

    friend std::ostream& operator<<(std::ostream& output, const Word& D) {
        output << "Word[ " << D.image << "(" << D.st << "," << D.et << ")]";
        return output;
    }

    void setAtt(std::shared_ptr<void> att) { this->att = att; }

    std::shared_ptr<void> getAtt() { return this->att; }

    void addLabels(const std::set<std::string>* tags) {
        if (tags != NULL) {
            labels.insert(tags->begin(), tags->end());
        }
    }
    void addLabels(const std::shared_ptr<Word> other) {
        if (other != nullptr) {
            labels.insert(other->labels.begin(), other->labels.end());
        }
    }

    void addLabel(const std::string& tag) { labels.insert(tag); }

    /**
     * @brief add tags into words
     *
     * @param tags
     */
    void addLabels(std::vector<std::string>* tags) {
        if (tags != NULL) {
            labels.insert(tags->begin(), tags->end());
        }
    }
    /**
     * @brief get HX(label) max value label
     *
     * @param hx_func
     * @return const std::string
     */
    const std::string maxHXlabel(std::function<float(const std::string&)> hx_func) const {
        if (labels.empty()) {
            return "";
        }
        if (hx_func == nullptr) {
            for (std::string l : labels) {
                return l;
            }
        } else {
            float oreder = -10;
            std::string ret;
            for (std::string l : labels) {
                float xorder = hx_func(l);
                if (xorder > oreder) {
                    oreder = xorder;
                    ret    = l;
                }
            }
            return ret;
        }
        return "";
    }

    bool hasLabel(const std::string& label) const { return labels.find(label) != labels.end(); }

    bool hasLabel(const std::set<std::string>* labels) const {
        if (labels == NULL || this->labels.empty()) {
            return false;
        }
        for (auto& t : *labels) {
            if (this->labels.find(t) != this->labels.end()) {
                return true;
            }
        }
        return false;
    }
    void joinLabel(const std::set<std::string>* otags) {
        if (!otags || otags->empty()) return;

        if (labels.empty()) {
            addLabels(otags);
            return;
        }
        for (auto it = labels.begin(); it != labels.end();) {
            if (otags->find(*it) == otags->end()) {
                it = labels.erase(it);
                continue;
            }
            ++it;
        }
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
inline Cursor makeCursor(std::shared_ptr<Word> word, Cursor pre, Cursor next) {
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
        this->head      = makeCursor(std::make_shared<Word>("", -1, 0), NULL, NULL);
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
                        nval->addLabels(cell);
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
                        nval->addLabels(cell);
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
