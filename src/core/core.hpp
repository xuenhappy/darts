/*
 * File: Darts.h
 * Project: core
 * File Created: Saturday, 11th December 2021 6:45:52 pm
 * Author: Xu En (xuen@mokar.com)
 * -----
 * Last Modified: Saturday, 11th December 2021 8:14:04 pm
 * Modified By: Xu En (nanhangxuen@163.com)
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
#include "../utils/chrtool.hpp"
#include "../utils/chspliter.hpp"
#include "../utils/codecvt.hpp"
#include "../utils/strtool.hpp"

namespace darts {
class Atom {
   public:
    std::string image;  // image of atom
    uint32_t st;        // start of this atom in str
    uint32_t et;        // end of this atom in str
    std::string char_type;
    bool masked;  // this is for training

    Atom(const std::string& image, uint32_t start, uint32_t end) : masked(false) {
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
    explicit AtomList(const std::string& str, bool skip_space = true, bool norm = false) {
        if (norm) {
            std::string nstr;
            this->str = to_utf32(normalize(str, nstr));
        } else {
            this->str = to_utf32(str);
        }
        data.reserve(this->str.length());
        auto accept = [&](const std::string& astr, const std::string& ttype, size_t s, size_t e) {
            if (skip_space && ttype == char_type::EMPTY) return;
            auto atom       = std::make_shared<Atom>(astr, s, e);
            atom->char_type = ttype;
            this->data.push_back(atom);
        };
        // When spaces are skipped they must first be emitted as independent
        // EMPTY atoms. Otherwise the splitter's legacy POS/EMPTY coalescing
        // can produce atoms such as " , " that the filter cannot remove.
        atomSplit(this->str, accept, !skip_space);
    }

    /**
     * @brief Construct a new Atom List object sub
     *
     * @param other
     * @param s
     * @param e
     */
    AtomList(const AtomList& other, int s, int e) {
        if (s < 0 || e > static_cast<int>(other.data.size()) || s >= e) return;
        auto st = other.data[s]->st;
        auto et = other.data[e - 1]->et;

        this->str = other.str.substr(st, et - st);
        this->data.insert(data.end(), other.data.begin() + s, other.data.begin() + e);
    }

    size_t size() const { return data.size(); }

    ~AtomList() {
        data.clear();
        str.clear();
    }

    std::string subAtom(size_t start, size_t end) const {
        if (start < 0 || end > data.size() || start >= end) return "";
        size_t bytes = 0;
        for (size_t i = start; i < end; ++i) bytes += data[i]->image.size();
        std::string result;
        result.reserve(bytes);
        for (size_t i = start; i < end; ++i) result.append(data[i]->image);
        return result;
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

    std::shared_ptr<Atom> operator[](int index) const { return at(index); }

    friend std::ostream& operator<<(std::ostream& output, const AtomList& D) {
        output << "AtomList[";
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
    std::shared_ptr<std::vector<float>> att;  // att data
    size_t att_offset = 0;
    size_t att_size = 0;
    std::set<std::string> labels;
    std::string image;

   public:
    int st;    // the word in atomlist start
    int et;    // the words in atom list end
    int feat;  // this word other type
    int vocab_id;

    bool isStSpecial() { return st < 0; }

    bool isEtSpecial() { return et < 0; }

    bool isSpecial() { return st < 0 || et < 0; }

    /**
     * @brief Construct a new Word object
     *
     * @param atom
     * @param start  start of postion in atom list
     * @param end  end of postion in atom list
     */
    Word(const std::string& image, int start, int end) : feat(-1), vocab_id(-1) {
        this->image = image;
        this->st    = start;
        this->et    = end;
        this->att   = nullptr;
    }

    Word(const AtomList& alist, int start, int end) : feat(-1), vocab_id(-1) {
        this->image = alist.subAtom(start, end);
        this->st    = start;
        this->et    = end;
        this->att   = nullptr;
    }

    Word(const Word& wd) {
        this->st   = wd.st;
        this->et   = wd.et;
        this->feat = wd.feat;
        this->vocab_id = wd.vocab_id;
        this->att  = wd.att;
        this->att_offset = wd.att_offset;
        this->att_size = wd.att_size;
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
        output << "Word[" << D.image << "(" << D.st << "," << D.et << ")]";
        return output;
    }

    void setAtt(std::shared_ptr<std::vector<float>> value) {
        this->att = std::move(value);
        this->att_offset = 0;
        this->att_size = this->att ? this->att->size() : 0;
    }
    void setAttView(const std::shared_ptr<std::vector<float>>& value, size_t offset, size_t size) {
        this->att = value;
        this->att_offset = offset;
        this->att_size = size;
    }
    std::shared_ptr<std::vector<float>> getAtt() { return this->att; }
    const float* getAttData() const {
        return att && att_offset + att_size <= att->size() ? att->data() + att_offset : nullptr;
    }
    size_t getAttSize() const { return att_size; }

    void addLabel(const std::string& tag) { labels.insert(tag); }
    void addLabels(const std::set<std::string>* tags) {
        if (tags != nullptr) labels.insert(tags->begin(), tags->end());
    }
    void addLabels(std::vector<std::string>* tags) {
        if (tags != nullptr) labels.insert(tags->begin(), tags->end());
    }
    void addLabels(const std::shared_ptr<Word> other) {
        if (other != nullptr) labels.insert(other->labels.begin(), other->labels.end());
    }

    size_t labels_nums() const { return labels.size(); }
    const std::set<std::string>& getLabels() const { return this->labels; }

    /**
     * @brief get HX(label) max value label
     *
     * @param hx_func
     * @return const std::string
     */
    const std::string& maxHlabel(std::function<float(const std::string&)> hx_func) const {
        if (labels.empty()) {
            static std::string empty("");
            return empty;
        }
        if (hx_func == nullptr) return *labels.begin();
        const std::string* ret = &(*labels.begin());
        float oreder           = -1e10;
        for (auto it = labels.begin(); it != labels.end(); ++it) {
            float xorder = hx_func(*it);
            if (xorder > oreder) {
                oreder = xorder;
                ret    = &(*it);
            }
        }
        return *ret;
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
    bool indexed;
    std::vector<Cursor> indexed_nodes;
    std::vector<std::vector<Cursor>> indexed_rows;

    std::shared_ptr<Word> src_node;
    std::shared_ptr<Word> end_node;

   public:
    std::shared_ptr<Word> SrcNode() { return src_node; }
    std::shared_ptr<Word> EndNode() { return end_node; }

    Cursor Head() { return this->head; }

    size_t Size() const { return this->size; }

    size_t Row() const { return this->rows; }

    size_t Column() const { return this->colums; }

    SegPath() {
        src_node   = std::make_shared<Word>("", -1, 0);
        end_node   = std::make_shared<Word>("", 0, -1);
        this->head = makeCursor(src_node, nullptr, nullptr);

        this->head->idx = -1;
        this->rows = this->colums = this->size = 0;
        this->indexed = false;
    }

    ~SegPath() {
        src_node  = nullptr;
        end_node  = nullptr;
        auto node = head;
        while (node) {
            auto next = node->lack;
            if (node->val) node->val = nullptr;
            delete node;
            node = next;
        }
        this->rows = this->colums = this->size = 0;
    }

    /**
     * @brief 创建原始索引
     *
     */
    void indexIt() {
        auto node = this->head;
        node->idx = -1;
        int index = node->idx;
        indexed_nodes.clear();
        indexed_nodes.reserve(size);
        indexed_rows.clear();
        indexed_rows.resize(rows + 1);
        while (node->lack) {
            node = node->lack;
            index++;
            node->idx = index;
            indexed_nodes.push_back(node);
            auto row = static_cast<size_t>(node->val->st);
            if (row >= indexed_rows.size()) indexed_rows.resize(row + 1);
            indexed_rows[row].push_back(node);
        }
        indexed = true;
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
            if (indexed) {
                for (auto node : indexed_nodes) dfunc(node);
                return;
            }
            while (cur->lack) {
                dfunc(cur->lack);
                cur = cur->lack;
            }
            return;
        }

        if (indexed) {
            auto index = static_cast<size_t>(row);
            if (index < indexed_rows.size()) {
                for (auto node : indexed_rows[index]) dfunc(node);
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
            this->indexed = false;
            cur->lack = m;
            n->prev   = m;
            return m;
        }
        cur->lack = makeCursor(cell, cur, nullptr);
        this->size++;
        this->indexed = false;
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
            this->indexed = false;
            cur->prev = m;
            n->lack   = m;
            return m;
        }
        cur->prev = makeCursor(cell, this->head, cur);
        this->size++;
        this->indexed = false;
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
        if (!cur) return addNext(this->head, cell);
        if (cur->val->st < cell->st) return addNext(cur, cell);
        if ((cur->val->st == cell->st) && (cur->val->et <= cell->et)) return addNext(cur, cell);
        return addPre(cur, cell);
    }
};

}  // namespace darts
#endif  // SRC_CORE_DARTS_HPP_
