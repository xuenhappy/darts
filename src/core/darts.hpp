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

#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "../core/wtype.h"
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
    }

    Atom(const Atom &atom) {
        this->image = atom.image;
        this->st = atom.st;
        this->et = atom.et;
        if (atom.tags != NULL) {
            this->tags = new std::set<WordType>(*atom.tags);
        }
    }

    void addType(WordType type) {
        if (type != NULL) {
            if (this->tags == NULL) {
                this->tags = new std::set<WordType>();
            }
            this->tags->insert(type);
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
        output << "Atom[ " << D.image << "]";
        return output;
    }
};

class AtomList {
   public:
    std::vector<Atom *> *data;
    std::string ori_str;

    explicit AtomList(size_t n) {
        this->data = new std::vector<Atom *>();
        this->data->reserve(n);
    }

    /***
     *create the Atom list from data
     * */
    static AtomList *createFromString(const std::string &str) {
        AtomList *result = new AtomList(str.length() / 3 + 5);
        result->ori_str = str;
        // TODO(en.xu): split str
        return result;
    }

    void clear() {
        if (!this->data) {
            return;
        }
        for (auto val : *(this->data)) {
            if (val) {
                delete val;
            }
        }
        this->data->clear();
    }

    size_t size() {
        return this->data == NULL ? 0 : this->data->size();
    }

    ~AtomList() {
        if (this->data) {
            clear();
            delete this->data;
        }
    }

    friend std::ostream &operator<<(std::ostream &output, const AtomList &D) {
        output << "AtomList[ ";
        if (D.data) {
            for (auto a : *D.data) {
                output << *a << ",";
            }
        }
        output << "]";
        return output;
    }
};

class Word {
   public:
    Atom *word;       // the word image
    uint32_t st;      // the word in atomlist start
    uint32_t et;      // the words in atom list end
    double *att;      // this word other attr
    size_t att_size;  // thid word attr size
    uint16_t feat;    // this word other type

    Word(Atom *atom, uint32_t start, uint32_t end) {
        this->word = atom;
        this->st = start;
        this->et = end;
    }

    Word(const Word &wd) {
        this->word = wd.word;
        this->st = wd.st;
        this->et = wd.et;
        this->feat = wd.feat;
    }
    friend std::ostream &operator<<(std::ostream &output, const Word &D) {
        if (D.word) {
            output << "Word[ " << D.word->image << "]";
        } else {
            output << "Word[NULL] ";
        }
        return output;
    }
};

class CellMap {
};

}  // namespace darts
#endif  // SRC_CORE_DARTS_HPP_
