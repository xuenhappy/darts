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

#include <set>
#include <string>
#include <vector>

class Atom {
   public:
    std::string image;  // image of atom
    uint32_t st;        // start of this atom in str
    uint32_t et;        // end of this atom in str
    std::set<std::string> *tags;

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
            this->tags = new std::set<std::string>(*atom.tags);
        }
    }

    void addType(const char *tag) {
        if (tag != NULL) {
            if (this->tags == NULL) {
                this->tags = new std::set<std::string>();
            }
            this->tags->insert(tag);
        }
    }

    ~Atom() {
        if (tags != NULL) {
            delete tags;
            tags = NULL;
        }
    }
};

class AtomList {
   public:
    std::vector<Atom *> data;
};

class Word {
   public:
    Atom *word;     // the word image
    uint32_t st;    // the word in atomlist start
    uint32_t et;    // the words in atom list end
    void *att;      // this word other attr
    uint16_t feat;  // this word other type

    Word(const Word &wd) {
        this->word = wd.word;
        this->st = wd.st;
        this->et = wd.et;
        this->feat = wd.feat;
    }
};

#endif  // SRC_CORE_DARTS_HPP_
