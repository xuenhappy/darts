/*
 * File: darts4py.hpp
 * Project: impl
 * 提供一个用于pythond的dll接口
 * File Created: Saturday, 11th December 2021 8:09:57 pm
 * Author: Xu En (xuen@mokar.com)
 * -----
 * Last Modified: Saturday, 11th December 2021 8:10:25 pm
 * Modified By: Xu En (xuen@mokahr.com)
 * -----
 * Copyright 2021 - 2021 Your Company, Moka
 */
#ifndef SRC_MAIN_DARTS4PY_HPP_
#define SRC_MAIN_DARTS4PY_HPP_
#include "./darts.h"
#include <string.h>
#include <cstddef>
#include <cstdlib>
#include "../core/segment.hpp"
#include "../impl/confparser.hpp"
#include "../utils/dcompile.hpp"
#include "../utils/utill.hpp"
#include "core/core.hpp"

struct _dregex {
    dregex::Trie* dat;
};
struct _segment {
    darts::Segment* segment;
};
struct _atomlist {
    darts::AtomList* alist;
};

struct _wordlist {
    std::vector<std::shared_ptr<darts::Word>> wlist;
};

// first do
void init_darts_env() { initUtils(); }
// last do
void destroy_darts_env() { google::protobuf::ShutdownProtobufLibrary(); }

// normalize a str
char* normalize_str(const char* str, size_t len, size_t* ret) {
    if (!str) return NULL;
    std::string normals = normalize(str);

    *ret = normals.size();
    return strdup(normals.c_str());
}
// a string word type
const char* chtype(const char* word) {
    if (!word) return NULL;
    auto& tname = charType(utf8_to_unicode(word));
    return tname.c_str();
}

// convert a text to a alist
atomlist asplit(const char* txt, size_t textlen, bool skip_space, bool normal_before) {
    if (!txt || textlen < 0) return NULL;
    atomlist alist = new struct _atomlist;
    alist->alist   = new darts::AtomList(txt, skip_space, normal_before);
    return alist;
}
// free the atomlist
void free_alist(atomlist alist) {
    if (alist) {
        if (alist->alist) {
            delete alist->alist;
            alist->alist = NULL;
        }
        delete alist;
    }
}
// give the alist len
size_t alist_len(atomlist alist) { return alist == NULL ? 0 : alist->alist->size(); }
void walk_alist(atomlist alist, walk_alist_hit hit, void* user_data) {
    if (!alist || !alist->alist) return;
    atom_ atm;
    size_t len = alist->alist->size();
    for (size_t i = 0; i < len; ++i) {
        auto x        = alist->alist->at(i);
        atm.image     = x->image.c_str();
        atm.char_type = x->char_type.c_str();
        atm.masked    = x->masked;
        atm.st = x->st, atm.et = x->et;
        if (hit(user_data, &atm)) break;
    }
}
// load the dregex from file
dreg load_dregex(const char* path) {
    if (!path) return NULL;
    dregex::Trie* trie = new dregex::Trie();
    if (trie->loadPb(path)) {
        delete trie;
        return NULL;
    }
    auto reg = new struct _dregex();
    reg->dat = trie;
    return reg;
}
// free the dregex
void free_dregex(dreg regex) {
    if (regex) {
        if (regex->dat) {
            delete regex->dat;
            regex->dat = NULL;
        }
        delete regex;
    }
}
class C_AtomIter_ : public dregex::StringIter {
   private:
    void* user_data;
    atomiter iter_func;

   public:
    C_AtomIter_(void* user_data, atomiter iter_func) {
        this->user_data = user_data;
        this->iter_func = iter_func;
    }
    void walks(std::function<bool(const std::string&, size_t)> hit) const {
        atomiter_ret ret;
        while (iter_func(user_data, &ret)) {
            if (hit(ret.word, ret.postion)) break;
        }
    }
};

// parse a atom list
void parse(dreg regex, atomiter atomlist, dhit hit, void* user_data) {
    if (!regex || !regex->dat) return;
    auto dat = regex->dat;
    C_AtomIter_ alist(user_data, atomlist);
    std::vector<const char*> tmpl;
    dhit_ret ret;
    auto hitfunc = [&](size_t s, size_t e, const std::set<int64_t>* labels) -> bool {
        tmpl.clear();
        if (labels != NULL && (!labels->empty())) {
            for (auto idx : *labels) {
                tmpl.push_back(dat->getLabel(idx));
            }
        }
        ret.s = s, ret.e = e;
        ret.labels_size = tmpl.size();
        ret.labels      = &tmpl[0];
        return hit(user_data, &ret);
    };
    dat->parse(alist, hitfunc);
}
class CStr_AtomIter_ : public dregex::StringIter {
   private:
    const char** words;
    size_t len;

   public:
    CStr_AtomIter_(const char** words, size_t len) {
        this->words = words;
        this->len   = len;
    }
    void walks(std::function<bool(const std::string&, size_t)> hit) const {
        for (size_t i = 0; i < len; ++i) hit(words[i], i);
    }
};

class C_KvIter_ : public dregex::KvPairsIter {
   private:
    void* user_data;
    kviter iter_func;

   public:
    C_KvIter_(void* user_data, kviter iter_func) {
        this->user_data = user_data;
        this->iter_func = iter_func;
    }
    int iter(std::function<void(const dregex::StringIter&, const char** lables, size_t label_size)> hit) const {
        kviter_ret ret;
        while (iter_func(user_data, &ret)) {
            CStr_AtomIter_ iter(ret.key, ret.keylen);
            hit(iter, ret.labels, ret.label_nums);
        }
        return EXIT_SUCCESS;
    }
};
// compile regex
int compile_regex(const char* outpath, kviter kvs, void* user_data) {
    if (!outpath) outpath = "dregex.pb";
    C_KvIter_ pairs(user_data, kvs);
    return compile(pairs, outpath);
}
// load segment
segment load_segment(const char* conffile, const char* mode, bool isdevel) {
    if (!conffile) return NULL;
    darts::Segment* sgemnet = NULL;
    if (loadSegment(conffile, &sgemnet, mode, isdevel)) {
        if (!sgemnet) delete sgemnet;
        return NULL;
    }
    if (!sgemnet) return NULL;
    segment sg = new struct _segment;

    sg->segment = sgemnet;
    return EXIT_SUCCESS;
}
// free segment
void free_segment(segment sg) {
    if (sg) {
        if (sg->segment) delete sg->segment;
        sg->segment = NULL;
        delete sg;
    }
}
// token str
wordlist token_str(segment sg, atomlist alist, bool max_mode) {
    if (!sg || !sg->segment || !alist || !alist->alist) return NULL;
    auto sgment  = sg->segment;
    wordlist ret = new struct _wordlist;
    darts::tokenize(*sgment, *(alist->alist), ret->wlist, max_mode);
    return ret;
}
// free list
void free_wordlist(wordlist wlist) {
    if (wlist) {
        wlist->wlist.clear();
        delete wlist;
    }
}

#endif  // SRC_MAIN_DARTS4PY_HPP_
