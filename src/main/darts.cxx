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
#include "../core/segment.hpp"
#include "../impl/confparser.hpp"
#include "../utils/dregex.hpp"
#include "../utils/utill.hpp"

struct _dregex {
    dregex::Trie* dat;
};
struct _segment {
    darts::Segment* segment;
};

void init_darts() { initUtils(); }

void destroy_darts() { google::protobuf::ShutdownProtobufLibrary(); }

char* normalize_str(const char* str, size_t len, size_t* ret) {
    if (!str) return NULL;
    std::string normals = normalize(str);

    *ret = normals.size();
    return strdup(normals.c_str());
}

dreg load_drgex(const char* path) {
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

void free_dregex(dreg regex) {
    if (regex) {
        if (regex->dat) {
            delete regex->dat;
            regex->dat = NULL;
        }
        delete regex;
    }
}

class _C_AtomListIter : public dregex::StringIter {
   private:
    ext_data user_data;
    atom_iter iter_func;

   public:
    _C_AtomListIter(ext_data user_data, atom_iter iter_func) {
        this->user_data = user_data;
        this->iter_func = iter_func;
    }
    void iter(std::function<bool(const std::string&, size_t)> hit) const {
        const char* chars = NULL;
        size_t postion    = 0;
        while (iter_func(&chars, &postion, user_data)) {
            if (hit(chars, postion)) {
                break;
            }
        }
    }
};

void parse(dreg regex, atom_iter atomlist, dregex_hit hit, ext_data user_data) {
    if (!regex || !regex->dat) return;
    auto dat = regex->dat;
    _C_AtomListIter alist(user_data, atomlist);
    std::vector<const char*> tmpl;
    dat->parse(alist, [&](size_t s, size_t e, const std::set<int64_t>* labels) -> bool {
        tmpl.clear();
        if (labels && (!labels->empty())) {
            for (auto idx : *labels) {
                tmpl.push_back(dat->getLabel(idx));
            }
        }
        return hit(s, e, tmpl.empty() ? NULL : &tmpl[0], tmpl.size(), user_data);
    });
}

int load_segment(const char* json_conf_file, segment* sg, const char* mode) {
    if (!json_conf_file) return EXIT_FAILURE;
    darts::Segment* sgemnet = NULL;
    if (loadSegment(json_conf_file, &sgemnet, mode)) {
        if (!sgemnet) delete sgemnet;
        return EXIT_FAILURE;
    }
    if (!sgemnet) return EXIT_FAILURE;
    *sg            = new struct _segment();
    (*sg)->segment = sgemnet;
    return EXIT_SUCCESS;
}

void free_segment(segment sg) {
    if (sg) {
        if (sg->segment) delete sg->segment;
        sg->segment = NULL;
        delete sg;
    }
}

void token_str(segment sg, const char* txt, word_hit hit, bool max_mode, ext_data user_data) {
    if (!sg || !sg->segment || !txt) return;
    auto sgment = sg->segment;
    std::vector<std::shared_ptr<darts::Word>> ret;
    darts::tokenize(*sgment, txt, ret, max_mode);
    for (auto w : ret) {
        hit(w->text().c_str(), NULL, a->st, a->et, w->st, w->et, user_data);
    }
}

int word_type(const char* word, char** ret) {
    if (!word) return 0;
    std::string tname = charType(utf8_to_unicode(word));
    char* c           = (char*)malloc(sizeof(char) * (tname.size() + 1));
    std::copy(tname.begin(), tname.end(), c);
    c[tname.size()] = '\0';
    *ret            = c;
    return tname.size();
}

void word_split(const char* str, token_hit hit, ext_data user_data) {
    if (!str) return;
    atomSplit(str, [&](const char* atom, std::string& type, size_t s, size_t e) {
        hit(atom, type.c_str(), s, e, user_data);
    });
}
void word_bpe(const char* str, token_hit hit, ext_data user_data) {
    if (!str) return;
}
#endif  // SRC_MAIN_DARTS4PY_HPP_
