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
#include "../core/segment.hpp"
#include "../impl/jsonconf.hpp"
#include "../utils/dregex.hpp"
#include "../utils/utils_base.hpp"

struct _dregex {
    darts::Trie* dat;
};
struct _segment {
    darts::Segment* segment;
};

void init_darts() { initUtils(); }

void destroy_darts() { google::protobuf::ShutdownProtobufLibrary(); }

int normalize_str(const char* str, char** ret) {
    if (!str) return 0;
    std::string normals = normalizeStr(str);
    char* c             = (char*)malloc(sizeof(char) * (normals.size() + 1));
    std::copy(normals.begin(), normals.end(), c);
    c[normals.size()] = '\0';
    *ret              = c;
    return normals.size();
}

int load_drgex(const char* path, dregex* regex) {
    if (!path) return EXIT_FAILURE;
    darts::Trie* trie = new darts::Trie();
    if (trie->loadPb(path)) {
        delete trie;
        return EXIT_FAILURE;
    }
    *regex        = new struct _dregex();
    (*regex)->dat = trie;
    return EXIT_SUCCESS;
}

void free_dregex(dregex regex) {
    if (regex) {
        if (regex->dat) {
            delete regex->dat;
            regex->dat = NULL;
        }
        delete regex;
    }
}

class _C_AtomListIter : public darts::StringIter {
   private:
    darts_ext user_data;
    atom_iter iter_func;

   public:
    _C_AtomListIter(darts_ext user_data, atom_iter iter_func) {
        this->user_data = user_data;
        this->iter_func = iter_func;
    }
    void iter(std::function<bool(const std::string&, size_t)> hit) {
        const char* chars = NULL;
        size_t postion    = 0;
        while (iter_func(&chars, &postion, user_data)) {
            if (hit(chars, postion)) {
                break;
            }
        }
    }
};

void parse(dregex regex, atom_iter atomlist, dregex_hit hit, darts_ext user_data) {
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

int load_segment(const char* json_conf_file, segment* sg) {
    if (!json_conf_file) return EXIT_FAILURE;
    darts::Segment* sgemnet = NULL;
    if (parseJsonConf(json_conf_file, &sgemnet)) {
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

void token_str(segment sg, const char* txt, word_hit hit, bool max_mode, darts_ext user_data) {
    if (!sg || !sg->segment || !txt) return;
    auto sgment = sg->segment;
    std::vector<std::shared_ptr<darts::Word>> ret;
    darts::tokenize(*sgment, txt, ret, max_mode);
    for (auto w : ret) {
        auto a = w->word;
        hit(a->image.c_str(), NULL, a->st, a->et, w->st, w->et, user_data);
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

void word_split(const char* str, token_hit hit, darts_ext user_data) {
    if (!str) return;
    atomSplit(str, [&](const char* atom, std::string& type, size_t s, size_t e) {
        hit(atom, type.c_str(), s, e, user_data);
    });
}
void word_bpe(const char* str, token_hit hit, darts_ext user_data) {
    if (!str) return;
}
#endif  // SRC_MAIN_DARTS4PY_HPP_
