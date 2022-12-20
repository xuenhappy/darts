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
#include <iostream>
#include <map>
#include <memory>
#include <utility>
#ifdef dmalloc
#include <dmalloc.h>
#endif
#include <string.h>
#include <cstddef>
#include <cstdlib>
#include <vector>
#include "../core/segment.hpp"
#include "../impl/confparser.hpp"
#include "../impl/encoder.hpp"
#include "../utils/biggram.hpp"
#include "../utils/dcompile.hpp"
#include "../utils/utill.hpp"
#include "./darts.h"
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
void normalize_str(const char* str, size_t len, void* cpp_string_cache) {
    if (!str || len < 1) return;
    normalize(str, len, *static_cast<std::string*>(cpp_string_cache));
}
// a string word type
const char* chtype(const char* word) {
    if (!word) return nullptr;
    auto& tname = charType(utf8_to_unicode(word));
    return tname.c_str();
}

// convert a text to a alist
atomlist asplit(const char* txt, size_t textlen, bool skip_space, bool normal_before) {
    if (!txt || textlen < 0) return nullptr;
    atomlist alist = new struct _atomlist;
    alist->alist   = new darts::AtomList(txt, skip_space, normal_before);
    return alist;
}
// free the atomlist
void free_alist(atomlist alist) {
    if (alist) {
        if (alist->alist) {
            delete alist->alist;
            alist->alist = nullptr;
        }
        delete alist;
    }
}
// give the alist len
size_t alist_len(atomlist alist) { return alist == nullptr ? 0 : alist->alist->size(); }
void walk_alist(atomlist alist, walk_alist_hit hit, void* user_data) {
    if (!alist || !alist->alist) return;
    atom_buffer buf;
    size_t len = alist->alist->size();
    for (size_t i = 0; i < len; ++i) {
        auto x        = alist->alist->at(i);
        buf.image     = x->image.c_str();
        buf.char_type = x->char_type.c_str();
        buf.masked    = x->masked;
        buf.st = x->st, buf.et = x->et;
        if (hit(user_data, &buf)) break;
    }
}
int get_npos_atom(atomlist alist, size_t idx, atom_buffer* buffer) {
    if (alist == nullptr || alist->alist == nullptr) return EXIT_FAILURE;
    auto al = alist->alist;
    if (idx >= al->size()) return EXIT_FAILURE;
    auto x = al->at(idx);

    buffer->image     = x->image.c_str();
    buffer->char_type = x->char_type.c_str();
    buffer->masked    = x->masked;
    buffer->st = x->st, buffer->et = x->et;
    return EXIT_SUCCESS;
}
// load the dregex from file
dreg load_dregex(const char* path) {
    if (!path) return nullptr;
    dregex::Trie* trie = new dregex::Trie();
    if (trie->loadPb(path)) {
        delete trie;
        return nullptr;
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
            regex->dat = nullptr;
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
        atomiter_buffer buf;
        while (iter_func(user_data, &buf))
            if (hit(buf.word, buf.postion)) break;
    }
};

// parse a atom list
void parse(dreg regex, atomiter atomlist, dhit hit, void* user_data) {
    if (!regex || !regex->dat) return;
    auto dat = regex->dat;
    C_AtomIter_ alist(user_data, atomlist);
    std::vector<const char*> tmpl;
    dhit_buffer buf;
    auto hitfunc = [&](size_t s, size_t e, const std::vector<int64_t>* labels) -> bool {
        tmpl.clear();
        if (labels != nullptr && (!labels->empty())) {
            for (auto idx : *labels) {
                tmpl.push_back(dat->getLabel(idx));
            }
        }
        buf.s = s, buf.e = e;
        buf.labels_size = tmpl.size();
        buf.labels      = &tmpl[0];
        return hit(user_data, &buf);
    };
    dat->parse(alist, hitfunc);
}
class CStr_AtomIter_ : public dregex::StringIter {
   private:
    std::vector<const char*>* cache;

   public:
    CStr_AtomIter_(std::vector<const char*>* cache) { this->cache = cache; }
    void walks(std::function<bool(const std::string&, size_t)> hit) const {
        for (size_t i = 0; i < cache->size(); ++i) hit((*cache)[i], i);
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
        // init cache
        std::vector<const char*> keycache;
        std::vector<const char*> labelcache;
        kviter_buffer buf;
        buf.key_cache   = &keycache;
        buf.label_cache = &labelcache;
        // set data and hit
        while (iter_func(user_data, &buf)) {
            CStr_AtomIter_ iter(&keycache);
            hit(iter, &labelcache[0], labelcache.size());
            keycache.clear();
            labelcache.clear();
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
    if (!conffile) return nullptr;
    darts::Segment* sgemnet = nullptr;
    if (loadSegment(conffile, &sgemnet, mode, isdevel)) {
        if (!sgemnet) delete sgemnet;
        return nullptr;
    }
    if (!sgemnet) return nullptr;
    segment sg = new struct _segment;

    sg->segment = sgemnet;
    return EXIT_SUCCESS;
}
// free segment
void free_segment(segment sg) {
    if (sg) {
        if (sg->segment) delete sg->segment;
        sg->segment = nullptr;
        delete sg;
    }
}

void walk_wlist(wordlist wlist, walk_wlist_hit hit, void* user_data) {
    word_buffer buf;
    std::vector<const char*> ptrs;
    buf.label_cache = &ptrs;
    for (size_t i = 0; i < wlist->wlist.size(); ++i) {
        auto w         = wlist->wlist[i];
        buf.image      = w->text().c_str();
        buf.atom_s     = w->st;
        buf.atom_e     = w->et;
        buf.label_nums = w->labels_nums();
        ptrs.clear();
        if (w->labels_nums() > 0) {
            for (auto& s : w->getLabels()) {
                ptrs.emplace_back(s.c_str());
            }
        }
        buf.labels = &ptrs[0];
        if (hit(user_data, &buf)) break;
    }
    ptrs.clear();
}
size_t wlist_len(wordlist wlist) { return !wlist ? 0 : wlist->wlist.size(); }
// get npos word
int get_npos_word(wordlist wlist, size_t index, word_buffer* buffer) {
    if (wlist == nullptr || index >= wlist->wlist.size()) return EXIT_FAILURE;
    std::vector<const char*>* ptrs;
    ptrs   = static_cast<std::vector<const char*>*>(buffer->label_cache);
    auto w = wlist->wlist[index];

    buffer->image      = w->text().c_str();
    buffer->atom_s     = w->st;
    buffer->atom_e     = w->et;
    buffer->label_nums = w->labels_nums();
    ptrs->clear();
    if (w->labels_nums() > 0) {
        for (auto& s : w->getLabels()) {
            ptrs->emplace_back(s.c_str());
        }
    }
    buffer->labels = &(*ptrs)[0];

    return EXIT_SUCCESS;
}

// token str
wordlist token_str(segment sg, atomlist alist, bool max_mode) {
    if (!sg || !sg->segment || !alist || !alist->alist) return nullptr;
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
int build_biggram_dict(const char* single_freq_dict, const char* union_freq_dict, const char* outfile) {
    if (!single_freq_dict || !union_freq_dict | !outfile) return EXIT_FAILURE;
    darts::BigramDict dict;
    if (dict.loadDictFromTxt(single_freq_dict, union_freq_dict)) return EXIT_FAILURE;
    return dict.saveDict(outfile);
}

struct _alist_encoder {
    darts::WordPice* piece;
};
struct _wtype_encoder {
    darts::TypeEncoder* encoder;
};

wtype_encoder get_wtype_encoder(void* map_param, const char* type_cls_name) {
    if (!map_param || !type_cls_name) return nullptr;
    darts::SegmentPlugin* plugin = darts::SegmentPluginRegisterer::GetInstanceByName(type_cls_name);
    if (!plugin) {
        std::cerr << "no wtype encoder found! " << type_cls_name << std::endl;
        return nullptr;
    }

    std::map<std::string, std::string>* params = static_cast<std::map<std::string, std::string>*>(map_param);
    if (!params) {
        std::cerr << "param must map[string,string]" << std::endl;
        return nullptr;
    }
    wtype_encoder encoder = new struct _wtype_encoder;
    encoder->encoder      = static_cast<darts::TypeEncoder*>(plugin);
    std::map<std::string, std::shared_ptr<darts::SegmentPlugin>> plugins;
    if (!encoder->encoder || encoder->encoder->initalize(*params, plugins)) {
        free_wtype_encoder(encoder);
        encoder = nullptr;
    }
    return encoder;
}
void encode_wlist_type(wtype_encoder encoder, wordlist wlist, void* int_vector_buf) {
    if (!encoder || !encoder->encoder || !wlist || wlist->wlist.empty() || !int_vector_buf) return;
    std::vector<int>& buf = *(static_cast<std::vector<int>*>(int_vector_buf));
    buf.reserve(wlist->wlist.size());
    auto& codec = *(encoder->encoder);
    for (auto w : wlist->wlist) buf.emplace_back(codec.encode(w));
}
size_t max_wtype_nums(wtype_encoder encoder) {
    return !encoder || !encoder->encoder ? 0 : encoder->encoder->getLabelSize();
}

void free_wtype_encoder(wtype_encoder encoder) {
    if (encoder) {
        if (encoder->encoder) delete encoder->encoder;
        encoder->encoder = nullptr;
        delete encoder;
    }
}
const char* decode_wtype(wtype_encoder encoder, int wtype) {
    return !encoder || !encoder->encoder ? nullptr : encoder->encoder->decode(wtype).c_str();
}
// wlist encoder
alist_encoder get_alist_encoder(void* map_param, const char* type_cls_name) {
    if (!map_param || !type_cls_name) return nullptr;
    darts::SegmentPlugin* plugin = darts::SegmentPluginRegisterer::GetInstanceByName(type_cls_name);
    if (!plugin) {
        std::cerr << "no alist encoder found! " << type_cls_name << std::endl;
        return nullptr;
    }
    std::map<std::string, std::string>* params = static_cast<std::map<std::string, std::string>*>(map_param);
    if (!params) {
        std::cerr << "param must map[string,string]" << std::endl;
        return nullptr;
    }
    alist_encoder encoder = new struct _alist_encoder;
    encoder->piece        = static_cast<darts::WordPice*>(plugin);
    std::map<std::string, std::shared_ptr<darts::SegmentPlugin>> plugins;
    if (!encoder->piece || encoder->piece->initalize(*params, plugins)) {
        free_alist_encoder(encoder);
        encoder = nullptr;
    }
    return encoder;
}
void encode_alist(alist_encoder encoder, atomlist alist, void* int_pair_vector_buf) {
    if (!encoder || !encoder->piece || !alist || !alist->alist || !int_pair_vector_buf) return;
    using pair_vector_buf = std::vector<std::pair<int, int>>;
    pair_vector_buf* buf  = static_cast<pair_vector_buf*>(int_pair_vector_buf);
    if (!buf) {
        std::cerr << "buf must vector[pair[int,int]]" << std::endl;
        return;
    }
    buf->reserve(alist->alist->size() * 2 + 2);
    auto ret_hit = [buf](int code, int atom_postion) { buf->emplace_back(std::make_pair(code, atom_postion)); };
    encoder->piece->encode(*(alist->alist), ret_hit);
}
void free_alist_encoder(alist_encoder encoder) {
    if (encoder) {
        if (encoder->piece) delete encoder->piece;
        encoder->piece = nullptr;
        delete encoder;
    }
}
size_t max_acode_nums(alist_encoder encoder) {
    return !encoder || !encoder->piece ? 0 : encoder->piece->getLabelSize();
}
const char* decode_atype(alist_encoder encoder, int atype) {
    return !encoder || !encoder->piece ? nullptr : encoder->piece->decode(atype).c_str();
}

#endif  // SRC_MAIN_DARTS4PY_HPP_
