/*
 * File: recognizer.hpp
 * Project: impl
 * File Created: Saturday, 25th December 2021 11:05:37 pm
 * Author: Xu En (xuen@mokar.com)
 * -----
 * Last Modified: Saturday, 1st January 2022 12:14:11 pm
 * Modified By: Xu En (xuen@mokahr.com)
 * -----
 * Copyright 2021 - 2022 Your Company, Moka
 */

#ifndef SRC_IMPL_RECOGNIZER_HPP_
#define SRC_IMPL_RECOGNIZER_HPP_

#include <assert.h>
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include "../core/segment.hpp"
#include "../utils/dregex.hpp"
#include "../utils/pinyin.hpp"
#include "../utils/strtool.hpp"
#include "./encoder.hpp"
#include "core/core.hpp"

namespace darts {
class AtomListIterator : public dregex::StringIter {
   private:
    const AtomList* m_list;

   public:
    explicit AtomListIterator(const AtomList& m_list) { this->m_list = &m_list; }
    void walks(std::function<bool(const std::string&, size_t)> hit) const {
        std::string tmp;
        for (auto i = 0; i < m_list->size(); i++) {
            tmp = m_list->at(i)->image;
            if (hit(tolower(tmp), i)) {
                break;
            }
        }
    }
};
/**
 * @brief a recongnizer that base on dict
 *
 */
class DictWordRecongnizer : public CellRecognizer {
   private:
    static const char* PB_FILE_KEY;
    static const char* ATOM_MODE_KEY;

    dregex::Trie trie;
    bool atom_mode;

   public:
    DictWordRecongnizer() {}
    explicit DictWordRecongnizer(const std::string& pbfile) { this->trie.loadPb(pbfile); }

    int initalize(const std::map<std::string, std::string>& param,
                  std::map<std::string, std::shared_ptr<SegmentPlugin>>& dicts) {
        auto it = param.find(ATOM_MODE_KEY);
        if (it != param.end() && "true" == it->second) {
            atom_mode = true;
        }
        auto iter = param.find(PB_FILE_KEY);
        if (iter == param.end()) {
            std::cerr << PB_FILE_KEY << " key not found in dictionary!" << std::endl;
            return EXIT_FAILURE;
        }
        return this->trie.loadPb(iter->second);
    }

    void addWords(const AtomList& dstSrc, SegPath& cmap) const {
        AtomListIterator iter(dstSrc);
        auto cur = cmap.Head();
        if (atom_mode) {
            auto hit = [&](size_t s, size_t e, const std::set<int64_t>* labels) -> bool {
                if (!labels) return false;
                int feat = -1;
                for (auto tidx : *labels) {
                    auto tag = this->trie.getLabel(tidx);
                    if (tag) {
                        auto word = std::make_shared<Word>(dstSrc, s, e);
                        word->addLabel(tag);
                        word->feat = feat;
                        feat--;
                        cur = cmap.addCell(word, cur);
                    }
                }
                return false;
            };
            this->trie.parse(iter, hit);
        } else {
            auto hit = [&](size_t s, size_t e, const std::set<int64_t>* labels) -> bool {
                auto word = std::make_shared<Word>(dstSrc, s, e);
                if (labels)
                    for (auto tidx : *labels) {
                        auto tag = this->trie.getLabel(tidx);
                        if (tag) word->addLabel(tag);
                    }
                cur = cmap.addCell(word, cur);
                return false;
            };
            this->trie.parse(iter, hit);
        }
    }
};
const char* DictWordRecongnizer::PB_FILE_KEY   = "pbfile.path";
const char* DictWordRecongnizer::ATOM_MODE_KEY = "atom.mode";
REGISTER_Recognizer(DictWordRecongnizer);

/**
 * pinyin标注,该插件不可以❌和其他插件混用
 */
class PinyinRecongnizer : public CellRecognizer {
   private:
    std::shared_ptr<PinyinEncoder> encoder;
    static const char* ENCODER_PARAM;

   public:
    int initalize(const std::map<std::string, std::string>& params,
                  std::map<std::string, std::shared_ptr<SegmentPlugin>>& plugins) {
        auto it = plugins.find(ENCODER_PARAM);
        if (it == plugins.end()) {
            std::cerr << "could not find plugin " << ENCODER_PARAM << std::endl;
            return EXIT_FAILURE;
        }
        this->encoder = std::dynamic_pointer_cast<PinyinEncoder>(it->second);
        if (this->encoder == nullptr) {
            std::cerr << "plugin in init error " << ENCODER_PARAM << std::endl;
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }

    void addWords(const AtomList& dstSrc, SegPath& cmap) const {
        std::vector<std::shared_ptr<Word>> adds;
        auto dfunc = [this, &adds](Cursor cur) {
            auto pw   = cur->val;
            auto pyin = pinyin(pw->text());
            if (pyin == nullptr) return;
            pw->addLabel(pyin->piyins[0]);
            pw->feat = encoder->encode(pyin->piyins[0]);
            for (size_t i = 1; i < pyin->piyins.size(); ++i) {
                auto w = std::make_shared<Word>(pw->text(), pw->st, pw->et);
                w->addLabel(pyin->piyins[i]);
                w->feat = encoder->encode(pyin->piyins[i]);
                adds.push_back(w);
            }
        };
        cmap.iterRow(nullptr, -1, dfunc);
        if (!adds.empty()) {
            auto cur = cmap.Head();
            for (auto w : adds) {
                cur = cmap.addNext(cur, w);
            }
            adds.clear();
        }
    }
    bool exclusive() { return true; }
};

const char* PinyinRecongnizer::ENCODER_PARAM = "pyin.encoder";
REGISTER_Recognizer(PinyinRecongnizer);

class RuleRecongnizer : public CellRecognizer {
   public:
    /**
     * @brief use rule find all possable word in alist and add into buffer
     *
     * @param dstSrc
     * @param buffer
     */
    virtual void seek(const AtomList& dstSrc, std::vector<std::shared_ptr<Word>>& buffer) const = 0;
    void addWords(const AtomList& dstSrc, SegPath& cmap) const {
        std::vector<std::shared_ptr<Word>> buffer;
        buffer.reserve(dstSrc.size() / 2 + 1);
        seek(dstSrc, buffer);
        auto comp = [](std::shared_ptr<Word> i1, std::shared_ptr<Word> i2) -> bool { return (i1->st < i2->st); };
        std::sort(buffer.begin(), buffer.end(), comp);
        Cursor cur = cmap.Head();
        for (auto x : buffer) {
            cur = cmap.addNext(cur, x);
        }
        buffer.clear();
    }
};

}  // namespace darts

#endif  // SRC_IMPL_RECOGNIZER_HPP_
