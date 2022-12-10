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
#include "../utils/str_utils.hpp"
#include "./encoder.hpp"

namespace darts {
class AtomListIterator : public StringIter {
   private:
    const AtomList* m_list;
    std::set<std::string> skiptypes;

   public:
    explicit AtomListIterator(const AtomList& m_list) {
        this->m_list = &m_list;
        this->skiptypes.insert("EMPTY");
    }

    void iter(std::function<bool(const std::string&, size_t)> hit) {
        std::string tmp;
        for (auto i = 0; i < m_list->size(); i++) {
            auto a = m_list->at(i);
            if (skiptypes.find(a->char_type) != skiptypes.end()) {
                continue;
            }
            tmp = a->image;
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
    Trie trie;
    static const char* PB_FILE_KEY;

   public:
    DictWordRecongnizer() {}
    explicit DictWordRecongnizer(const std::string& pbfile) { this->trie.loadPb(pbfile); }

    int initalize(const std::map<std::string, std::string>& param,
                  std::map<std::string, std::shared_ptr<SegmentPlugin>>& dicts) {
        auto iter = param.find(PB_FILE_KEY);
        if (iter == param.end()) {
            std::cerr << PB_FILE_KEY << " key not found in dictionary!" << std::endl;
            return EXIT_FAILURE;
        }
        return this->trie.loadPb(iter->second);
    }

    void addSomeCells(const AtomList& dstSrc, SegPath& cmap) const {
        auto cur = cmap.Head();
        AtomListIterator iter(dstSrc);
        this->trie.parse(iter, [&](size_t s, size_t e, const std::set<int64_t>* labels) -> bool {
            auto word = std::make_shared<Word>(dstSrc, s, e);
            if (labels) {
                for (auto tidx : *labels) {
                    auto tag = this->trie.getLabel(tidx);
                    if (tag) {
                        word->addLabel(tag);
                    }
                }
            }
            cur = cmap.addCell(word, cur);
            return false;
        });
    }
};
const char* DictWordRecongnizer::PB_FILE_KEY = "pbfile.path";
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

    void addSomeCells(const AtomList& dstSrc, SegPath& cmap) const {
        std::vector<std::shared_ptr<Word>> adds;
        cmap.iterRow(NULL, -1, [this, &adds](Cursor cur) {
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
        });
        if (!adds.empty()) {
            auto cur = cmap.Head();
            for (auto w : adds) {
                cur = cmap.addNext(cur, w);
            }
        }
    }
    bool exclusive() { return true; }
};

const char* PinyinRecongnizer::ENCODER_PARAM = "pyin.encoder";
REGISTER_Recognizer(PinyinRecongnizer);

/**
 * @brief time and date recongnizer
 *
 */
class DateRecongnizer : public CellRecognizer {
   public:
    void addSomeCells(const AtomList& dstSrc, SegPath& cmap) const {}
};

}  // namespace darts

#endif  // SRC_IMPL_RECOGNIZER_HPP_
