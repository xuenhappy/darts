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
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "../core/segment.hpp"
#include "../utils/dregex.hpp"
#include "../utils/str_utils.hpp"


namespace darts {
class AtomListStrIterator : public StringIter {
   private:
    AtomList *m_list;
    std::set<WordType> skiptypes;

   public:
    explicit AtomListStrIterator(AtomList *m_list) {
        this->m_list = m_list;
        this->skiptypes.insert(WordType::EMPTY);
    }


    void iter(std::function<bool(const std::string &, size_t)> hit) {
        std::string tmp;
        for (auto i = 0; i < m_list->size(); i++) {
            auto a = m_list->at(i);
            if (a->hasType(&skiptypes)) {
                continue;
            }
            tmp = a->image;
            if (hit(tolower(tmp), i)) {
                break;
            }
        }
    }
};
class DictWordRecongnizer : public CellRecognizer {
   private:
    Trie trie;
    static const char *PB_FILE_KEY;

   public:
    DictWordRecongnizer() {}
    explicit DictWordRecongnizer(const std::string &pbfile) { this->trie.loadPb(pbfile); }

    int initalize(const std::map<std::string, std::string> &param) {
        auto iter = param.find(PB_FILE_KEY);
        if (iter == param.end()) {
            std::cerr << PB_FILE_KEY << " key not found in dictionary!" << std::endl;
            return EXIT_FAILURE;
        }
        return this->trie.loadPb(iter->second);
    }

    void addSomeCells(AtomList *dstSrc, CellMap *cmap) const {
        auto cur = cmap->Head();
        AtomListStrIterator iter(dstSrc);
        this->trie.parse(iter, [&](size_t s, size_t e, const std::vector<int64_t> *labels) -> bool {
            auto word = std::make_shared<Word>(dstSrc->at(s, e), s, e);
            if (labels) {
                for (auto tidx : *labels) {
                    auto tag = this->trie.getLabel(tidx);
                    if (!tag) continue;
                    auto h = getWordType(tag);
                    if (h == WordType::NONE) continue;
                    word->addTag(h);
                }
            }
            cur = cmap->addCell(word, cur);
            return false;
        });
    }
};
const char *DictWordRecongnizer::PB_FILE_KEY = "pbfile.path";
// add this pulg
REGISTER_Recognizer(DictWordRecongnizer);

}  // namespace darts


#endif  // SRC_IMPL_RECOGNIZER_HPP_
