/*
 * File: Segment.hpp
 * Project: core
 * File Created: Saturday, 11th December 2021 8:12:54 pm
 * Author: Xu En (xuen@mokar.com)
 * -----
 * Last Modified: Saturday, 18th December 2021 7:39:14 pm
 * Modified By: Xu En (xuen@mokahr.com)
 * -----
 * Copyright 2021 - 2021 Your Company, Moka
 */

#ifndef SRC_CORE_SEGMENT_HPP_
#define SRC_CORE_SEGMENT_HPP_

#include <memory>
#include <vector>

#include "../core/darts.hpp"
namespace darts {

class CellPersenter {
   public:
    /***
     * 对每个Word进行向量表示
     **/
    virtual void embed(AtomList *dstSrc, CellMap *cmap) = 0;
    virtual double ranging(Word *pre, Word *next) = 0;
    virtual ~CellPersenter() {}
};

class CellRecognizer {
   public:
    // recognizer all Wcell possable in the atomlist
    virtual void addSomeCells(AtomList *dstSrc, CellMap *cmap) = 0;
    virtual ~CellRecognizer() {}
};


class Segment {
   private:
    std::vector<std::shared_ptr<CellRecognizer>> cellRecognizers;
    std::shared_ptr<CellPersenter> quantizer;


    void splitContent(AtomList *context, CellMap *cmap, std::vector<std::shared_ptr<Word>> &ret) {
    }

    void buildMap(AtomList *atomList, CellMap *cmap) {
        auto cur = cmap->Head();
        // add basic cells
        for (size_t i = 0; i < atomList->size(); i++) {
            auto a = atomList->at(i);
            cur = cmap->addNext(cur, std::make_shared<Word>(a, i, i + 1));
        }
        // add cell regnize
        for (auto recognizer : cellRecognizers) {
            recognizer->addSomeCells(atomList, cmap);
        }
    }

   public:
    explicit Segment(std::shared_ptr<CellPersenter> quantizer) {
        this->quantizer = quantizer;
    }
    ~Segment() {
        if (this->quantizer) {
            this->quantizer = nullptr;
        }
        auto iter = cellRecognizers.begin();
        while (iter != cellRecognizers.end()) {
            iter = cellRecognizers.erase(iter);
        }
        cellRecognizers.clear();
    }
    /**
     * @brief 对数据进行切分
     *
     * @param atomList
     * @param ret
     * @param maxMode
     */
    void smartCut(AtomList *atomList, std::vector<std::shared_ptr<Word>> &ret, bool maxMode) {
        if (atomList->size() < 1) {
            return;
        }
        if (atomList->size() == 1) {
            auto atom = atomList->at(0);
            ret.push_back(std::make_shared<Word>(atom, 0, 1));
            return;
        }
        auto cmap = new CellMap();
        buildMap(atomList, cmap);
        if (maxMode) {
            auto call = [&ret](Cursor *cur) {
                ret.push_back(cur->val);
            };
            cmap->iterRow(NULL, -1, call);
        } else {
            splitContent(atomList, cmap, ret);
        }
        delete cmap;
    }
};

}  // namespace darts
#endif  // SRC_CORE_SEGMENT_HPP_
