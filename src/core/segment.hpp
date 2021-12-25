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
#include "../core/darts.hpp"

namespace darts {

class CellPersenter {
   public:
    /***
     * 对每个Word进行向量表示
     **/
    virtual void embed(const AtomList *dstSrc, CellMap *cmap) = 0;
    virtual void ranging(const Word *pre, const Word *next) = 0;
    virtual ~CellPersenter() {}
};

class CellRecognizer {
   public:
    // recognizer all Wcell possable in the atomlist
    virtual void addSomeCells(const AtomList *dstSrc, CellMap *cmap) = 0;
    virtual ~CellRecognizer() {}
};

}  // namespace darts
#endif  // SRC_CORE_SEGMENT_HPP_
