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

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "../core/darts.hpp"
namespace darts {

class CellPersenter {
   public:
    /***
     * 对每个Word进行向量表示
     **/
    virtual void embed(AtomList *dstSrc, CellMap *cmap) = 0;
    /**
     * @brief
     *
     * @param pre
     * @param next
     * @return double must >=0
     */
    virtual double ranging(Word *pre, Word *next) = 0;
    virtual ~CellPersenter() {}
};

class CellRecognizer {
   public:
    // recognizer all Wcell possable in the atomlist
    virtual void addSomeCells(AtomList *dstSrc, CellMap *cmap) = 0;
    virtual ~CellRecognizer() {}
};

typedef struct _GraphEdge {
    int st;
    int et;
    double weight;
} * GraphEdge;


class Segment {
   private:
    std::vector<std::shared_ptr<CellRecognizer>> cellRecognizers;
    std::shared_ptr<CellPersenter> quantizer;


    void buildGraph(AtomList *context, CellMap *cmap, std::map<int, std::vector<GraphEdge> *> &graph) {
        cmap->makeCurIndex();
        this->quantizer->embed(context, cmap);

        cmap->iterRow(NULL, -1, [&](Cursor pre) {
            std::vector<GraphEdge> *tmp = new std::vector<GraphEdge>();
            cmap->iterRow(pre, pre->val->et, [&](Cursor next) {
                auto dist = quantizer->ranging(pre->idx < 0 ? NULL : pre->val.get(), next->val.get());
                tmp->push_back(new _GraphEdge{pre->idx, next->idx, dist});
            });
            if (tmp->empty()) {
                auto dist = quantizer->ranging(pre->idx < 0 ? NULL : pre->val.get(), NULL);
                int cidx = cmap->Size();
                tmp->push_back(new _GraphEdge{pre->idx, cidx, dist});
            }
            graph.insert(std::make_pair(pre->idx, tmp));
        });
    }

    void selectPath(std::map<int, std::vector<GraphEdge> *> &graph, std::vector<int> &bestPaths) {
        auto sz = graph.size();
        std::vector<double> dist(sz);
        std::vector<int> prev(sz);
        for (auto i = 0; i < sz; ++i) {
            dist[i] = -1.0;
            prev[i] = -2;
        }
        std::set<int> used;
        for (auto j = 1; j < sz; j++) {
            used.insert(j);
        }
        std::set<int> visted;
        for (auto nw : *graph[-1]) {
            prev[nw->et] = -1;
            dist[nw->et] = nw->weight;
            visted.insert(nw->et);
        }

        // dijkstra
        while (used.find(sz - 1) != used.end()) {
            // minDist, u := float32(3.4e30), 0
            // used.Intersect(visted).Each(func(i interface{}) bool {
            // 	idx := i.(int)
            // 	if dist[idx] < minDist {
            // 		minDist = dist[idx]
            // 		u = idx
            // 	}
            // 	return false
            // })
            // if u == sz-1 {
            // 	break
            // }
            // used.Remove(u)
            // for _, nw := range graph[u] {
            // 	node, weight := int(nw[0]), nw[1]
            // 	if !used.Contains(node) {
            // 		continue
            // 	}
            // 	c := dist[u] + weight
            // 	visted.Add(node)
            // 	if (dist[node] < 0) || (c < dist[node]) {
            // 		dist[node] = c
            // 		prev[node] = u
            // 	}
            // }
        }


        bestPaths.push_back(sz - 1);
        while (bestPaths.back() > -1) {
            bestPaths.push_back(prev[bestPaths.back()]);
        }
        std::reverse(bestPaths.begin(), bestPaths.end());
    }


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
            auto call = [&ret](Cursor cur) {
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
