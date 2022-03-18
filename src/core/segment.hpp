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
#include <float.h>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "../core/darts.hpp"
#include "../utils/registerer.hpp"
namespace darts {

class SegmentPlugin {
   public:
    /**
     * @brief init this pulg by dict map
     *
     * @param param
     * @return int
     */
    virtual int initalize(const std::map<std::string, std::string>& param) = 0;
    virtual ~SegmentPlugin() {}
};

class CellPersenter : public SegmentPlugin {
   public:
    /***
     * 对每个Word进行向量表示
     **/
    virtual void embed(AtomList* dstSrc, CellMap* cmap) const = 0;
    /**
     * @brief
     *
     * @param pre
     * @param next
     * @return double must >=0
     */
    virtual double ranging(const Word* pre, const Word* next) const = 0;
    virtual ~CellPersenter() {}
};

class CellRecognizer : public SegmentPlugin {
   public:
    // recognizer all Wcell possable in the atomlist
    virtual void addSomeCells(AtomList* dstSrc, CellMap* cmap) const = 0;
    virtual ~CellRecognizer() {}
};

// define some registerer
REGISTER_REGISTERER(CellRecognizer);
REGISTER_REGISTERER(CellPersenter);
#define REGISTER_Recognizer(name) REGISTER_CLASS(CellRecognizer, name)
#define REGISTER_Persenter(name) REGISTER_CLASS(CellPersenter, name)
// end defined

typedef struct _GraphEdge {
    int et;
    double weight;
} * GraphEdge;

class Segment {
   private:
    std::vector<std::shared_ptr<CellRecognizer>> cellRecognizers;
    std::shared_ptr<CellPersenter> quantizer;

   public:
    /**
     * @brief  add a
     *
     * @param reg
     */
    void addRecognizer(std::shared_ptr<CellRecognizer> reg) {
        if (reg) {
            cellRecognizers.push_back(reg);
        }
    }

   private:
    /**
     * @brief build map
     *
     * @param context
     * @param cmap
     * @param graph
     */
    void buildGraph(AtomList* context, CellMap* cmap, std::map<int, std::vector<GraphEdge>*>& graph) {
        cmap->makeCurIndex();
        this->quantizer->embed(context, cmap);
        // add head
        std::vector<GraphEdge>* head_tmp = new std::vector<GraphEdge>();
        cmap->iterRow(NULL, 0, [&](Cursor pre) {
            auto dist = quantizer->ranging(NULL, pre->val.get());
            tmp->push_back(new _GraphEdge{next->idx, dist});
        });
        graph.insert(std::make_pair(-1, head_tmp));

        cmap->iterRow(NULL, -1, [&](Cursor pre) {
            std::vector<GraphEdge>* tmp = new std::vector<GraphEdge>();
            cmap->iterRow(pre, pre->val->et, [&](Cursor next) {
                auto dist = quantizer->ranging(pre->val.get(), next->val.get());
                tmp->push_back(new _GraphEdge{next->idx, dist});
            });
            // add tail
            if (tmp->empty()) {
                auto dist = quantizer->ranging(pre->val.get(), NULL);
                int cidx  = cmap->Size();
                tmp->push_back(new _GraphEdge{cidx, dist});
            }
            graph.insert(std::make_pair(pre->idx, tmp));
        });
    }

    /**
     * @brief select best path
     *
     * @param graph
     * @param bestPaths
     */
    void selectPath(std::map<int, std::vector<GraphEdge>*>& graph, std::vector<int>& bestPaths) {
        auto sz = graph.size();
        std::vector<double> dist;
        dist.assign(sz, -1.0);
        std::vector<int> prev;
        prev.assign(sz, -2);

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
            double minDist = DBL_MAX;
            int u          = 0;
            for (auto idx : visted) {
                if (used.find(idx) == used.end()) continue;
                if (dist[idx] < minDist) {
                    minDist = dist[idx];
                    u       = idx;
                }
            }
            if (u == sz - 1) break;
            used.erase(u);
            for (auto nw : *graph[u]) {
                auto node = nw->et;
                if (used.find(node) == used.end()) continue;
                auto c = dist[u] + nw->weight;
                visted.insert(node);
                if ((dist[node] < 0) || (c < dist[node])) {
                    dist[node] = c;
                    prev[node] = u;
                }
            }
        }

        bestPaths.push_back(sz - 1);
        while (bestPaths.back() > -1) {
            bestPaths.push_back(prev[bestPaths.back()]);
        }
        std::reverse(bestPaths.begin(), bestPaths.end());
    }

    /**
     * @brief 进行最优选择
     *
     * @param context
     * @param cmap
     * @param ret
     */
    void splitContent(AtomList* context, CellMap* cmap, std::vector<std::shared_ptr<Word>>& ret,
                      int atom_start_pos = 0) {
        // get best path
        std::map<int, std::vector<GraphEdge>*> graph;
        buildGraph(context, cmap, graph);
        std::vector<int> bestPaths;
        selectPath(graph, bestPaths);
        // output result
        cmap->iterRow(NULL, -1, [&](Cursor cur) {
            if (std::binary_search(bestPaths.begin(), bestPaths.end(), cur->idx)) {
                auto word = cur->val;
                word->st += atom_start_pos;
                word->et += atom_start_pos;
                ret.push_back(word);
            }
        });
        // clear memory
        bestPaths.clear();
        for (auto p : graph) {
            for (auto v : *(p.second)) {
                delete v;
            }
            p.second->clear();
            delete p.second;
        }
        graph.clear();
    }

    void buildMap(AtomList* atomList, CellMap* cmap) {
        auto cur = cmap->Head();
        // add basic cells
        for (size_t i = 0; i < atomList->size(); i++) {
            auto a = atomList->at(i);
            cur    = cmap->addNext(cur, std::make_shared<Word>(a, i, i + 1));
        }
        // add cell regnize
        for (auto recognizer : cellRecognizers) {
            recognizer->addSomeCells(atomList, cmap);
        }
    }

   public:
    explicit Segment(std::shared_ptr<CellPersenter> quantizer) {
        assert(!quantizer);  // quantizer must be not null
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
    void smartCut(AtomList* atomList, std::vector<std::shared_ptr<Word>>& ret, bool maxMode = false,
                  int atom_start_pos = 0) {
        if (!atomList || atomList->size() < 1) {
            return;
        }
        if (atomList->size() == 1) {
            auto atom = atomList->at(0);
            ret.push_back(std::make_shared<Word>(atom, atom_start_pos, atom_start_pos + 1));
            return;
        }
        auto cmap = new CellMap();
        buildMap(atomList, cmap);
        if (maxMode) {
            cmap->iterRow(NULL, -1, [&](Cursor cur) {
                auto word = cur->val;
                word->st += atom_start_pos;
                word->et += atom_start_pos;
                ret.push_back(word);
            });
        } else {
            splitContent(atomList, cmap, ret, atom_start_pos);
        }
        delete cmap;
    }
};

const std::set<std::string> SENTENCE_POS = {"!", "。", ",", "?", ";", ":", "！", "，", "？", "；", "："};

/**
 * @brief 对原始的字符串进行
 *
 * @param sg
 * @param src
 * @param ret
 * @param maxMode
 */
void tokenize(Segment& sg, const char* src, std::vector<std::shared_ptr<Word>>& ret, bool maxMode = false,
              size_t maxLineLength = 100, size_t minLineLength = 10) {
    if (!src) return;
    AtomList ori(src);
    if (ori.size() <= maxLineLength) {
        sg.smartCut(&ori, ret, maxMode);
        return;
    }
    // too long context
    size_t pre = 0;
    for (size_t pos = minLineLength; pos < ori.size(); pos++) {
        auto a = ori.at(pos);
        if (pos - pre >= maxLineLength) {
            continue;
            AtomList temp(ori, pre, pos + 1);
            sg.smartCut(&temp, ret, maxMode, pre);
            pre = pos + 1;
        }

        if (SENTENCE_POS.find(a->image) == SENTENCE_POS.end()) {
            continue;
        }
        if (pos - pre < minLineLength) {
            continue;
        }
        AtomList temp(ori, pre, pos + 1);
        sg.smartCut(&temp, ret, maxMode, pre);
        pre = pos + 1;
    }
    if (pre < ori.size()) {
        AtomList temp(ori, pre, ori.size());
        sg.smartCut(&temp, ret, maxMode, pre);
        return;
    }
}

}  // namespace darts
#endif  // SRC_CORE_SEGMENT_HPP_
