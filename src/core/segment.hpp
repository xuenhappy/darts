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
#include <queue>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "../core/darts.hpp"
#include "../utils/registerer.hpp"
namespace darts {

/**
 * @brief segment plugin for segment
 *
 */
class SegmentPlugin {
   public:
    /**
     * @brief init this pulg by dict map
     *
     * @param param
     * @return int
     */
    virtual int initalize(const std::map<std::string, std::string>& params,
                          std::map<std::string, std::shared_ptr<SegmentPlugin>>& plugins) = 0;
    virtual ~SegmentPlugin() {}
};
/**
 * @brief embeding atomlist for segment
 *
 */
class CellPersenter : public SegmentPlugin {
   public:
    /***
     * 对每个Word进行向量表示
     **/
    virtual void embed(const AtomList& dstSrc, SegPath& cmap) const = 0;
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
/**
 * @brief recongnize tokens
 *
 */
class CellRecognizer : public SegmentPlugin {
   public:
    // recognizer all Wcell possable in the atomlist
    virtual void addSomeCells(const AtomList& dstSrc, SegPath& cmap) const = 0;
    // is this plugin exclusive other plugin
    bool exclusive() { return false; }
    virtual ~CellRecognizer() {}
};

// define some registerer
REGISTER_REGISTERER(SegmentPlugin);
REGISTER_REGISTERER(CellRecognizer);
REGISTER_REGISTERER(CellPersenter);
#define REGISTER_Service(name) REGISTER_CLASS(SegmentPlugin, name)
#define REGISTER_Recognizer(name) REGISTER_CLASS(CellRecognizer, name)
#define REGISTER_Persenter(name) REGISTER_CLASS(CellPersenter, name)
// end defined

typedef struct _GraphEdge {
    int et;
    double weight;
} * GraphEdge;

class SegGraph {
   private:
    std::map<int, std::vector<GraphEdge>*> graph;

   public:
    SegGraph() {}
    ~SegGraph() {
        for (auto p : graph) {
            for (auto v : *(p.second)) {
                delete v;
            }
            p.second->clear();
            delete p.second;
        }
        graph.clear();
    }

    /**
     * @brief put edges
     *
     * @param st
     * @param edges
     */
    void putEdges(int st, std::vector<GraphEdge>* edges) { graph.insert(std::make_pair(st, edges)); }

    /**
     * @brief select best path
     *
     * @param graph
     * @param bestPaths
     */
    void selectPath(std::vector<int>& bestPaths) {
        auto sz = graph.size() + 1;
        std::vector<double> dist(sz, std::numeric_limits<double>::max());
        std::vector<int> prev(sz, -2);
        using iPair = std::pair<double, int>;
        std::priority_queue<iPair, std::vector<iPair>, std::greater<iPair>> pq;
        // init var
        pq.push(std::make_pair(0.0, -1));
        dist[0] = 0;

        // dijkstra
        while (!pq.empty()) {
            int u = pq.top().second;
            pq.pop();
            for (auto nw : *graph[u]) {
                int v             = nw->et;
                double new_weight = dist[u + 1] + nw->weight;
                if (dist[v + 1] > new_weight) {
                    dist[v + 1] = new_weight;
                    prev[v + 1] = u;
                    pq.push(std::make_pair(new_weight, v));
                }
            }
        }
        int pre = prev[prev.size() - 1];
        while (pre > -1) {
            bestPaths.emplace_back(pre);
            pre = prev[pre + 1];
        }
        std::reverse(bestPaths.begin(), bestPaths.end());
    }
};

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
            if (!cellRecognizers.empty() && reg->exclusive()) {
                std::cerr
                    << "WARN: this segment add a exclusive plugin but cellRecognizers not empty!please check conf file"
                    << std::endl;
            }
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
    void buildGraph(const AtomList& context, SegPath& cmap, SegGraph& graph) {
        cmap.makeCurIndex();
        this->quantizer->embed(context, cmap);
        // add head
        std::vector<GraphEdge>* head_tmp = new std::vector<GraphEdge>();
        cmap.iterRow(NULL, 0, [&](Cursor pre) {
            auto dist = quantizer->ranging(NULL, pre->val.get());
            head_tmp->push_back(new _GraphEdge{pre->idx, dist});
        });
        graph.putEdges(-1, head_tmp);

        cmap.iterRow(NULL, -1, [&](Cursor pre) {
            std::vector<GraphEdge>* tmp = new std::vector<GraphEdge>();
            cmap.iterRow(pre, pre->val->et, [&](Cursor next) {
                auto dist = quantizer->ranging(pre->val.get(), next->val.get());
                tmp->push_back(new _GraphEdge{next->idx, dist});
            });
            // add tail
            if (tmp->empty()) {
                auto dist = quantizer->ranging(pre->val.get(), NULL);
                int cidx  = cmap.Size();
                tmp->push_back(new _GraphEdge{cidx, dist});
            }
            graph.putEdges(pre->idx, tmp);
        });
    }

    /**
     * @brief 进行最优选择
     *
     * @param context
     * @param cmap
     * @param ret
     */
    void splitContent(const AtomList& context, SegPath& cmap, std::vector<std::shared_ptr<Word>>& ret,
                      int atom_start_pos = 0) {
        // get best path
        SegGraph graph;
        buildGraph(context, cmap, graph);
        std::vector<int> bestPaths;
        graph.selectPath(bestPaths);

        // output result
        cmap.iterRow(NULL, -1, [&](Cursor cur) {
            if (std::binary_search(bestPaths.begin(), bestPaths.end(), cur->idx)) {
                auto word = cur->val;
                word->st += atom_start_pos;
                word->et += atom_start_pos;
                ret.push_back(word);
            }
        });
    }

    void buildSegPath(const AtomList& atomList, SegPath& cmap) {
        auto cur = cmap.Head();
        // add basic cells
        for (size_t i = 0; i < atomList.size(); i++) {
            auto a = atomList.at(i);
            cur    = cmap.addNext(cur, std::make_shared<Word>(a->image, i, i + 1));
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
    void smartCut(const AtomList& atomList, std::vector<std::shared_ptr<Word>>& ret, bool maxMode = false,
                  int atom_start_pos = 0) {
        if (atomList.size() < 1) {
            return;
        }
        if (atomList.size() == 1) {
            auto atom = atomList.at(0);
            ret.push_back(std::make_shared<Word>(atom->image, atom_start_pos, atom_start_pos + 1));
            return;
        }
        auto cmap = new SegPath();
        buildSegPath(atomList, *cmap);
        if (maxMode) {
            cmap->iterRow(NULL, -1, [&](Cursor cur) {
                auto word = cur->val;
                word->st += atom_start_pos;
                word->et += atom_start_pos;
                ret.push_back(word);
            });
        } else {
            splitContent(atomList, *cmap, ret, atom_start_pos);
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
void tokenize(Segment& sg, const AtomList& ori, std::vector<std::shared_ptr<Word>>& ret, bool maxMode = false,
              size_t maxLineLength = 100, size_t minLineLength = 10) {
    if (ori.size() <= maxLineLength) {
        sg.smartCut(ori, ret, maxMode);
        return;
    }
    // too long context
    size_t pre = 0;
    for (size_t pos = minLineLength; pos < ori.size(); pos++) {
        auto a = ori.at(pos);
        if (pos - pre >= maxLineLength) {
            continue;
            AtomList temp(ori, pre, pos + 1);
            sg.smartCut(temp, ret, maxMode, pre);
            pre = pos + 1;
        }

        if (SENTENCE_POS.find(a->image) == SENTENCE_POS.end()) {
            continue;
        }
        if (pos - pre < minLineLength) {
            continue;
        }
        AtomList temp(ori, pre, pos + 1);
        sg.smartCut(temp, ret, maxMode, pre);
        pre = pos + 1;
    }
    if (pre < ori.size()) {
        AtomList temp(ori, pre, ori.size());
        sg.smartCut(temp, ret, maxMode, pre);
        return;
    }
}

}  // namespace darts
#endif  // SRC_CORE_SEGMENT_HPP_
