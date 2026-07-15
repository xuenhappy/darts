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
#include <cassert>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "../core/core.hpp"
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
 * @brief decide which seg path for use
 *
 */
class Decider : public SegmentPlugin {
   public:
    virtual ~Decider() {}
    // embeding the words
    virtual void embed(const AtomList& dstSrc, SegPath& cmap) const = 0;
    // calculate the two word distance
    virtual double ranging(const std::shared_ptr<Word> pre, const std::shared_ptr<Word> next) const = 0;
    virtual void rangingBatch(
        const std::vector<std::pair<std::shared_ptr<Word>, std::shared_ptr<Word>>>& pairs,
        std::vector<double>& weights) const {
        weights.reserve(weights.size() + pairs.size());
        for (const auto& pair : pairs) weights.push_back(ranging(pair.first, pair.second));
    }
};

/**
 * @brief recongnize all posssable tokens
 *
 */
class CellRecognizer : public SegmentPlugin {
   public:
    // recognizer all possable word in the atomlist
    virtual void addWords(const AtomList& dstSrc, SegPath& cmap) const = 0;
    // is this plugin exclusive other plugin
    virtual bool exclusive() const { return false; }
    virtual ~CellRecognizer() {}
};

// define some registerer
REGISTER_REGISTERER(SegmentPlugin);
REGISTER_REGISTERER(CellRecognizer);
REGISTER_REGISTERER(Decider);
#define REGISTER_Service(name) REGISTER_CLASS(SegmentPlugin, name)
#define REGISTER_Recognizer(name) REGISTER_CLASS(CellRecognizer, name)
#define REGISTER_Decider(name) REGISTER_CLASS(Decider, name)
// end defined

struct GraphEdge {
    int et;
    double weight;
};

class SegGraph {
   private:
    std::vector<std::vector<GraphEdge>> graph;
    size_t candidate_count;

   public:
    explicit SegGraph(size_t candidates) : graph(candidates + 1), candidate_count(candidates) {}

    /**
     * @brief put edges
     *
     * @param st
     * @param edges
     */
    void addEdge(int st, int et, double weight) { graph[st + 1].push_back(GraphEdge{et, weight}); }

    /**
     * @brief select best path
     *
     * @param graph
     * @param bestPaths
     */
    void selectPath(std::vector<int>& bestPaths) {
        auto sz = candidate_count + 2;
        std::vector<double> dist(sz, std::numeric_limits<double>::max());
        std::vector<int> prev(sz, -2);
        dist[0] = 0;

        // Candidate ids follow start/end order, so the segmentation graph is a DAG.
        for (int u = -1; u < static_cast<int>(candidate_count); ++u) {
            if (dist[u + 1] == std::numeric_limits<double>::max()) continue;
            for (const auto& edge : graph[u + 1]) {
                int v = edge.et;
                double new_weight = dist[u + 1] + edge.weight;
                if (dist[v + 1] > new_weight) {
                    dist[v + 1] = new_weight;
                    prev[v + 1] = u;
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
    std::shared_ptr<Decider> decider;

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
        cmap.indexIt();
        this->decider->embed(context, cmap);
        struct PendingEdge {
            int from;
            int to;
            std::shared_ptr<Word> pre;
            std::shared_ptr<Word> next;
        };
        std::vector<PendingEdge> pending;
        // add head
        auto dfunc_head = [&](Cursor pre) {
            pending.push_back(PendingEdge{-1, pre->idx, cmap.SrcNode(), pre->val});
        };
        cmap.iterRow(nullptr, 0, dfunc_head);

        auto dfunc = [&](Cursor pre) {
            bool has_next = false;
            cmap.iterRow(pre, pre->val->et, [&](Cursor next) {
                pending.push_back(PendingEdge{pre->idx, next->idx, pre->val, next->val});
                has_next = true;
            });
            // add tail
            if (!has_next) {
                int cidx = cmap.Size();
                pending.push_back(PendingEdge{pre->idx, cidx, pre->val, cmap.EndNode()});
            }
        };
        cmap.iterRow(nullptr, -1, dfunc);

        std::vector<std::pair<std::shared_ptr<Word>, std::shared_ptr<Word>>> pairs;
        pairs.reserve(pending.size());
        for (const auto& edge : pending) pairs.emplace_back(edge.pre, edge.next);
        std::vector<double> weights;
        decider->rangingBatch(pairs, weights);
        assert(weights.size() == pending.size());
        for (size_t i = 0; i < pending.size(); ++i) {
            assert(weights[i] >= 0);
            graph.addEdge(pending[i].from, pending[i].to, weights[i]);
        }
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
        SegGraph graph(cmap.Size());
        buildGraph(context, cmap, graph);
        std::vector<int> bestPaths;
        graph.selectPath(bestPaths);

        // output result
        auto dfunc = [&](Cursor cur) {
            if (std::binary_search(bestPaths.begin(), bestPaths.end(), cur->idx)) {
                auto word = cur->val;
                word->st += atom_start_pos;
                word->et += atom_start_pos;
                ret.push_back(word);
            }
        };
        cmap.iterRow(nullptr, -1, dfunc);
    }

    void buildSegPath(const AtomList& atomList, SegPath& cmap) {
        auto cur = cmap.Head();
        // add basic cells
        for (size_t i = 0; i < atomList.size(); i++) {
            auto a = atomList.at(i);
            auto w = std::make_shared<Word>(a->image, i, i + 1);
            w->addLabel(a->char_type);
            cur = cmap.addNext(cur, w);
        }
        // add cell regnize
        for (auto recognizer : cellRecognizers) {
            recognizer->addWords(atomList, cmap);
        }
    }

   public:
    explicit Segment(std::shared_ptr<Decider> quantizer) { this->decider = quantizer; }
    ~Segment() {
        if (this->decider) {
            this->decider = nullptr;
        }
        cellRecognizers.clear();
    }
    /**
     * @brief select best token that split
     *
     * @param atomList
     * @param ret
     * @param maxMode
     * @param atom_start_pos
     */
    void select(const AtomList& atomList, std::vector<std::shared_ptr<Word>>& ret, bool maxMode = false,
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
        if (decider == nullptr || maxMode) {
            auto dfunc = [&](Cursor cur) {
                auto word = cur->val;
                word->st += atom_start_pos;
                word->et += atom_start_pos;
                ret.push_back(word);
            };
            cmap->iterRow(nullptr, -1, dfunc);
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
inline void tokenize(Segment& sg, const AtomList& ori, std::vector<std::shared_ptr<Word>>& ret, bool maxMode = false,
                     size_t maxLineLength = 100, size_t minLineLength = 10) {
    if (ori.size() <= maxLineLength) {
        sg.select(ori, ret, maxMode);
        return;
    }
    // too long context
    size_t pre = 0;
    for (size_t pos = minLineLength; pos < ori.size(); pos++) {
        if (pos - pre >= maxLineLength) {
            AtomList temp(ori, pre, pos + 1);
            sg.select(temp, ret, maxMode, pre);
            pre = pos + 1;
            continue;
        }
        if (SENTENCE_POS.find(ori.at(pos)->image) == SENTENCE_POS.end()) continue;
        if (pos - pre < minLineLength) continue;

        AtomList temp(ori, pre, pos + 1);
        sg.select(temp, ret, maxMode, pre);
        pre = pos + 1;
    }
    if (pre < ori.size()) {
        AtomList temp(ori, pre, ori.size());
        sg.select(temp, ret, maxMode, pre);
        return;
    }
}

}  // namespace darts
#endif  // SRC_CORE_SEGMENT_HPP_
