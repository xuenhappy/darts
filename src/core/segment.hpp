/*
 * File: Segment.hpp
 * Project: core
 * File Created: Saturday, 11th December 2021 8:12:54 pm
 * Author: Xu En (nanhangxuen@163.com)
 * -----
 * Last Modified: Saturday, 18th December 2021 7:39:14 pm
 * Modified By: Xu En (nanhangxuen@163.com)
 * -----
 * Copyright 2021 - 2021 XuEn
 */

#ifndef SRC_CORE_SEGMENT_HPP_
#define SRC_CORE_SEGMENT_HPP_
#include <float.h>
#include <cassert>
#include <cmath>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
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
    static constexpr double DETERMINISTIC_TEMPERATURE = 1e-8;

    static double logAdd(double lhs, double rhs) {
        if (lhs == -std::numeric_limits<double>::infinity()) return rhs;
        if (rhs == -std::numeric_limits<double>::infinity()) return lhs;
        const double upper = std::max(lhs, rhs);
        return upper + std::log(std::exp(lhs - upper) + std::exp(rhs - upper));
    }

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
    void selectPath(std::vector<int>& bestPaths) const {
        bestPaths.clear();
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
        if (prev.back() == -2) return;
        int pre = prev[prev.size() - 1];
        while (pre > -1) {
            bestPaths.emplace_back(pre);
            pre = prev[pre + 1];
        }
        std::reverse(bestPaths.begin(), bestPaths.end());
    }

    /**
     * Sample a complete path with probability proportional to exp(-cost / temperature).
     *
     * The graph is already topologically ordered by candidate id. A single
     * reverse pass computes the log partition of every suffix; sampling then
     * only visits edges on the selected path. This is O(V + E) time and O(V)
     * memory, without enumerating complete paths.
     */
    template <typename RandomEngine>
    void selectPath(std::vector<int>& sampled_path, double temperature, RandomEngine& rng) const {
        if (!(temperature > DETERMINISTIC_TEMPERATURE) || !std::isfinite(temperature)) {
            selectPath(sampled_path);
            return;
        }

        const size_t terminal = candidate_count + 1;
        const double unreachable = -std::numeric_limits<double>::infinity();
        std::vector<double> suffix_log_partition(candidate_count + 2, unreachable);
        suffix_log_partition[terminal] = 0.0;

        for (int node = static_cast<int>(candidate_count) - 1; node >= -1; --node) {
            double value = unreachable;
            for (const auto& edge : graph[node + 1]) {
                const size_t next = static_cast<size_t>(edge.et + 1);
                if (next >= suffix_log_partition.size() || suffix_log_partition[next] == unreachable) continue;
                value = logAdd(value, -edge.weight / temperature + suffix_log_partition[next]);
            }
            suffix_log_partition[node + 1] = value;
        }

        sampled_path.clear();
        if (suffix_log_partition[0] == unreachable) return;
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        int current = -1;
        while (current < static_cast<int>(candidate_count)) {
            const auto& edges = graph[current + 1];
            const double normalizer = suffix_log_partition[current + 1];
            double draw = uniform(rng);
            const GraphEdge* fallback = nullptr;
            for (const auto& edge : edges) {
                const size_t next = static_cast<size_t>(edge.et + 1);
                if (next >= suffix_log_partition.size() || suffix_log_partition[next] == unreachable) continue;
                fallback = &edge;
                draw -= std::exp(-edge.weight / temperature + suffix_log_partition[next] - normalizer);
                if (draw <= 0.0) {
                    fallback = &edge;
                    break;
                }
            }
            // Floating-point rounding can leave a tiny positive residual.
            if (!fallback) {
                sampled_path.clear();
                return;
            }
            current = fallback->et;
            if (current >= 0 && current < static_cast<int>(candidate_count)) sampled_path.push_back(current);
        }
    }
};

class Segment {
   private:
    std::vector<std::shared_ptr<CellRecognizer>> cellRecognizers;
    std::shared_ptr<Decider> decider;
    double temperature = 0.0;
    std::mt19937_64 random_engine{std::random_device{}()};
    std::mutex random_mutex;

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
                      int atom_start_pos = 0, double temperature_override = -1.0) {
        // get best path
        SegGraph graph(cmap.Size());
        buildGraph(context, cmap, graph);
        std::vector<int> bestPaths;
        const double selected_temperature = temperature_override >= 0.0 ? temperature_override : temperature;
        if (selected_temperature > 0.0) {
            std::lock_guard<std::mutex> lock(random_mutex);
            graph.selectPath(bestPaths, selected_temperature, random_engine);
        } else {
            graph.selectPath(bestPaths);
        }

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
    void setTemperature(double value) { temperature = std::isfinite(value) && value > 0.0 ? value : 0.0; }
    double getTemperature() const { return temperature; }
    void setRandomSeed(uint64_t seed) {
        std::lock_guard<std::mutex> lock(random_mutex);
        random_engine.seed(seed);
    }
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
                int atom_start_pos = 0, double temperature_override = -1.0) {
        if (atomList.size() < 1) {
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
            splitContent(atomList, *cmap, ret, atom_start_pos, temperature_override);
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
                     size_t maxLineLength = 100, size_t minLineLength = 10, double temperature = -1.0) {
    if (ori.size() <= maxLineLength) {
        sg.select(ori, ret, maxMode, 0, temperature);
        return;
    }
    // too long context
    size_t pre = 0;
    for (size_t pos = minLineLength; pos < ori.size(); pos++) {
        if (pos - pre >= maxLineLength) {
            AtomList temp(ori, pre, pos + 1);
            sg.select(temp, ret, maxMode, pre, temperature);
            pre = pos + 1;
            continue;
        }
        if (SENTENCE_POS.find(ori.at(pos)->image) == SENTENCE_POS.end()) continue;
        if (pos - pre < minLineLength) continue;

        AtomList temp(ori, pre, pos + 1);
        sg.select(temp, ret, maxMode, pre, temperature);
        pre = pos + 1;
    }
    if (pre < ori.size()) {
        AtomList temp(ori, pre, ori.size());
        sg.select(temp, ret, maxMode, pre, temperature);
        return;
    }
}

}  // namespace darts
#endif  // SRC_CORE_SEGMENT_HPP_
