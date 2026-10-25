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
#include <array>
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

    /** 报告用：遍历全部边 (from, to, weight)；from=-1 句首，to=candidate_count 句尾 */
    void collectEdges(const std::function<void(int, int, double)>& hit) const {
        for (int u = -1; u < static_cast<int>(candidate_count); ++u) {
            for (const auto& edge : graph[static_cast<size_t>(u + 1)]) {
                hit(u, edge.et, edge.weight);
            }
        }
    }

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

// ── 切分报告结构（seg_report 特性，2026-10-25）──────────────────────
// 复用正常切分管线：select/splitContent 在 need_report 时把中间产物
// （原子/候选格/边权/最优路径）旁路采集到报告结构，零额外计算路径。
struct SegReportEdge {
    int from;     // 源节点 idx（-1=句首哨兵）
    int to;       // 目标节点 idx（size=句尾哨兵）
    double weight;
};

struct SegReportNode {
    int idx;
    std::string text;
    int st;       // atom 起点（句内）
    int et;       // atom 终点（句内，半开）
    int feat;     // 同 span 多标签副本序号（-1 起）
    std::vector<std::string> labels;
};

struct SubSegReport {
    std::string text;                              // 子句原文（规范化后）
    std::vector<std::array<std::string, 4>> atoms; // image/type/st/et
    std::vector<SegReportNode> nodes;              // 候选格（idx 0..n-1）
    std::vector<SegReportEdge> edges;              // 全部相邻边+权重
    std::vector<int> best_path;                    // 最优路径节点序列
    std::vector<std::pair<std::string, std::vector<std::string>>> outputs;
};

struct SegReport {
    std::vector<SubSegReport> subs;                // 长文切窗的每个子句
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
                      int atom_start_pos = 0, double temperature_override = -1.0,
                      SubSegReport* report = nullptr) {
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

        if (report != nullptr) {
            graph.collectEdges([&report](int f, int t, double w) {
                report->edges.push_back(SegReportEdge{f, t, w});
            });
            report->best_path = bestPaths;
            for (const auto& word : ret) {
                std::vector<std::string> labels(word->getLabels().begin(), word->getLabels().end());
                report->outputs.emplace_back(word->text(), std::move(labels));
            }
        }
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
                int atom_start_pos = 0, double temperature_override = -1.0,
                SubSegReport* report = nullptr) {
        if (atomList.size() < 1) {
            return;
        }
        // P0（2026-10-25）：裸 new 在 embed 抛异常时泄漏整条 Cursor 链——
        // 改 unique_ptr 保证异常安全
        std::unique_ptr<SegPath> cmap(new SegPath());
        buildSegPath(atomList, *cmap);
        if (decider == nullptr || maxMode) {
            auto dfunc = [&](Cursor cur) {
                auto word = cur->val;
                word->st += atom_start_pos;
                word->et += atom_start_pos;
                ret.push_back(word);
            };
            cmap->iterRow(nullptr, -1, dfunc);
            if (report != nullptr) {
                cmap->indexIt();
                for (const auto& word : ret) {
                    std::vector<std::string> labels(word->getLabels().begin(), word->getLabels().end());
                    report->outputs.emplace_back(word->text(), std::move(labels));
                }
            }
        } else {
            if (report != nullptr) {
                cmap->indexIt();  // buildGraph 内也会 indexIt（幂等）
                for (size_t i = 0; i < atomList.size(); i++) {
                    auto a = atomList.at(i);
                    report->atoms.push_back({a->image, a->char_type,
                                             std::to_string(a->st), std::to_string(a->et)});
                }
                cmap->collectCells([&report](int idx, const std::shared_ptr<Word>& w) {
                    SegReportNode node;
                    node.idx = idx;
                    node.text = w->text();
                    node.st = w->st;
                    node.et = w->et;
                    node.feat = w->feat;
                    node.labels.assign(w->getLabels().begin(), w->getLabels().end());
                    report->nodes.push_back(std::move(node));
                });
                report->text = atomList.subAtom(0, static_cast<int>(atomList.size()));
            }
            splitContent(atomList, *cmap, ret, atom_start_pos, temperature_override, report);
        }
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
                     size_t maxLineLength = 100, size_t minLineLength = 10, double temperature = -1.0,
                     SegReport* report = nullptr) {
    // 长文切窗的每个子句独立成子报告；need_report 与正常路径完全复用
    auto sub_report = [&report]() -> SubSegReport* {
        if (report == nullptr) return nullptr;
        report->subs.emplace_back();
        return &report->subs.back();
    };
    if (ori.size() <= maxLineLength) {
        sg.select(ori, ret, maxMode, 0, temperature, sub_report());
        return;
    }
    // too long context
    size_t pre = 0;
    for (size_t pos = minLineLength; pos < ori.size(); pos++) {
        if (pos - pre >= maxLineLength) {
            AtomList temp(ori, pre, pos + 1);
            sg.select(temp, ret, maxMode, pre, temperature, sub_report());
            pre = pos + 1;
            continue;
        }
        if (SENTENCE_POS.find(ori.at(pos)->image) == SENTENCE_POS.end()) continue;
        if (pos - pre < minLineLength) continue;

        AtomList temp(ori, pre, pos + 1);
        sg.select(temp, ret, maxMode, pre, temperature, sub_report());
        pre = pos + 1;
    }
    if (pre < ori.size()) {
        AtomList temp(ori, pre, ori.size());
        sg.select(temp, ret, maxMode, pre, temperature, sub_report());
        return;
    }
}

}  // namespace darts
#endif  // SRC_CORE_SEGMENT_HPP_
