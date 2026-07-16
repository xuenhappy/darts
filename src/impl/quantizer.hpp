/*
 * File: quantizer.hpp
 * Project: impl
 * File Created: Saturday, 25th December 2021 11:05:37 pm
 * Author: Xu En (nanhangxuen@163.com)
 * -----
 * Last Modified: Saturday, 1st January 2022 7:40:19 pm
 * Modified By: Xu En (nanhangxuen@163.com)
 * -----
 * Copyright 2021 - 2022 XuEn
 */
#ifndef SRC_IMPL_QUANTIZER_HPP_
#define SRC_IMPL_QUANTIZER_HPP_

#include <cstdlib>
#include <cmath>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "../core/segment.hpp"
#include "../utils/biggram.hpp"
#include "../utils/filetool.hpp"
#include "./encoder.hpp"

namespace darts {
class MinCoverDecider : public Decider {
   public:
    int initalize(const std::map<std::string, std::string>& params,
                  std::map<std::string, std::shared_ptr<SegmentPlugin>>& plugins) {
        // do nothing
        return EXIT_SUCCESS;
    }

    void embed(const AtomList& dstSrc, SegPath& cmap) const {}

    /**
     * @brief
     *
     * @param pre
     * @param next
     * @return double must >=0
     */
    double ranging(const std::shared_ptr<Word> pre, const std::shared_ptr<Word> next) const {
        double lena = 0.0, lenb = 0.0;
        if (pre != nullptr && !pre->isStSpecial()) {
            lena = 100.0 / (1.0 + pre->et - pre->st);
        }
        if (next != nullptr && !next->isEtSpecial()) {
            lenb = 100.0 / (1.0 + next->et - next->st);
        }
        return lena + lenb;
    }
    ~MinCoverDecider() {}
};

REGISTER_Decider(MinCoverDecider);

class HybridStatDecider : public Decider {
   private:
    BigramDict ngdict;
    double bigram_weight = 1.0;
    double length_weight = 100.0;
    double length_power = 1.0;
    double token_penalty = 0.0;
    double unknown_penalty = 0.0;

    static double readNonNegative(const std::map<std::string, std::string>& params, const char* key,
                                  double fallback) {
        auto it = params.find(key);
        if (it == params.end()) return fallback;
        try {
            double value = std::stod(it->second);
            return std::isfinite(value) && value >= 0.0 ? value : fallback;
        } catch (const std::exception&) {
            return fallback;
        }
    }

    double wordCost(const std::shared_ptr<Word>& word) const {
        if (!word || word->isStSpecial() || word->isEtSpecial()) return 0.0;
        const double length = std::max(1, word->et - word->st);
        double cost = token_penalty + length_weight / std::pow(length, length_power);
        if (word->vocab_id < 0) cost += unknown_penalty;
        return cost;
    }

   public:
    int initalize(const std::map<std::string, std::string>& params,
                  std::map<std::string, std::shared_ptr<SegmentPlugin>>& plugins) override {
        auto path = params.find("dat.path");
        if (path == params.end() || path->second.empty() || ngdict.loadDict(getResource(path->second))) {
            std::cerr << "ERROR: HybridStatDecider requires a valid dat.path" << std::endl;
            return EXIT_FAILURE;
        }
        bigram_weight = readNonNegative(params, "bigram.weight", bigram_weight);
        length_weight = readNonNegative(params, "length.weight", length_weight);
        length_power = readNonNegative(params, "length.power", length_power);
        token_penalty = readNonNegative(params, "token.penalty", token_penalty);
        unknown_penalty = readNonNegative(params, "unknown.penalty", unknown_penalty);
        return EXIT_SUCCESS;
    }

    void embed(const AtomList& dstSrc, SegPath& cmap) const override {
        cmap.iterRow(nullptr, -1, [this](Cursor cur) { cur->val->vocab_id = ngdict.getWordKey(cur->val->text()); });
    }

    double ranging(const std::shared_ptr<Word> pre, const std::shared_ptr<Word> next) const override {
        const int pre_idx = pre && !pre->isStSpecial() ? pre->vocab_id : -1;
        const int next_idx = next && !next->isEtSpecial() ? next->vocab_id : -1;
        return bigram_weight * ngdict.wordDist(pre_idx, next_idx) + wordCost(pre) + wordCost(next);
    }
};

REGISTER_Decider(HybridStatDecider);

/** Address-specific transition quantizer with probability-derived costs. */
class AddressDecider : public Decider {
   private:
    enum class Role { START, PROVINCE, CITY, DISTRICT, STREET, ROAD, COMPONENT, POI, UNKNOWN, END };
    BigramDict dictionary;
    double lexical_weight = 0.15;

    static Role roleOf(const std::shared_ptr<Word>& word) {
        if (!word) return Role::UNKNOWN;
        if (word->isStSpecial()) return Role::START;
        if (word->isEtSpecial()) return Role::END;
        Role result = Role::UNKNOWN;
        for (const auto& label : word->getLabels()) {
            if (label == "ADDR_PROVINCE") return Role::PROVINCE;
            if (label == "ADDR_CITY") return Role::CITY;
            if (label == "ADDR_DISTRICT") return Role::DISTRICT;
            if (label == "ADDR_STREET") return Role::STREET;
            if (label == "ADDR_ROAD") result = Role::ROAD;
            else if (label == "ADDR_COMPONENT") result = Role::COMPONENT;
            else if (label == "ADDR_POI") result = Role::POI;
            else if (label == "ADDR_DIVISION" && result == Role::UNKNOWN) result = Role::DISTRICT;
        }
        return result;
    }

    static double probability(Role left, Role right) {
        if (left == Role::START) {
            if (right == Role::PROVINCE || right == Role::CITY || right == Role::DISTRICT) return 0.70;
            if (right == Role::STREET || right == Role::ROAD || right == Role::POI) return 0.20;
            return 0.05;
        }
        if (right == Role::END) {
            if (left == Role::POI || left == Role::COMPONENT || left == Role::ROAD) return 0.75;
            if (left == Role::STREET || left == Role::DISTRICT) return 0.40;
            return 0.10;
        }
        if (left == Role::PROVINCE && right == Role::CITY) return 0.90;
        if (left == Role::PROVINCE && right == Role::DISTRICT) return 0.35;
        if (left == Role::CITY && right == Role::DISTRICT) return 0.85;
        if (left == Role::CITY && (right == Role::STREET || right == Role::ROAD)) return 0.25;
        if (left == Role::DISTRICT && right == Role::STREET) return 0.70;
        if (left == Role::DISTRICT && right == Role::ROAD) return 0.55;
        if (left == Role::DISTRICT && right == Role::POI) return 0.30;
        if (left == Role::STREET && right == Role::ROAD) return 0.65;
        if (left == Role::STREET && right == Role::POI) return 0.45;
        if (left == Role::ROAD && right == Role::COMPONENT) return 0.85;
        if (left == Role::ROAD && right == Role::POI) return 0.55;
        if (left == Role::COMPONENT && right == Role::POI) return 0.50;
        if (left == Role::POI && right == Role::COMPONENT) return 0.20;
        if (left == right && left != Role::UNKNOWN) return 0.12;
        if (left == Role::UNKNOWN || right == Role::UNKNOWN) return 0.08;
        return 0.015;
    }

    static double ruleNll(const std::shared_ptr<Word>& word) {
        if (!word || word->isStSpecial() || word->isEtSpecial()) return 0.0;
        bool is_rule = false;
        bool is_division = false;
        for (const auto& label : word->getLabels()) {
            if (label == "ADDR_RULE") is_rule = true;
            if (label == "ADDR_DIVISION") is_division = true;
        }
        if (!is_rule) return 0.0;
        const double length = static_cast<double>(word->et - word->st);
        // P(rule candidate is a real component) starts at exp(-0.35) and
        // decreases for unusually long suffix matches. This preserves OOV
        // recall without letting one rule span swallow known hierarchy nodes.
        if (is_division) return 0.75 + std::max(0.0, length - 3.0) * 1.25;
        return 0.35 + std::max(0.0, length - 5.0) * 0.9;
    }

   public:
    int initalize(const std::map<std::string, std::string>& params,
                  std::map<std::string, std::shared_ptr<SegmentPlugin>>&) override {
        auto path = params.find("dat.path");
        if (path == params.end() || dictionary.loadDict(getResource(path->second))) return EXIT_FAILURE;
        auto weight = params.find("lexical.weight");
        if (weight != params.end()) {
            try {
                const double parsed = std::stod(weight->second);
                if (std::isfinite(parsed) && parsed >= 0.0) lexical_weight = parsed;
            } catch (const std::exception&) {
                return EXIT_FAILURE;
            }
        }
        return EXIT_SUCCESS;
    }

    void embed(const AtomList&, SegPath& path) const override {
        path.iterRow(nullptr, -1, [this](Cursor cursor) {
            cursor->val->vocab_id = dictionary.getWordKey(cursor->val->text());
        });
    }

    double ranging(const std::shared_ptr<Word> previous,
                   const std::shared_ptr<Word> next) const override {
        // This is an actual negative log probability, not an arbitrary score.
        const double role_nll = -std::log(probability(roleOf(previous), roleOf(next)));
        const int previous_id = previous && !previous->isStSpecial() ? previous->vocab_id : -1;
        const int next_id = next && !next->isEtSpecial() ? next->vocab_id : -1;
        return role_nll + ruleNll(next) + lexical_weight * dictionary.wordDist(previous_id, next_id);
    }
};
REGISTER_Decider(AddressDecider);

/**
 * @brief this use cedar store thing
 *
 */
class BigramDecider : public Decider {
   private:
    static const char* DAT_PATH_KEY;
    static const char* TYPE_ENC_PARAM;
    BigramDict ngdict;
    std::shared_ptr<LabelEncoder> type_encoder;

   public:
    int initalize(const std::map<std::string, std::string>& params,
                  std::map<std::string, std::shared_ptr<SegmentPlugin>>& dicts) {
        auto it = params.find(DAT_PATH_KEY);
        if (it == params.end() || it->second.empty()) {
            std::cerr << "ERROR: could not find key:" << DAT_PATH_KEY << std::endl;
            return EXIT_FAILURE;
        }
        std::string dictfile = getResource(it->second);
        if (ngdict.loadDict(dictfile)) return EXIT_FAILURE;
        auto pit = dicts.find(TYPE_ENC_PARAM);
        if (pit != dicts.end()) {
            type_encoder = std::dynamic_pointer_cast<LabelEncoder>(pit->second);
            if (type_encoder == nullptr) {
                std::cerr << "ERROR:value not type LabelEncoder " << TYPE_ENC_PARAM << std::endl;
                return EXIT_FAILURE;
            }
        }

        return EXIT_SUCCESS;
    }
    /**
     * @brief set word idx
     *
     * @param dstSrc
     * @param cmap
     */
    void embed(const AtomList& dstSrc, SegPath& cmap) const {
        auto dfunc = [this](Cursor cur) {
            auto w     = cur->val;
            int pidx = this->ngdict.getWordKey(w->text());
            if (pidx < 0) {
                if (type_encoder == nullptr) {
                    pidx = this->ngdict.getWordKey(w->maxHlabel(nullptr));
                } else {
                    auto hunc = std::bind(&LabelEncoder::labelH, this->type_encoder, std::placeholders::_1);
                    pidx      = this->ngdict.getWordKey(w->maxHlabel(hunc));
                }
            }
            w->vocab_id = pidx;
        };
        cmap.iterRow(nullptr, -1, dfunc);
    }

    /**
     * @brief
     *
     * @param pre
     * @param next
     * @return double must >=0
     */
    double ranging(const std::shared_ptr<Word> pre, const std::shared_ptr<Word> next) const {
        int pidx = -1;
        if (pre != nullptr && !pre->isStSpecial()) {
            pidx = pre->vocab_id;
        }
        int nidx = -1;
        if (next != nullptr && !next->isEtSpecial()) {
            nidx = next->vocab_id;
        }
        return ngdict.wordDist(pidx, nidx);
    }

    ~BigramDecider() {}
};
const char* BigramDecider::DAT_PATH_KEY   = "dat.path";
const char* BigramDecider::TYPE_ENC_PARAM = "type.enc";
REGISTER_Decider(BigramDecider);

}  // namespace darts

#endif  // SRC_IMPL_QUANTIZER_HPP_
