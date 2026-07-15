/*
 * File: quantizer.hpp
 * Project: impl
 * File Created: Saturday, 25th December 2021 11:05:37 pm
 * Author: Xu En (xuen@mokar.com)
 * -----
 * Last Modified: Saturday, 1st January 2022 7:40:19 pm
 * Modified By: Xu En (xuen@mokahr.com)
 * -----
 * Copyright 2021 - 2022 Your Company, Moka
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
