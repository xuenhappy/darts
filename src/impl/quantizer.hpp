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

REGISTER_Persenter(MinCoverDecider);

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
            float pidx = this->ngdict.getWordKey(w->text());
            if (pidx < 0) {
                if (type_encoder == nullptr) {
                    pidx = this->ngdict.getWordKey(w->maxHlabel(nullptr));
                } else {
                    auto hunc = std::bind(&LabelEncoder::labelH, this->type_encoder, std::placeholders::_1);
                    pidx      = this->ngdict.getWordKey(w->maxHlabel(hunc));
                }
            }
            w->setAtt(std::shared_ptr<std::vector<float>>(new std::vector<float>{pidx}));
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
        if (pre != nullptr && !pre->isStSpecial() && pre->getAtt() != nullptr) {
            pidx = (int)(*(pre->getAtt()))[0];
        }
        int nidx = -1;
        if (next != nullptr && !next->isEtSpecial() && next->getAtt() != nullptr) {
            nidx = (int)(*(next->getAtt()))[0];
        }
        return ngdict.wordDist(pidx, nidx);
    }

    ~BigramDecider() {}
};
const char* BigramDecider::DAT_PATH_KEY   = "dat.path";
const char* BigramDecider::TYPE_ENC_PARAM = "type.enc";
REGISTER_Persenter(BigramDecider);

}  // namespace darts

#endif  // SRC_IMPL_QUANTIZER_HPP_
