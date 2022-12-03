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

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "../core/segment.hpp"
#include "../utils/biggram.hpp"
#include "../utils/file_utils.hpp"

namespace darts {
class MinCoverPersenter : public CellPersenter {
   public:
    int initalize(const std::map<std::string, std::string>& params,
                  std::map<std::string, std::shared_ptr<SegmentPlugin>>& plugins) {
        // pass
        return EXIT_SUCCESS;
    }

    /**
     * @brief do nothing
     *
     * @param dstSrc
     * @param cmap
     */
    void embed(const AtomList& dstSrc, SegPath& cmap) const {}

    /**
     * @brief
     *
     * @param pre
     * @param next
     * @return double must >=0
     */
    double ranging(const Word* pre, const Word* next) const {
        double len_a = (pre == NULL) ? 0.0 : 100.0 / (1.0 + pre->text().length());
        double len_b = (next == NULL) ? 0.0 : 100.0 / (1.0 + next->text().length());
        return len_a + len_b;
    }
    ~MinCoverPersenter() {}
};

REGISTER_Persenter(MinCoverPersenter);

/**
 * @brief this use cedar store thing
 *
 */
class BigramPersenter : public CellPersenter {
   private:
    static const char* DAT_DIR_KEY;
    BigramDict ngdict;

   public:
    int initalize(const std::map<std::string, std::string>& params,
                  std::map<std::string, std::shared_ptr<SegmentPlugin>>& dicts) {
        auto it = params.find(DAT_DIR_KEY);
        if (it == params.end() || it->second.empty()) {
            std::cerr << "ERROR: could not find key:" << DAT_DIR_KEY << std::endl;
            return EXIT_FAILURE;
        }
        std::string path = getResource(it->second, true);
        return ngdict.loadDict(path);
    }
    /**
     * @brief do nothing
     *
     * @param dstSrc
     * @param cmap
     */
    void embed(const AtomList& dstSrc, SegPath& cmap) const {}

    /**
     * @brief
     *
     * @param pre
     * @param next
     * @return double must >=0
     */
    double ranging(const Word* pre, const Word* next) const {
        return ngdict.wordDist(pre == NULL ? NULL : pre->text().c_str(), next == NULL ? NULL : next->text().c_str());
    }

    ~BigramPersenter() {}
};
const char* BigramPersenter::DAT_DIR_KEY = "dat.dir";
REGISTER_Persenter(BigramPersenter);

class ElmoPersenter : public CellPersenter {
   private:
    static const char* MODEL_PATH_KEY;
    static const char* CHAR_TABLE_FILE_KEY;

   public:
    /**
     * @brief init this
     *
     * @param param
     * @return int
     */
    int initalize(const std::map<std::string, std::string>& params,
                  std::map<std::string, std::shared_ptr<SegmentPlugin>>& plugins) {
        return EXIT_SUCCESS;
    }
    /**
     * @brief set all word embeding
     *
     * @param dstSrc
     * @param cmap
     */
    void embed(const AtomList& dstSrc, SegPath& cmap) const {}
    /**
     * @brief
     *
     * @param pre
     * @param next
     * @return double must >=0
     */
    double ranging(const Word* pre, const Word* next) const { return 0.0; }
    ~ElmoPersenter() {}
};

const char* ElmoPersenter::MODEL_PATH_KEY = "model.path";
REGISTER_Persenter(ElmoPersenter);

class TinyBertPersenter : public CellPersenter {
   private:
    static const char* MODEL_PATH_KEY;

   public:
    /**
     * @brief init this
     *
     * @param param
     * @return int
     */
    int initalize(const std::map<std::string, std::string>& params,
                  std::map<std::string, std::shared_ptr<SegmentPlugin>>& plugins) {
        return EXIT_SUCCESS;
    }
    /**
     * @brief set all word embeding
     *
     * @param dstSrc
     * @param cmap
     */
    void embed(const AtomList& dstSrc, SegPath& cmap) const {}
    /**
     * @brief
     *
     * @param pre
     * @param next
     * @return double must >=0
     */
    double ranging(const Word* pre, const Word* next) const { return 0.0; }
    ~TinyBertPersenter() {}
};

const char* TinyBertPersenter::MODEL_PATH_KEY = "model.path";
REGISTER_Persenter(TinyBertPersenter);

}  // namespace darts

#endif  // SRC_IMPL_QUANTIZER_HPP_
