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
#include "../utils/str_utils.hpp"

namespace darts {
class MinCoverPersenter {
   public:
    int initalize(const std::map<std::string, std::string> &param) { return EXIT_SUCCESS; }
    void embed(AtomList *dstSrc, CellMap *cmap) {}
    /**
     * @brief
     *
     * @param pre
     * @param next
     * @return double must >=0
     */
    double ranging(const Word *pre, const Word *next) {
        double len_a = (pre == NULL) ? 0.0 : 100.0 / (1.0 + pre->word->image.length());
        double len_b = (next == NULL) ? 0.0 : 100.0 / (1.0 + next->word->image.length());
        return len_a + len_b;
    }
    ~MinCoverPersenter() {}
};

REGISTER_Persenter(MinCoverPersenter);

class BigramPersenter {
   public:
    int initalize(const std::map<std::string, std::string> &param) { return EXIT_SUCCESS; }
    void embed(AtomList *dstSrc, CellMap *cmap) {}
    /**
     * @brief
     *
     * @param pre
     * @param next
     * @return double must >=0
     */
    double ranging(const Word *pre, const Word *next) { return 0.0; }
    ~BigramPersenter() {}
};

REGISTER_Persenter(BigramPersenter);

class ElmoPersenter {
   private:
    static const char *MODEL_PATH_KEY;

   public:
    /**
     * @brief init this
     *
     * @param param
     * @return int
     */
    int initalize(const std::map<std::string, std::string> &param) { return EXIT_SUCCESS; }
    /**
     * @brief set all word embeding
     *
     * @param dstSrc
     * @param cmap
     */
    void embed(AtomList *dstSrc, CellMap *cmap) {}
    /**
     * @brief
     *
     * @param pre
     * @param next
     * @return double must >=0
     */
    double ranging(const Word *pre, const Word *next) { return 0.0; }
    ~ElmoPersenter() {}
};

const char *ElmoPersenter::MODEL_PATH_KEY = "model.path";
REGISTER_Persenter(ElmoPersenter);


class TinyBertPersenter {
   private:
    static const char *MODEL_PATH_KEY;

   public:
    /**
     * @brief init this
     *
     * @param param
     * @return int
     */
    int initalize(const std::map<std::string, std::string> &param) { return EXIT_SUCCESS; }
    /**
     * @brief set all word embeding
     *
     * @param dstSrc
     * @param cmap
     */
    void embed(AtomList *dstSrc, CellMap *cmap) {}
    /**
     * @brief
     *
     * @param pre
     * @param next
     * @return double must >=0
     */
    double ranging(const Word *pre, const Word *next) { return 0.0; }
    ~TinyBertPersenter() {}
};

const char *TinyBertPersenter::MODEL_PATH_KEY = "model.path";
REGISTER_Persenter(TinyBertPersenter);

}  // namespace darts

#endif  // SRC_IMPL_QUANTIZER_HPP_
