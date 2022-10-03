/*
 * File: networdqi.hpp
 * Project: impl
 * File Created: Sunday, 2nd January 2022 10:06:07 am
 * Author: Xu En (xuen@mokar.com)
 * -----
 * Last Modified: Sunday, 2nd January 2022 10:06:10 am
 * Modified By: Xu En (xuen@mokahr.com)
 * -----
 * Copyright 2021 - 2022 Your Company, Moka
 */
#ifndef SRC_IMPL_NETWORDQI_HPP_
#define SRC_IMPL_NETWORDQI_HPP_
#include "../core/darts.hpp"
namespace darts {

/**
 * @brief 利用ONNX神经网络对数据进行表示与
 *
 */
class OnnxPersenter : public CellPersenter {
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
    int initalize(const std::map<std::string, std::string>& param) { return EXIT_SUCCESS; }
    /**
     * @brief set all word embeding
     *
     * @param dstSrc
     * @param cmap
     */
    void embed(AtomList* dstSrc, SegPath* cmap) const {}
    /**
     * @brief
     *
     * @param pre
     * @param next
     * @return double must >=0
     */
    double ranging(const Word* pre, const Word* next) const { return 0.0; }
    ~OnnxPersenter() {}
};

const char* OnnxPersenter::MODEL_PATH_KEY = "model.path";
REGISTER_Persenter(OnnxPersenter);
}  // namespace darts
#endif  // SRC_IMPL_NETWORDQI_HPP_
