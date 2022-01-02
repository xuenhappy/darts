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
class KQNetMeasure {
   public:
    /**
     * @brief 加载模型文件
     *
     * @param model_path
     * @return int
     */
    int loadModel(const char *model_path) { return EXIT_SUCCESS; }
    /**
     * @brief 利用神经网络计算两个词间距
     *
     * @param pre
     * @param next
     * @return double
     */
    double ranging(const darts::Word *pre, const darts::Word *next) const { return 0.0; }
};


#endif  // SRC_IMPL_NETWORDQI_HPP_
