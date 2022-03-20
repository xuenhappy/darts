/*
 * File: inet.hpp
 * Project: nets
 * File Created: Monday, 17th January 2022 11:48:11 am
 * Author: Xu En (xuen@mokar.com)
 * -----
 * Last Modified: Sunday, 20th March 2022 1:52:28 pm
 * Modified By: Xu En (xuen@mokahr.com)
 * -----
 * Copyright 2021 - 2022 Your Company, Moka
 */
#ifndef SRC_NETS_INET_HPP_
#define SRC_NETS_INET_HPP_
#include <Eigen/Dense>

class Linear {
   private:
    Eigen::MatrixXf w;
    Eigen::VectorXf b;

   public:


    Eigen::Matrix call(Eigen::Matrix input) { 
        return input * w + b; }
}

#endif  // SRC_NETS_INET_HPP_
