/*
 * File: hmm.hpp
 * Project: utils
 * File Created: Tuesday, 6th December 2022 1:15:11 pm
 * Author: Xu En (xuen@mokar.com)
 * -----
 * Last Modified: Tuesday, 6th December 2022 1:15:17 pm
 * Modified By: Xu En (xuen@mokahr.com)
 * -----
 * Copyright 2021 - 2022 Your Company, Moka
 */

#ifndef __HMM__H__
#define __HMM__H__

#include <cstddef>
#include <vector>

/// \param

inline void viterbi_decode(size_t tagnum, const double** trans, const std::vector<double[]>& props,
                           std::vector<size_t>& seq) {
    seq.resize(props.size(), 0);
    size_t length = props.size();

    double prob[length][tagnum];
    size_t prevs[length][tagnum];
    for (int i = 0; i < tagnum; i++) {
        prob[0][i] = props[i][1];
    }

    for (int i = 1; i < length; i++) {
        for (int j = 0; j < tagnum; j++) {
            double pmax = 0, p;
            int dmax;
            for (int k = 0; k < tagnum; k++) {
                p = prob[i - 1][k] * trans[k][j];
                if (p > pmax) {
                    pmax = p;
                    dmax = k;
                }
            }
            prob[i][j]      = props[j][1] * pmax;
            prevs[i - 1][j] = dmax;
        }
    }

    double pmax = 0;
    int dmax;
    for (int i = 0; i < tagnum; i++) {
        if (prob[length - 1][i] > pmax) {
            pmax = prob[length - 1][i];
            dmax = i;
        }
    }
    seq[length - 1] = dmax;

    for (int i = length - 2; i >= 0; i--) {
        seq[i] = prevs[i][seq[i + 1]];
    }
}

#endif  //!__HMM__H__
