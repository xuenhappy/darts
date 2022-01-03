/*
 * File: atomtype.h
 * Project: core
 * File Created: Saturday, 25th December 2021 1:21:40 pm
 * Author: Xu En (xuen@mokar.com)
 * -----
 * Last Modified: Saturday, 25th December 2021 1:22:05 pm
 * Modified By: Xu En (xuen@mokahr.com)
 * -----
 * Copyright 2021 - 2021 Your Company, Moka
 */

#ifndef SRC_CORE_WTYPE_H_
#define SRC_CORE_WTYPE_H_
#include <algorithm>
#include <cstring>
#include <string>

#include "../utils/magic_enum.hpp"


enum class WordType {
    NONE,     // defalut wordtype for init
    WUNK,     // words unknown word
    TOKEN,    // common token
    NLINE,    // space but \n in
    POS,      // some common pos
    MASK,     // for mask use
    CLS,      // for cls use
    NUM,      // number str
    ENG,      // english str
    CJK,      // cjk str
    EMPTY,    // empty str
    SYMBOLS,  // some symbols
    FACE,     // some face
    ARROW,    // some arrow words
    RUSH,     // rush str
    ORDER,    // order str
    IP,       // ip str
    CNUM,     // chinese word num
    KEYW,     // some keyword str
    TIME,     // time string
    MOTH,     // moth str
    DAY,      // day str
    EMAIL,    // email str
    URL,      // url str
    TMS,      // single timerange str
    TMR,      // timerange str
    NAME,     // name str
    LOC,      // location str
    CAM,
    ACD,
    SCH,
    MAJ,
    COM,
    TIL,
    DPT
};

/**
 * @brief wordtype to string
 *
 * @param wordtype
 * @return std::string
 */
std::string wordType2str(WordType wordtype) {
    auto _name = magic_enum::enum_name(wordtype);
    return std::string(_name);
}

/**
 * @brief string to wordtype
 *
 * @param name
 * @return WordType
 */
WordType str2wordType(const std::string &name) {
    auto wordtype = magic_enum::enum_cast<WordType>(name);
    if (wordtype.has_value()) {
        return wordtype.value();
    }
    return WordType::NONE;
}


#endif  // SRC_CORE_WTYPE_H_
