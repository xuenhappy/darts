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
enum WordType {
    NONE,
    WUNK,
    POS,
    NUM,
    ENG,
    CJK,
    EMPTY,
    SYMBOLS,
    FACE,
    ARROW,
    RUSH,
    ORDER,
    IP,
    CNUM,
    KEYW,
    TIME,
    MOTH,
    DAY,
    EMAIL,
    URL,
    TMS,
    TMR,
    NAME,
    LOC,
    CAM,
    ACD,
    SCH,
    MAJ,
    COM,
    TIL,
    DPT,
};

static const char* EnumStrings[] = {
    "NONE",  "WUNK", "POS",  "NUM",  "ENG",  "CJK",  "EMPTY", "SYMBOLS", "FACE", "ARROW", "RUSH",
    "ORDER", "IP",  "CNUM", "KEYW", "TIME", "MOTH", "DAY",   "EMAIL",   "URL",  "TMS",   "TMR",
    "NAME",  "LOC", "CAM",  "ACD",  "SCH",  "MAJ",  "COM",   "TIL",     "DPT",
};

const char* getWordTypeText(int enumVal) { return EnumStrings[enumVal]; }

WordType getWordType(const char* name) {
    int n = sizeof(EnumStrings) / sizeof(EnumStrings[0]);
    size_t i = 0;
    while (i < n) {
        if (strcmp(EnumStrings[i], name) == 0) {
            return WordType(i);
        }
        i++;
    }
    return WordType::NONE;
}


#endif  // SRC_CORE_WTYPE_H_
