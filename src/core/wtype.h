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
#include <string>
enum WordType { UNK, POS, SYMBOL, CJK, LOC, COMPANY };

static const char* EnumStrings[] = {"People", "Tree",   "Car",   "Text",
                                    "Cave",   "QRcode", "Pillar"};

const char* getWordTypeText(int enumVal) { return EnumStrings[enumVal]; }

#endif  // SRC_CORE_WTYPE_H_
