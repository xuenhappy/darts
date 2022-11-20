/*
 * File: main.hpp
 * Project: impl
 * darts工具箱主入口
 * File Created: Saturday, 11th December 2021 8:03:10 pm
 * Author: Xu En (xuen@mokar.com)
 * -----
 * Last Modified: Saturday, 11th December 2021 8:04:25 pm
 * Modified By: Xu En (xuen@mokahr.com)
 * -----
 * Copyright 2021 - 2021 Your Company, Moka
 */

#include <stdio.h>
#include <stdlib.h>

#include <iostream>

#include "core/segment.hpp"
#include "impl/jsonconf.hpp"
#include "impl/quantizer.hpp"
#include "impl/recognizer.hpp"
#include "utils/argparse.hpp"
#include "utils/codecvt.hpp"
#include "utils/dcompile.hpp"
#include "utils/dregex.hpp"
#include "utils/file_utils.hpp"
#include "utils/utils_base.hpp"

void testGetResource() {
    printf("----- test function %s-----------------\n", "testGetResource");
    std::cout << getResource("test1") << std::endl;
    std::cout << getResource("/test1") << std::endl;
    std::cout << getResource("test1/tst") << std::endl;
}

void testNormalization() {
    printf("----- test function %s-----------------\n", "testNormalization");
    std::string ori("这是一段123 ss中文测试；看】🅿Ａ,Ｂ,Ｃ,Ｄ,Ｅ,Ｆ,Ｇ,Ｈ,Ｉ,   Ｊ,Ｋ,Ｌ,Ｍ,Ｎ,Ｏ,看？ ss");
    std::string normals = normalizeStr(ori);
    std::cout << "ori: " << ori << std::endl;
    std::cout << "normal: " << normals << std::endl;
}

void testAtomListSplit() {
    printf("----- test function %s-----------------\n", "testAtomListSplit");
    std::string ori("使用zlib做压缩,先调用deflateInit(),这个函数必须在使用deflate之前.");
    std::cout << "ori: " << ori << std::endl;
    std::cout << darts::AtomList(ori) << std::endl;
}


void testDregexRW() {
    printf("----- test function %s-----------------\n", "testDregexRW");
    darts::Trie dat;
    std::cout << "write pb file ..." << std::endl;
    dat.writePb("test.pb.gz");
    std::cout << "load pb file ..." << std::endl;
    darts::Trie newdat;
    newdat.loadPb("test.pb.gz");
}

void testDregexParse() {
    printf("----- test function %s-----------------\n", "testDregexParse");
    darts::Trie newTrie;
    newTrie.loadPb(getResource("data/model-dict/mini_dict.pb.gz"));
    std::string teststr = "dd清华大学的学在北京大学的中国人民解放军海军广州舰艇学院k安徽大学里面有个北大II";
    std::cout << "ori str: " << teststr << std::endl;
    std::u32string text = to_utf32(teststr);
    dregex::U32StrIterator testStr(&text[0], text.size());
    std::vector<std::string> labels;
    newTrie.parse(testStr, [&](size_t s, size_t e, const std::set<int64_t> *label) -> bool {
        labels.clear();
        if (label) {
            for (auto lidx : *label) {
                labels.push_back(newTrie.getLabel(lidx));
            }
        }
        std::cout << to_utf8(text.substr(s, e - s)) << ",s:" << s << ",e:" << e << "," << darts::join(labels, "|")
                  << std::endl;
        return false;
    });
}

void testDictWordRecongnizer() {
    printf("----- test function %s-----------------\n", "testDictWordRecongnizer");
    darts::DictWordRecongnizer dict("mini_dict.pb.gz");
    std::cout << darts::CellRecognizerRegisterer::IsValid("DictWordRecongnizer") << std::endl;
}


void testBigramPersenterDictMake() {
    printf("----- test function %s-----------------\n", "testBigramPersenterDictMake");
    darts::BigramDict::buildDict("SogouLabDic.txt", "/Users/xuen/Downloads/SogouR.txt", "bigram");
}

void testJsonConfLoad() {
    printf("----- test function %s-----------------\n", "testJsonConfLoad");
    darts::Segment *segment = NULL;
    if (parseJsonConf(getResource("data/darts.conf.json").c_str(), &segment)) {
        std::cout << "load segment error!" << std::endl;
    } else {
        std::cout << "load segment success!" << std::endl;
    }
    if (segment) {
        delete segment;
    }
}

int main(int argc, char *argv[]) {
    initUtils();
    testGetResource();
    testNormalization();
    testAtomListSplit();
    testDregexRW();
    testDregexParse();
    testDictWordRecongnizer();
    testBigramPersenterDictMake();
    testJsonConfLoad();
    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}
