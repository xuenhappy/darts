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
#include <cstddef>
#include <iostream>
#include "core/segment.hpp"
#include "impl/confparser.hpp"
#include "impl/quantizer.hpp"
#include "impl/recognizer.hpp"
#include "utils/codecvt.hpp"
#include "utils/dcompile.hpp"
#include "utils/dregex.hpp"
#include "utils/filetool.hpp"
#include "utils/utill.hpp"

void testGetResource() {
    printf("----- test function %s-----------------\n", "testGetResource");
    std::cout << getResource("test1") << std::endl;
    std::cout << getResource("/test1") << std::endl;
    std::cout << getResource("test1/tst") << std::endl;
}

void testNormalization() {
    printf("----- test function %s-----------------\n", "testNormalization");
    std::string ori("这是一段123 ss中文测试；看】🅿Ａ,Ｂ,Ｃ,Ｄ,Ｅ,Ｆ,Ｇ,Ｈ,Ｉ,   Ｊ,Ｋ,Ｌ,Ｍ,Ｎ,Ｏ,看？ ss");
    std::string normals;
    normalize(ori, normals);
    std::cout << "ori: " << ori << std::endl;
    std::cout << "normal: " << normals << std::endl;
}

void testAtomListSplit() {
    printf("----- test function %s-----------------\n", "testAtomListSplit");
    std::string ori("使用zlib做压缩,先调用deflateInit(),这个函数必须在使用deflate之前.");
    std::cout << "ori: " << ori << std::endl;
    std::cout << darts::AtomList(ori) << std::endl;
}

void testJsonConfLoad() {
    printf("----- test function %s-----------------\n", "testJsonConfLoad");
    darts::Segment* segment = NULL;
    if (loadSegment(getResource("data/conf.json").c_str(), &segment)) {
        std::cout << "load segment error!" << std::endl;
    } else {
        std::cout << "load segment success!" << std::endl;
    }
    if (segment) {
        std::string ori("目标检测 yolov5模型量化安装教程以及转ONXX，torchscript，engine和速度比较一栏表!");
        std::cout << "ori: " << ori << std::endl;
        darts::AtomList alist(ori);
        std::cout << alist << std::endl;
        std::vector<std::shared_ptr<darts::Word>> ret;
        segment->select(alist, ret, true);
        for (size_t i = 0; i < ret.size(); i++) {
            std::cout << ret[i] << ",";
        }
        std::cout << std::endl;
        ret.clear();
        delete segment;
    }
}

int main(int argc, char* argv[]) {
    std::locale::global(std::locale("en_US.UTF-8"));
    setlocale(LC_ALL, "zh_CN.UTF-8");
    initUtils();
    testGetResource();
    testNormalization();
    testAtomListSplit();
    testJsonConfLoad();
    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}
