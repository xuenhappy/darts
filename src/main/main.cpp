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

#ifndef SRC_MAIN_MAIN_HPP_
#define SRC_MAIN_MAIN_HPP_

#include "../core/segment.hpp"
#include "../utils/argparse.hpp"
#include "../utils/dcompile.hpp"
#include "../utils/dregex.hpp"
#include "../utils/file_utils.hpp"
#include "../utils/utils_base.hpp"

int main(int argc, char *argv[]) {
    initUtils();
    std::cout << "----------------0----------------" << std::endl;
    std::cout << getResource("test1") << std::endl;
    std::cout << getResource("/test1") << std::endl;
    std::cout << getResource("test1/tst") << std::endl;
    std::cout << "----------------1----------------" << std::endl;
    std::string ori("这是一段123 ss中文测试；看】🅿Ａ,Ｂ,Ｃ,Ｄ,Ｅ,Ｆ,Ｇ,Ｈ,Ｉ,   Ｊ,Ｋ,Ｌ,Ｍ,Ｎ,Ｏ,看？ ss");
    std::string normals = normalizeStr(ori);
    std::cout << "ori: " << ori << std::endl;
    std::cout << "normal: " << normals << std::endl;
    std::cout << "---------------2-----------------" << std::endl;
    std::cout << darts::AtomList(normals) << std::endl;
    std::cout << "---------------3-----------------" << std::endl;
    darts::Trie dat;
    std::cout << "write pb file ..." << std::endl;
    dat.writePb("test.pb");
    std::cout << "load pb file ..." << std::endl;
    darts::Trie newdat;
    newdat.loadPb("test.pb");
    std::cout << "----------------4----------------" << std::endl;
    darts::Trie schoolsdat;
    std::vector<std::string> paths = {"/Users/xuen/school_zh.txt"};
    dregex::compileStringDict(paths, "schools.ph", NULL);
    std::cout << "----------------5----------------" << std::endl;
    argparse::ArgumentParser program("darts");
    program.add_argument("square").help("display the square of a given integer").scan<'i', int>();

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error &err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    auto input = program.get<int>("square");
    std::cout << (input * input) << std::endl;

    return 0;
}

#endif  // SRC_MAIN_MAIN_HPP_
