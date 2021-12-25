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

#include "../utils/argparse.hpp"
#include "../utils/file_utils.hpp"
int main(int argc, char *argv[]) {
    std::cout << getResource("test1") << std::endl;
    std::cout << getResource("/test1") << std::endl;
    std::cout << getResource("test1/tst") << std::endl;

    argparse::ArgumentParser program("darts");
    program.add_argument("square")
        .help("display the square of a given integer")
        .scan<'i', int>();

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
