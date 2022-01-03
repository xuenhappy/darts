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


#include "../core/segment.hpp"
#include "../impl/quantizer.hpp"
#include "../impl/recognizer.hpp"
#include "../utils/argparse.hpp"
#include "../utils/codecvt.hpp"
#include "../utils/dcompile.hpp"
#include "../utils/dregex.hpp"
#include "../utils/file_utils.hpp"
#include "../utils/utils_base.hpp"

int main(int argc, char *argv[]) {
    initUtils();
    argparse::ArgumentParser program("darts");
    program.add_argument("--compile")
        .help("does need compile a trie, switch")
        .default_value(false)
        .implicit_value(true)
        .nargs(0);
    program.add_argument("-f", "--input_files").help("The list of input files used for build trie").remaining();
    program.add_argument("-o", "--output_file")
        .help("trie pb file output dir")
        .default_value(std::string("build_dregex.pb.gz"));

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error &err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }
    try {
        if (program["--compile"] == true) {
            auto files = program.get<std::vector<std::string>>("--input_files");
            auto outfile = program.get<std::string>("--output_file");
            if (!files.empty()) {
                std::cout << "compile files to trie [" << darts::join(files, " , ") << "] " << std::endl;
                std::set<WordType> skipType = {WordType::POS, WordType::EMPTY, WordType::NLINE};
                dregex::compileStringDict(files, outfile, &skipType);
                std::cout << "build tire success, pb out file path:" << outfile << std::endl;
            }
        }
    } catch (std::logic_error &e) {
        std::cout << program << std::endl;
    }
    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}
