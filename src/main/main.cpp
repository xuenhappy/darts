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

#include <functional>

#include "../core/segment.hpp"
#include "../impl/quantizer.hpp"
#include "../impl/recognizer.hpp"
#include "../utils/argparse.hpp"
#include "../utils/codecvt.hpp"
#include "../utils/dcompile.hpp"
#include "../utils/dregex.hpp"
#include "../utils/file_utils.hpp"
#include "../utils/utils_base.hpp"

int main(int argc, char **argv) {
    std::map<std::string, std::function<void()>> functions;
    // dregex-build
    argparse::ArgumentParser program1("dregex-build");
    program1.add_argument("files").help("The list of input files used for build trie").remaining();
    program1.add_argument("-o").help("trie pb file output path").default_value(std::string("build_dregex.pb.gz"));

    functions[program1.pname()] = [&]() {
        try {
            program1.parse_args(argc - 1, argv + 1);
            auto files = program1.get<std::vector<std::string>>("files");
            auto outfile = program1.get<std::string>("-o");
            if (!files.empty()) {
                std::cout << "compile files to trie [" << darts::join(files, " , ") << "] " << std::endl;
                std::set<WordType> skipType = {WordType::POS, WordType::EMPTY, WordType::NLINE};
                if (!dregex::compileStringDict(files, outfile, &skipType)) {
                    std::cout << "build tire success, pb out file path:" << outfile << std::endl;
                }
            }
        } catch (const std::runtime_error &err) {
            std::cerr << err.what() << std::endl;
            std::cerr << program1;
            std::exit(1);
        } catch (std::logic_error &e) {
            std::cout << program1 << std::endl;
        }
    };

    // gram-build
    argparse::ArgumentParser program2("ngram-build");
    program2.add_argument("-f").help("freq file and the union gram freq file,[freq file,union freq file]").nargs(2);
    program2.add_argument("-o").help("output dir").default_value(std::string("ngram"));

    functions[program2.pname()] = [&]() {
        try {
            program2.parse_args(argc - 1, argv + 1);
            auto files = program2.get<std::vector<std::string>>("-f");
            auto outdir = program2.get<std::string>("-o");
            if (files.size() == 2) {
                if (darts::BigramPersenter::buildDict(files[0], files[1], outdir)) {
                    std::cout << "compile ngram files to dir " << outdir << std::endl;
                }
            }
        } catch (const std::runtime_error &err) {
            std::cerr << err.what() << std::endl;
            std::cerr << program2;
            std::exit(1);
        } catch (std::logic_error &e) {
            std::cout << program2 << std::endl;
        }
    };
    // tokenize
    argparse::ArgumentParser program3("tokenize");


    // check function
    std::vector<std::string> func_names;
    for (auto &kv : functions) {
        func_names.push_back(kv.first);
    }

    if (argc < 2) {
        std::cout << argv[0] << " [" << darts::join(func_names, "|") << "]" << std::endl;
        exit(0);
    }
    auto it = functions.find(std::string(argv[1]));
    if (it == functions.end()) {
        std::cout << argv[0] << " [" << darts::join(func_names, "|") << "]" << std::endl;
        exit(1);
    }
    // call function
    initUtils();
    it->second();
    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}
