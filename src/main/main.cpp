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
#include "../utils/codecvt.hpp"
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
    std::set<WordType> skipTypes = {WordType::POS, WordType::EMPTY, WordType::NLINE};
    dregex::compileStringDict({"/Users/xuen/school_zh.txt"}, "schools.pb", &skipTypes);
    darts::Trie newTrie;
    newTrie.loadPb("schools.pb");
    std::string teststr = "dd清华大学的学在北京大学的中国人民解放军海军广州舰艇学院k安徽大学里面有个北大II";
    std::u32string text = to_utf32(teststr);

    dregex::UTF8StrIterator testStr(&text[0], text.size());
    newTrie.parse(testStr, [&](size_t s, size_t e, const std::vector<int64_t> *label) -> bool {
        std::cout << to_utf8(text.substr(s, e - s)) << ",s:" << s << ",e:" << e << "," << (*label)[0] << "|"
                  << newTrie.getLabel((*label)[0]) << std::endl;
        return false;
    });
    std::cout << "----------------5----------------" << std::endl;

    argparse::ArgumentParser program("darts");
    program.add_argument("--compile")
        .help("does need compile a trie, switch")
        .default_value(false)
        .implicit_value(false);
    program.add_argument("-f", "--input_files").help("The list of input files used for build trie");
    program.add_argument("-o", "--output_file").help("trie pb file output dir").default_value("build_trie.pb");

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error &err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }
    if (program["--compile"] == true) {
        auto files = program.get<std::vector<std::string>>("--input_files");
        auto outfile = program.get<std::string>("--output_file");
        if (outfile.empty()) {
            outfile = "build_trie.pb";
        }
        if (!files.empty()) {
            std::cout << "compile  files to trie:\n" << darts::join(files, "\n") << std::endl;
            std::set<WordType> skipType = {WordType::POS, WordType::EMPTY, WordType::NLINE};
            dregex::compileStringDict(files, outfile, &skipType);
        }
    } else {
        std::cout << program << std::endl;
    }
    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}

#endif  // SRC_MAIN_MAIN_HPP_
