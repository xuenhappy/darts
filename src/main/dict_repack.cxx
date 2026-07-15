#include <cstdlib>
#include <iostream>
#include <string>

#include "../utils/dregex.hpp"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: darts-dict-repack INPUT.pbs OUTPUT.pbs" << std::endl;
        return EXIT_FAILURE;
    }
    if (std::string(argv[1]) == argv[2]) {
        std::cerr << "input and output paths must differ" << std::endl;
        return EXIT_FAILURE;
    }
    dregex::Trie trie;
    if (trie.loadPb(argv[1]) != EXIT_SUCCESS) return EXIT_FAILURE;
    if (trie.writePb(argv[2]) != EXIT_SUCCESS) return EXIT_FAILURE;
    return EXIT_SUCCESS;
}
