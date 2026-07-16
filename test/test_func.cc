#include <cstdlib>
#include <clocale>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "core/core.hpp"
#include "core/segment.hpp"
#include "impl/confparser.hpp"
#include "utils/dcompile.hpp"
#include "utils/dregex.hpp"
#include "utils/utill.hpp"

namespace {

int failures = 0;

#define CHECK(...)                                                                              \
    do {                                                                                        \
        if (!(__VA_ARGS__)) {                                                                    \
            std::cerr << __FILE__ << ':' << __LINE__ << ": check failed: " #__VA_ARGS__ << '\n'; \
            ++failures;                                                                         \
        }                                                                                       \
    } while (false)

class VectorStringIter : public dregex::StringIter {
   public:
    explicit VectorStringIter(const std::vector<std::string>& values) : values_(values) {}

    void walks(std::function<bool(const std::string&, size_t)> hit) const override {
        for (size_t i = 0; i < values_.size(); ++i)
            if (hit(values_[i], i)) return;
    }

   private:
    const std::vector<std::string>& values_;
};

class DictionaryFixture : public dregex::KvPairsIter {
   public:
    using Entry = std::pair<std::vector<std::string>, std::vector<std::string>>;

    explicit DictionaryFixture(std::vector<Entry> entries) : entries_(std::move(entries)) {}

    int iter(std::function<void(const dregex::StringIter&, const char**, size_t)> hit) const override {
        for (const auto& entry : entries_) {
            VectorStringIter key(entry.first);
            std::vector<const char*> labels;
            labels.reserve(entry.second.size());
            for (const auto& label : entry.second) labels.push_back(label.c_str());
            hit(key, labels.data(), labels.size());
        }
        return EXIT_SUCCESS;
    }

   private:
    std::vector<Entry> entries_;
};

std::filesystem::path temporaryDictionary() {
    return std::filesystem::temp_directory_path() / "darts-core-test.pbs";
}

void testAtomList() {
    darts::AtomList atoms("中文ABC123");
    CHECK(atoms.size() == 4);
    if (atoms.size() >= 4) {
        CHECK(atoms.at(0)->image == "中");
        CHECK(atoms.at(2)->image == "ABC");
        CHECK(atoms.at(3)->image == "123");
    }
    CHECK(atoms.subAtom(1, 4) == "文ABC123");
    CHECK(atoms.subAtom(4, 4).empty());

    darts::AtomList slice(atoms, 1, 4);
    CHECK(slice.size() == 3);
    CHECK(slice.subAtom(0, 3) == "文ABC123");
}

void testNormalization() {
    std::string output = "stale";
    normalize("ＡＢＣ１２３", output);
    CHECK(output == "ABC123");
    normalize("中文", output);
    CHECK(output == "中文");
}

void testCoreLanguageData() {
    CHECK(charType(0x3042) == char_type::WUNK);  // Hiragana is not CJK ideographic text.
    CHECK(charType(0xD55C) == char_type::WUNK);  // Hangul is not CJK ideographic text.
    CHECK(charType(0x20000) == char_type::CJK);
    auto middle = pinyin("中");
    CHECK(middle != nullptr);
    if (middle) {
        CHECK(!middle->piyins.empty());
        CHECK(middle->piyins.front() == "zhōng");
        for (const auto& reading : middle->piyins) CHECK(reading.find('#') == std::string::npos);
    }
}

void testGraphSelection() {
    darts::SegGraph graph(3);
    graph.addEdge(-1, 0, 1.0);
    graph.addEdge(0, 3, 1.0);
    graph.addEdge(-1, 1, 0.5);
    graph.addEdge(1, 2, 2.0);
    graph.addEdge(2, 3, 0.5);
    std::vector<int> path;
    graph.selectPath(path);
    CHECK(path == std::vector<int>({0}));

    std::mt19937_64 near_zero_rng(7);
    graph.selectPath(path, 0.0, near_zero_rng);
    CHECK(path == std::vector<int>({0}));
}

void testGraphBoltzmannSampling() {
    // Path {0} costs 2, path {1, 2} costs 3. At T=1 the first path has
    // probability exp(-2)/(exp(-2)+exp(-3)) = 0.731...
    darts::SegGraph graph(3);
    graph.addEdge(-1, 0, 1.0);
    graph.addEdge(0, 3, 1.0);
    graph.addEdge(-1, 1, 0.5);
    graph.addEdge(1, 2, 2.0);
    graph.addEdge(2, 3, 0.5);

    std::mt19937_64 first_rng(42);
    std::mt19937_64 second_rng(42);
    for (int i = 0; i < 100; ++i) {
        std::vector<int> first;
        std::vector<int> second;
        graph.selectPath(first, 0.7, first_rng);
        graph.selectPath(second, 0.7, second_rng);
        CHECK(first == second);
    }

    std::mt19937_64 distribution_rng(1234);
    int shortest_count = 0;
    constexpr int samples = 20000;
    for (int i = 0; i < samples; ++i) {
        std::vector<int> path;
        graph.selectPath(path, 1.0, distribution_rng);
        if (path == std::vector<int>({0})) ++shortest_count;
        else CHECK(path == std::vector<int>({1, 2}));
    }
    const double observed = static_cast<double>(shortest_count) / samples;
    CHECK(std::abs(observed - 0.7310585786) < 0.02);
}

void testBigramUnknownCost() {
    darts::BigramDict dict;
    CHECK(dict.loadDict("data/models/ngram_dict.bdf") == EXIT_SUCCESS);
    CHECK(dict.wordDist(-1, -1) > 0.0);
}

void testDictionaryRoundTrip() {
    DictionaryFixture fixture({{{"中", "文"}, {"LANG"}},
                               {{"文", "分", "词"}, {"NLP", "TECH"}},
                               {{"abc", "123"}, {"MIXED"}}});
    const auto path = temporaryDictionary();
    {
        dregex::Trie trie;
        CHECK(dregex::compile(fixture, trie) == EXIT_SUCCESS);
        CHECK(trie.writePb(path.string()) == EXIT_SUCCESS);
    }
    {
        dregex::Trie trie;
        CHECK(trie.loadPb(path.string()) == EXIT_SUCCESS);
        const std::vector<std::string> tokens = {"中", "文", "分", "词", "ABC", "123"};
        std::vector<std::pair<size_t, size_t>> ranges;
        trie.parseContiguous(tokens.size(), [&tokens](size_t index) -> const std::string& { return tokens[index]; },
                             [&ranges](size_t start, size_t end, const std::vector<int64_t>* labels) {
                                 CHECK(labels != nullptr);
                                 ranges.emplace_back(start, end);
                                 return false;
                             });
        CHECK(ranges == std::vector<std::pair<size_t, size_t>>({{0, 2}, {1, 4}, {4, 6}}));
    }
    std::error_code error;
    std::filesystem::remove(path, error);
}

void testConfiguredSegment() {
    darts::Segment* segment = nullptr;
    CHECK(loadSegment("data/conf.json", &segment, "faster") == EXIT_SUCCESS);
    CHECK(segment != nullptr);
    if (!segment) return;

    darts::AtomList atoms("目标检测模型量化和中文分词测试");
    std::vector<std::shared_ptr<darts::Word>> words;
    segment->select(atoms, words);
    CHECK(!words.empty());
    CHECK(words.front()->st == 0);
    CHECK(words.back()->et == static_cast<int>(atoms.size()));
    for (size_t i = 1; i < words.size(); ++i) CHECK(words[i - 1]->et == words[i]->st);
    delete segment;
}

}  // namespace

int main() {
    std::setlocale(LC_ALL, "");
    initUtils();
    testAtomList();
    testNormalization();
    testCoreLanguageData();
    testGraphSelection();
    testGraphBoltzmannSampling();
    testBigramUnknownCost();
    testDictionaryRoundTrip();
    testConfiguredSegment();
    if (failures != 0) {
        std::cerr << failures << " test assertion(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "all C++ tests passed\n";
    return EXIT_SUCCESS;
}
