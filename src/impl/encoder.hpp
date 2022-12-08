/*
 * File: bpe_model.hpp
 * Project: impl
 * File Created: Saturday, 24th September 2022 3:17:35 pm
 * Author: dell (xuen@mokar.com)
 * this module is use for word pice and new word regnizer
 * -----
 * Last Modified: Saturday, 24th September 2022 3:17:39 pm
 * Modified By: dell (xuen@mokahr.com)
 * -----
 * Copyright 2021 - 2022 Your Company, Moka
 */
#ifndef SRC_IMPL_ENCODER_HPP_
#define SRC_IMPL_ENCODER_HPP_

#include <fmt/core.h>
#include <algorithm>
#include <functional>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "../core/darts.hpp"
#include "../core/segment.hpp"
#include "../utils/biggram.hpp"
#include "../utils/pinyin.hpp"
#include "../utils/str_utils.hpp"

struct symbol_pair {
    int left;      // left index of this pair
    int right;     // right index of this pair
    double score;  // score of this pair. small is better
};

struct symbol {
    int start;
    int size;
    int idx;
};
inline bool symbol_compare_func(symbol& i1, symbol& i2) {
    return (i1.start < i2.start) || (i1.start == i2.start && i1.size < i2.size);
}

class WordGraph {
   private:
    std::vector<symbol> nodes;       // nodes
    size_t path_nums;                // path nums
    std::vector<symbol_pair> paths;  // paths

    void addPath(int left, int right, double weight) {
        if (path_nums >= paths.size()) {
            paths.resize(paths.size() * 2 + 1);
        }

        auto* h  = &paths[path_nums++];
        h->left  = left;
        h->right = right;
        h->score = weight;
    }

   public:
    explicit WordGraph(int nodecap) : path_nums(0), paths(256) { nodes.reserve(nodecap); }

    void addNode(int pos, int size, int idx) {
        symbol s;
        s.start = pos;
        s.size  = size;
        s.idx   = idx;
        nodes.emplace_back(s);
    }

    void setPath(int endidx, std::function<double(int, int)> idx_dist) {
        std::sort(nodes.begin(), nodes.end(), symbol_compare_func);
        // Lookup all bigrams.
        for (size_t i = 0; i < nodes.size(); i++) {
            if (nodes[i].start == 0) {
                addPath(-1, i, idx_dist(-1, nodes[i].idx));
                continue;
            }
            break;
        }
        for (size_t i = 0; i < nodes.size(); i++) {
            auto prenode = &nodes[i];
            int nextpos  = prenode->start + prenode->size;
            if (nextpos >= endidx) {
                addPath(i, -1, 0);
                break;
            }
            for (size_t j = i + 1; j < nodes.size(); j++) {
                auto nxnode = &nodes[j];
                if (nxnode->start == nextpos) {
                    addPath(i, j, idx_dist(prenode->idx, nxnode->idx));
                    continue;
                }
                if (nxnode->start > nextpos) break;
            }
        }
    }

    void bestPaths(const std::string& eng, std::vector<std::string>& ret) const {
        // init gloab val
        std::vector<double> dist(nodes.size() + 2, std::numeric_limits<double>::max());
        std::vector<int> prev(nodes.size() + 2, -2);
        using iPair = std::pair<double, int>;
        std::priority_queue<iPair, std::vector<iPair>, std::greater<iPair>> pq;
        pq.push(std::make_pair(0.0, -1));
        dist[0] = 0;
        // calute best path
        while (!pq.empty()) {
            int u = pq.top().second;
            pq.pop();
            for (size_t i = u + 1; i < path_nums; i++) {
                if (paths[i].right == u) {
                    int v  = paths[i].left;
                    int pv = v < 0 ? dist.size() - 1 : v + 1;

                    double new_weight = dist[u + 1] + paths[i].score;
                    // If there is shorted path to v through u.
                    if (dist[pv] > new_weight) {
                        dist[pv] = new_weight;
                        prev[pv] = u;
                        if (v >= 0) pq.push(std::make_pair(new_weight, v));
                    }
                    continue;
                }
                if (paths[i].right > u) break;
            }
        }
        // get code
        int pre = prev[prev.size() - 1];
        while (pre > -1) {
            ret.emplace_back(eng.substr(nodes[pre].start, nodes[pre].size));
            pre = prev[pre + 1];
        }
        std::reverse(ret.begin(), ret.end());
    }

    ~WordGraph() {
        nodes.clear();
        paths.clear();
        path_nums = 0;
    }
};

inline bool is_digits(const std::string& str) {
    return std::all_of(str.begin(), str.end(), ::isdigit);  // C++11
}

namespace codemap {
static const int special_code_nums = 5;

static const char* special_image[special_code_nums] = {
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
};
// const code
static const int pad_code  = 0;
static const int unk_code  = 1;
static const int cls_code  = 2;
static const int sep_code  = 3;
static const int mask_code = 4;

}  // namespace codemap

namespace darts {
class WordPice : public SegmentPlugin {
   private:
    static const char* BASE_DIR;
    // vars
    darts::BigramDict english_token_dict;
    std::unordered_map<std::string, int> codes;
    std::vector<std::string> chars_list;
    // code it

   private:
    void numToken(const std::string& num, int piceNum, std::vector<std::string>& ret) const {
        // token num str
        if (num.length() < piceNum) {
            ret.push_back(num);
            return;
        }
        int sidx    = 0;
        int leftnum = num.length() % piceNum;
        int lenOuts = num.length();
        if (leftnum != 0) {
            lenOuts += 1;
        } else {
            leftnum = piceNum;
        }
        for (int i = 0; i < lenOuts; i++) {
            int endi = piceNum * i + leftnum;
            ret.push_back(num.substr(sidx, endi - sidx));
            sidx = endi;
        }
    }

    int getImageCode(const std::string& image, const std::string& ttype) const {
        std::unordered_map<std::string, int>::const_iterator _it;
        _it = codes.find(image);
        if (_it == codes.end()) {
            _it = codes.find(ttype);
            if (_it == codes.end()) {
                return codemap::unk_code;
            }
        }
        // keep special image label
        return _it->second + codemap::special_code_nums;
    }

    void engToken(const std::string& eng, std::vector<std::string>& ret) const {
        // token english str
        std::string token = fmt::format("▁{}", eng);
        if (token.length() < 3 || token.length() > 50) {  // too long or short codes
            ret.push_back(token);
            return;
        }
        // load code
        WordGraph graph(token.size() * 2);
        // Splits the input into character sequence
        graph.addNode(0, 2, english_token_dict.getWordKey(token.substr(0, 2)));
        for (size_t i = 2; i < token.size(); i++) {
            graph.addNode(i, 1, english_token_dict.getWordKey(token.substr(i, 1)));
        }
        english_token_dict.matchKey(token, [&graph](int pos, int size, int idx) {
            if (size > 1) {
                graph.addNode(pos, size, idx);
            }
        });
        // set uni-gram
        graph.setPath(token.size(), [this](int pidx, int nidx) -> double {
            // call dict
            return this->english_token_dict.wordDist(pidx, nidx);
        });
        // select best tokens
        graph.bestPaths(token, ret);
    }

    int loadCodeMap(const std::string& relpath) {
        // load codes and chars_list
        std::set<std::string> _list;
        // load relpath data into _list
        std::ifstream fin(relpath.c_str());
        if (!fin.is_open()) {
            std::cerr << "ERROR: open data " << relpath << " file failed " << std::endl;
            return EXIT_FAILURE;
        }
        std::string line;
        size_t s;
        while (std::getline(fin, line)) {
            darts::trim(line);
            if (line.empty()) continue;
            // split line use first item
            s = line.find(' ');
            if (s != std::string::npos) {
                line = line.substr(0, s);
            }
            s = line.find('\t');
            if (s != std::string::npos) {
                line = line.substr(0, s);
            }
            if (line.empty()) continue;
            _list.insert(line);
        }
        fin.close();
        // init data
        chars_list.reserve(_list.size());
        chars_list.insert(chars_list.end(), _list.begin(), _list.end());
        std::sort(chars_list.begin(), chars_list.end());
        for (int i = 0; i < chars_list.size(); i++) {
            codes[chars_list[i]] = i;
        }
        return EXIT_SUCCESS;
    }

   public:
    int initalize(const std::map<std::string, std::string>& params,
                  std::map<std::string, std::shared_ptr<SegmentPlugin>>& plugins) {
        auto it = params.find(BASE_DIR);
        if (it == params.end() || it->second.empty()) {
            std::cerr << "ERROR: could not find key:" << BASE_DIR << std::endl;
            return EXIT_FAILURE;
        }
        namespace fs = std::filesystem;

        fs::path engdict_dir(it->second);
        engdict_dir.append("engdict");
        std::string releng_dir = getResource(engdict_dir.c_str(), true);
        if (!english_token_dict.loadDict(releng_dir)) {
            std::cerr << "ERROR: load english token dict dir " << releng_dir << " failed " << std::endl;
            return EXIT_FAILURE;
        }

        fs::path dict_path(it->second);
        dict_path.append("chars.map.txt");
        std::string relpath = getResource(dict_path.c_str());
        if (loadCodeMap(relpath)) {
            return EXIT_FAILURE;
        }

        return EXIT_SUCCESS;
    }

    /**
     * @brief 对给定的字符串句子进行分解
     *
     */
    void encode(const darts::AtomList& input, std::function<void(int code, int atom_postion)> hit,
                bool skip_empty_token = true) const {
        std::vector<std::string> _cache;
        hit(codemap::sep_code, -1);
        int postion = -1;
        for (std::shared_ptr<darts::Atom> atom : input) {
            postion += 1;
            if (skip_empty_token && !atom->char_type.compare("EMPTY")) continue;
            if (atom->masked) {  // mask atom
                hit(codemap::mask_code, postion);
                continue;
            }
            if (!atom->char_type.compare("ENG")) {
                _cache.clear();
                engToken(atom->image, _cache);
                for (std::string w : _cache) {
                    hit(getImageCode(w, atom->char_type), postion);
                }
                continue;
            }
            if (!atom->char_type.compare("NUM") && is_digits(atom->image)) {
                _cache.clear();
                numToken(atom->image, 3, _cache);
                for (std::string w : _cache) {
                    hit(getImageCode(w, atom->char_type), postion);
                }
                continue;
            }
            hit(getImageCode(atom->image, atom->char_type), postion);
        }
        hit(codemap::cls_code, -1);
    }

    /**
     * @brief 输出原始的code对应的字符串
     *
     * @param code
     * @return const char32_t*
     */
    const std::string decode(int code) const {
        if (code < codemap::special_code_nums) {
            return codemap::special_image[code];
        }
        code -= codemap::special_code_nums;
        if (code >= 0 && code < chars_list.size()) {
            return chars_list[code];
        }
        return "";
    }
};

class TypeEncoder : public SegmentPlugin {
   public:
    /**
     * get word type code
     */
    virtual int encode(const std::shared_ptr<darts::Word> word) const = 0;

    /**
     * explain a type code
     */
    virtual const std::string decode(int code) const = 0;
};

class LabelEncoder : public TypeEncoder {
   private:
    static const char* LABEL_HX_FILE;
    std::unordered_map<std::string, std::pair<int, float>> kind_codes;
    std::vector<std::string> labels;

   private:
    int loadKindCodeMap(const std::string& relpath) {
        // load relpath data into _list
        std::ifstream fin(relpath.c_str());
        if (!fin.is_open()) {
            std::cerr << "ERROR: open kind data " << relpath << " file failed " << std::endl;
            return EXIT_FAILURE;
        }
        std::string line;
        size_t s;
        while (std::getline(fin, line)) {
            darts::trim(line);
            if (line.empty()) continue;
            s = line.find('#');
            if (s == std::string::npos) continue;

            std::string key(line.substr(0, s));
            darts::trim(key);
            std::string value(line.substr(s + 1));
            darts::trim(value);
            float val = std::stof(value);
            // set data
            kind_codes[key] = std::make_pair(kind_codes.size(), val);
        }
        fin.close();
        labels.resize(kind_codes.size());
        for (auto& p : kind_codes) {
            labels[p.second.first] = p.first;
        }
        return EXIT_SUCCESS;
    }

   public:
    int initalize(const std::map<std::string, std::string>& params,
                  std::map<std::string, std::shared_ptr<SegmentPlugin>>& plugins) {
        auto it = params.find(LABEL_HX_FILE);
        if (it == params.end() || it->second.empty()) {
            std::cerr << "ERROR: could not find key:" << LABEL_HX_FILE << std::endl;
            return EXIT_FAILURE;
        }
        if (loadKindCodeMap(it->second)) {
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }

    int encode(const std::shared_ptr<darts::Word> word) const {
        auto labelHx = [this](const std::string& label) -> float {
            std::unordered_map<std::string, std::pair<int, float>>::const_iterator _it;
            _it = this->kind_codes.find(label);
            if (_it == this->kind_codes.end()) {
                return 0;
            }
            return _it->second.second;
        };
        std::string label(word->maxHXlabel(labelHx));
        std::unordered_map<std::string, std::pair<int, float>>::const_iterator _it;
        _it = kind_codes.find(label);
        if (_it == kind_codes.end()) {
            return codemap::unk_code;
        }
        return _it->second.first + codemap::special_code_nums;
    }

    /**
     * @brief 输出原始的code对应的字符串
     *
     * @param code
     * @return const char32_t*
     */
    const std::string decode(int code) const {
        if (code < codemap::special_code_nums) {
            return codemap::special_image[code];
        }
        code -= codemap::special_code_nums;
        if (code >= 0 && code < labels.size()) {
            return labels[code];
        }
        return "";
    }
};

/**
 * pinyin标注
 */
class PinyinEncoder : public TypeEncoder {
   public:
    const static int pad = 0;
    const static int st  = 1;
    const static int et  = 2;
    const static int unk = 3;

   private:
    std::map<std::string, int> pyin_codes;
    std::vector<std::string> plist;

   public:
    int initalize(const std::map<std::string, std::string>& params,
                  std::map<std::string, std::shared_ptr<SegmentPlugin>>& plugins) {
        std::set<std::string> datas;
        for (auto it = _WordPinyin.begin(); it != _WordPinyin.end(); ++it) {
            for (auto w : it->second->piyins) {
                datas.insert(w);
            }
        }
        plist.reserve(datas.size());
        plist.insert(plist.end(), datas.begin(), datas.end());
        std::sort(plist.begin(), plist.end());
        for (int i = 0; i < plist.size(); ++i) {
            pyin_codes[plist[i]] = i;
        }
        return EXIT_SUCCESS;
    }

    int encode(const std::string& pinyin) const {
        auto it = pyin_codes.find(pinyin);
        if (it == pyin_codes.end()) {
            return unk;
        }
        return it->second + 4;
    }

    int encode(const std::shared_ptr<darts::Word> word) const {
        if (word->feat >= 0) return word->feat;
        // encode word
        return encode(word->maxHXlabel(nullptr));
    }

    const std::string decode(int code) const {
        code -= 4;
        if (code >= 0 && code < plist.size()) {
            return plist[code];
        }
        return "";
    }
};

}  // end namespace darts

#endif  // SRC_IMPL_ENCODER_HPP_
