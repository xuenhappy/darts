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

struct _SymbolPair {
    int left;      // left index of this pair
    int right;     // right index of this pair
    double score;  // score of this pair. small is better
};

struct _Symbol {
    int start;
    int size;
    int idx;
};
inline bool symbol_compare_func(_Symbol& i1, _Symbol& i2) {
    return (i1.start < i2.start) || (i1.start == i2.start && i1.size < i2.size);
}

class _WordGraph {
   private:
    std::vector<_Symbol> nodes;      // nodes
    size_t path_nums;                // path nums
    std::vector<_SymbolPair> paths;  // paths

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
    explicit _WordGraph(int nodecap) : path_nums(0), paths(256) { nodes.reserve(nodecap); }

    void addNode(int pos, int size, int idx) {
        _Symbol s;
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
        prev[0] = -2;
        // calute best path
        while (!pq.empty()) {
            int u = pq.top().second;
            pq.pop();
            for (size_t i = u + 1; i < path_nums; i++) {
                if (paths[i].right == u) {
                    // adjacent of u.

                    int v         = paths[i].left;
                    double weight = paths[i].score;
                    int pv        = v < 0 ? dist.size() - 1 : v + 1;
                    // If there is shorted path to v through u.
                    if (dist[pv] > dist[u + 1] + weight) {
                        // Updating distance of v
                        dist[pv] = dist[u + 1] + weight;
                        prev[pv] = u;
                        if (v >= 0) pq.push(std::make_pair(dist[v], v));
                    }
                    continue;
                }
                if (paths[i].right > u) break;
            }
        }
        // get code
        int pre = prev[prev.size() - 1];
        while (pre > 0) {
            ret.emplace_back(eng.substr(nodes[pre].start, nodes[pre].size));
        }
        std::reverse(ret.begin(), ret.end());
    }

    ~_WordGraph() {
        nodes.clear();
        paths.clear();
        path_nums = 0;
    }
};

inline bool is_digits(const std::string& str) {
    return std::all_of(str.begin(), str.end(), ::isdigit);  // C++11
}

namespace codemap {
// const code
static const int pad_code  = 0;
static const int unk_code  = 1;
static const int cls_code  = 2;
static const int sep_code  = 3;
static const int mask_code = 4;

// const code char
static const char* unk_char  = "[UNK]";
static const char* cls_char  = "[CLS]";
static const char* sep_char  = "[SEP]";
static const char* mask_char = "[MASK]";
static const char* pad_char  = "[PAD]";
}  // namespace codemap

namespace darts {
class WordPice :public SegmentPlugin {
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

    int getTypeCode(const std::string& type) const {
        std::unordered_map<std::string, int>::const_iterator _it = codes.find(type);
        if (_it != codes.end()) {
            return _it->second;
        }
        return codemap::unk_code;
    }

    void engToken(const std::string& eng, std::vector<std::string>& ret) const {
        // token english str
        std::string token = fmt::format("▁{}", eng);
        if (token.length() < 3 || token.length() > 50) {  // too long or short codes
            ret.push_back(token);
            return;
        }
        // load code
        _WordGraph graph(token.size() * 2);
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

   public:
    int initalize(const std::map<std::string, std::string>& params,
                  std::map<std::string, std::shared_ptr<SegmentPlugin>>& plugins) {
        auto it = params.find(BASE_DIR);
        if (it == params.end() || it->second.empty()) {
            std::cerr << "ERROR: could not find key:" << BASE_DIR << std::endl;
            return EXIT_FAILURE;
        }
        namespace fs = std::filesystem;
        fs::path dict_path(it->second);
        dict_path.append("chars.map.txt");
        fs::path engdict_dir(it->second);
        engdict_dir.append("engdict");
        std::string releng_dir = getResource(engdict_dir.c_str(), true);
        if (!english_token_dict.loadDict(releng_dir)) {
            std::cerr << "ERROR: load english token dict dir " << releng_dir << " failed " << std::endl;
            return EXIT_FAILURE;
        }
        std::string relpath = getResource(dict_path.c_str());
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
        chars_list.reserve(_list.size() + 10);
        chars_list.resize(5);
        chars_list[codemap::pad_code]  = codemap::pad_char;
        chars_list[codemap::unk_code]  = codemap::unk_char;
        chars_list[codemap::cls_code]  = codemap::cls_char;
        chars_list[codemap::sep_code]  = codemap::sep_char;
        chars_list[codemap::mask_code] = codemap::mask_char;
        chars_list.insert(chars_list.end(), _list.begin(), _list.end());
        for (int i = 0; i < chars_list.size(); i++) {
            codes[chars_list[i]] = i;
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
        std::unordered_map<std::string, int>::const_iterator _it;
        int postion = -1;
        for (std::shared_ptr<darts::Atom> atom : input) {
            postion += 1;
            if (skip_empty_token && atom->hasLabel("EMPTY")) continue;
            if (atom->masked) {  // mask atom
                hit(codemap::mask_code, postion);
                continue;
            }
            if (atom->hasLabel("ENG")) {
                _cache.clear();
                engToken(atom->image, _cache);
                for (std::string w : _cache) {
                    _it = codes.find(w);
                    if (_it == codes.end()) {
                        hit(getTypeCode("ENG"), postion);
                        continue;
                    }
                    hit(_it->second, postion);
                }
                continue;
            }
            if (atom->hasLabel("NUM") && is_digits(atom->image)) {
                _cache.clear();
                numToken(atom->image, 3, _cache);
                for (std::string w : _cache) {
                    _it = codes.find(w);
                    if (_it == codes.end()) {
                        hit(getTypeCode("NUM"), postion);
                        continue;
                    }
                    hit(_it->second, postion);
                }
                continue;
            }
            _it = codes.find(atom->image);
            if (_it == codes.end()) {
                // check tye code for return value
                std::string label(atom->maxHXlabel(nullptr));
                hit(getTypeCode(label), postion);
                continue;
            }
            hit(_it->second, postion);
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
        if (code >= 0 && code < chars_list.size()) {
            return chars_list[code];
        }
        return "";
    }
};
}  // end namespace darts


#endif  // SRC_IMPL_ENCODER_HPP_
