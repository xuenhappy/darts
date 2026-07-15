/*
 * File: recognizer.hpp
 * Project: impl
 * File Created: Saturday, 25th December 2021 11:05:37 pm
 * Author: Xu En (xuen@mokar.com)
 * -----
 * Last Modified: Saturday, 1st January 2022 12:14:11 pm
 * Modified By: Xu En (nanhangxuen@163.com)
 * -----
 * Copyright 2021 - 2022 Your Company, Moka
 */

#ifndef SRC_IMPL_RECOGNIZER_HPP_
#define SRC_IMPL_RECOGNIZER_HPP_

#include <assert.h>
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include "../core/segment.hpp"
#include "../utils/dregex.hpp"
#include "../utils/filetool.hpp"
#include "../utils/pinyin.hpp"
#include "../utils/strtool.hpp"
#include "./encoder.hpp"

namespace darts {
class AtomListIterator : public dregex::StringIter {
   private:
    const AtomList* m_list;

   public:
    explicit AtomListIterator(const AtomList& m_list) { this->m_list = &m_list; }
    void walks(std::function<bool(const std::string&, size_t)> hit) const {
        std::string tmp;
        for (auto i = 0; i < m_list->size(); i++) {
            tmp = m_list->at(i)->image;
            if (hit(tolower(tmp), i)) {
                break;
            }
        }
    }
};
/**
 * @brief a recongnizer that base on dict
 *
 */
class DictWordRecongnizer : public CellRecognizer {
   private:
    static const char* PB_FILE_KEY;
    static const char* ATOM_MODE_KEY;

    dregex::Trie trie;
    bool atom_mode;

   public:
    DictWordRecongnizer() : atom_mode(false) {}
    explicit DictWordRecongnizer(const std::string& pbfile) : atom_mode(false) { this->trie.loadPb(pbfile); }

    int initalize(const std::map<std::string, std::string>& param,
                  std::map<std::string, std::shared_ptr<SegmentPlugin>>& dicts) {
        auto it = param.find(ATOM_MODE_KEY);
        if (it != param.end() && "true" == it->second) {
            atom_mode = true;
        }
        auto iter = param.find(PB_FILE_KEY);
        if (iter == param.end()) {
            std::cerr << PB_FILE_KEY << " key not found in dictionary!" << std::endl;
            return EXIT_FAILURE;
        }
        std::string dictfile = getResource(iter->second);
        return this->trie.loadPb(dictfile);
    }

    void addWords(const AtomList& dstSrc, SegPath& cmap) const {
        auto cur = cmap.Head();
        auto token_at = [&dstSrc](size_t index) -> const std::string& { return dstSrc.at(index)->image; };
        if (atom_mode) {
            auto hit = [&](size_t s, size_t e, const std::vector<int64_t>* labels) -> bool {
                if (!labels) return false;
                int feat = -1;
                for (auto tidx : *labels) {
                    auto tag = this->trie.getLabel(tidx);
                    if (tag && *tag) {
                        auto word = std::make_shared<Word>(dstSrc, s, e);
                        word->addLabel(tag);
                        word->feat = feat;
                        feat--;
                        cur = cmap.addCell(word, cur);
                    }
                }
                return false;
            };
            this->trie.parseContiguous(dstSrc.size(), token_at, hit);
        } else {
            auto hit = [&](size_t s, size_t e, const std::vector<int64_t>* labels) -> bool {
                auto word = std::make_shared<Word>(dstSrc, s, e);
                if (labels)
                    for (auto tidx : *labels) {
                        auto tag = this->trie.getLabel(tidx);
                        if (tag && *tag) word->addLabel(tag);
                    }
                cur = cmap.addCell(word, cur);
                return false;
            };
            this->trie.parseContiguous(dstSrc.size(), token_at, hit);
        }
    }
};
const char* DictWordRecongnizer::PB_FILE_KEY   = "pbfile.path";
const char* DictWordRecongnizer::ATOM_MODE_KEY = "atom.mode";
REGISTER_Recognizer(DictWordRecongnizer);

/** Recognize numeric address components that are unsuitable for a finite POI dictionary. */
class AddressRecongnizer : public CellRecognizer {
   private:
    static bool endsWith(const std::string& value, const std::set<std::string>& suffixes) {
        for (const auto& suffix : suffixes) {
            if (value.size() >= suffix.size() &&
                value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0) return true;
        }
        return false;
    }

   public:
    int initalize(const std::map<std::string, std::string>&,
                  std::map<std::string, std::shared_ptr<SegmentPlugin>>&) override {
        return EXIT_SUCCESS;
    }

    void addWords(const AtomList& atoms, SegPath& path) const override {
        static const std::set<std::string> component_suffixes = {
            "号", "号院", "栋", "幢", "座", "层", "室", "弄", "单元", "楼"
        };
        static const std::set<std::string> road_suffixes = {
            "路", "街", "大道", "公路", "道", "巷", "胡同", "街道"
        };
        static const std::set<std::string> poi_suffixes = {
            "小区", "大厦", "广场", "中心", "公园", "医院", "学校", "大学", "车站", "机场",
            "园", "园区"
        };
        static const std::set<std::string> division_suffixes = {
            "省", "市", "区", "县", "镇", "乡", "自治州", "自治区"
        };
        Cursor cursor = path.Head();
        for (size_t start = 0; start < atoms.size(); ++start) {
            const auto& first = atoms.at(start);
            std::string candidate;
            const size_t limit = std::min(atoms.size(), start + 12);
            for (size_t end = start; end < limit; ++end) {
                const auto& atom = atoms.at(end);
                candidate.append(atom->image);
                const size_t atom_length = end - start + 1;
                if (atom_length < 2) continue;

                const bool numeric_component = atom_length <= 3 &&
                    (first->char_type == char_type::NUM || first->char_type == char_type::ENG) &&
                    endsWith(candidate, component_suffixes);
                const bool cjk_candidate = first->char_type == char_type::CJK &&
                    std::all_of(atoms.begin() + start, atoms.begin() + end + 1,
                                [](const std::shared_ptr<Atom>& item) {
                                    return item->char_type == char_type::CJK;
                                });
                const char* label = nullptr;
                if (numeric_component) label = "ADDR_COMPONENT";
                else if (cjk_candidate && atom_length <= 8 && endsWith(candidate, road_suffixes)) label = "ADDR_ROAD";
                else if (cjk_candidate && atom_length <= 10 && endsWith(candidate, poi_suffixes)) label = "ADDR_POI";
                else if (cjk_candidate && atom_length <= 6 && endsWith(candidate, division_suffixes)) label = "ADDR_DIVISION";
                if (label) {
                    auto word = std::make_shared<Word>(atoms, start, end + 1);
                    word->addLabel(label);
                    word->addLabel("ADDR_RULE");
                    cursor = path.addCell(word, cursor);
                }
            }
        }
    }
};
REGISTER_Recognizer(AddressRecongnizer);

/**
 * pinyin标注,该插件不可以❌和其他插件混用
 */
class PinyinRecongnizer : public CellRecognizer {
   private:
    std::shared_ptr<PinyinEncoder> encoder;
    static const char* ENCODER_PARAM;
    static const char* NON_CJK_LABEL_PARAM;
    std::string non_cjk_label;

   public:
    int initalize(const std::map<std::string, std::string>& params,
                  std::map<std::string, std::shared_ptr<SegmentPlugin>>& plugins) {
        auto it = plugins.find(ENCODER_PARAM);
        if (it == plugins.end()) {
            std::cerr << "could not find plugin " << ENCODER_PARAM << std::endl;
            return EXIT_FAILURE;
        }
        this->encoder = std::dynamic_pointer_cast<PinyinEncoder>(it->second);
        if (this->encoder == nullptr) {
            std::cerr << "plugin in init error " << ENCODER_PARAM << std::endl;
            return EXIT_FAILURE;
        }
        auto label = params.find(NON_CJK_LABEL_PARAM);
        if (label != params.end()) non_cjk_label = label->second;
        return EXIT_SUCCESS;
    }

    void addWords(const AtomList& dstSrc, SegPath& cmap) const {
        auto dfunc = [this, &dstSrc](Cursor cur) {
            auto pw   = cur->val;
            auto pyin = pinyin(pw->text());
            std::string annotation;
            if (pyin != nullptr && pw->et - pw->st > 1 && pyin->piyins.size() == pw->et - pw->st) {
                annotation = darts::join(pyin->piyins, " ");
            } else {
                bool is_cjk = true;
                std::vector<std::string> readings;
                for (size_t pos = pw->st; pos < pw->et; ++pos) {
                    auto atom = dstSrc.at(pos);
                    if (atom->char_type != char_type::CJK) {
                        is_cjk = false;
                        break;
                    }
                    auto reading = pinyin(atom->image);
                    if (!reading || reading->piyins.empty()) {
                        is_cjk = false;
                        break;
                    }
                    readings.push_back(reading->piyins.front());
                }
                if (is_cjk) annotation = darts::join(readings, " ");
            }
            if (annotation.empty()) annotation = non_cjk_label;
            if (annotation.empty()) return;
            pw->addLabel(annotation);
            pw->feat = annotation.find(' ') == std::string::npos ? encoder->encode(annotation) : PinyinEncoder::unk;
        };
        cmap.iterRow(nullptr, -1, dfunc);
    }
};

const char* PinyinRecongnizer::ENCODER_PARAM = "pyin.encoder";
const char* PinyinRecongnizer::NON_CJK_LABEL_PARAM = "non-cjk.label";
REGISTER_Recognizer(PinyinRecongnizer);

class RuleRecongnizer : public CellRecognizer {
   public:
    /**
     * @brief use rule find all possable word in alist and add into buffer
     *
     * @param dstSrc
     * @param buffer
     */
    virtual void seek(const AtomList& dstSrc, std::vector<std::shared_ptr<Word>>& buffer) const = 0;
    void addWords(const AtomList& dstSrc, SegPath& cmap) const {
        std::vector<std::shared_ptr<Word>> buffer;
        buffer.reserve(dstSrc.size() / 2 + 1);
        seek(dstSrc, buffer);
        auto comp = [](const std::shared_ptr<Word>& i1, const std::shared_ptr<Word>& i2) {
            return i1->st == i2->st ? i1->et < i2->et : i1->st < i2->st;
        };
        std::sort(buffer.begin(), buffer.end(), comp);
        Cursor cur = cmap.Head();
        for (auto x : buffer) {
            cur = cmap.addNext(cur, x);
        }
        buffer.clear();
    }
};

}  // namespace darts

#endif  // SRC_IMPL_RECOGNIZER_HPP_
