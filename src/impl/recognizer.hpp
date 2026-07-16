/*
 * File: recognizer.hpp
 * Project: impl
 * File Created: Saturday, 25th December 2021 11:05:37 pm
 * Author: Xu En (nanhangxuen@163.com)
 * -----
 * Last Modified: Saturday, 1st January 2022 12:14:11 pm
 * Modified By: Xu En (nanhangxuen@163.com)
 * -----
 * Copyright 2021 - 2022 XuEn
 */

#ifndef SRC_IMPL_RECOGNIZER_HPP_
#define SRC_IMPL_RECOGNIZER_HPP_

#include <assert.h>
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cctype>
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
 * Recognize common date, time, and number-unit expressions with bounded
 * deterministic state machines. The recognizer deliberately operates on
 * AtomList tokens instead of regular expressions, keeping the scan linear and
 * preserving the atom indexes consumed by graph selection and neural data.
 */
class TemporalQuantityRecongnizer : public CellRecognizer {
   private:
    static bool isToken(const AtomList& atoms, size_t index, const char* value) {
        return index < atoms.size() && atoms.at(index)->image == value;
    }

    static std::string asciiLower(std::string value) {
        for (char& character : value)
            if (static_cast<unsigned char>(character) < 128)
                character = static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
        return value;
    }

    static bool isChineseNumber(const std::string& value) {
        static const std::set<std::string> digits = {
            "零", "〇", "一", "二", "两", "兩", "三", "四", "五", "六", "七", "八", "九",
            "十", "百", "千", "万", "萬", "亿", "億", "兆", "半", "几", "幾", "廿", "卅"
        };
        return digits.find(value) != digits.end();
    }

    static size_t consumeNumber(const AtomList& atoms, size_t start) {
        if (start >= atoms.size()) return start;
        size_t index = start;
        if ((isToken(atoms, index, "-") || isToken(atoms, index, "+")) &&
            index + 1 < atoms.size() &&
            (atoms.at(index + 1)->char_type == char_type::NUM ||
             isChineseNumber(atoms.at(index + 1)->image))) {
            ++index;
        }
        if (atoms.at(index)->char_type == char_type::NUM) {
            ++index;
            // Decimal and grouped forms: 12.5, 1,000.25.
            while (index + 1 < atoms.size() &&
                   (isToken(atoms, index, ".") || isToken(atoms, index, ",")) &&
                   atoms.at(index + 1)->char_type == char_type::NUM) {
                index += 2;
            }
            return index;
        }
        while (index < atoms.size() && index - start < 16 &&
               isChineseNumber(atoms.at(index)->image)) {
            ++index;
        }
        return index;
    }

    static bool isMonthName(const std::string& value) {
        static const std::set<std::string> months = {
            "jan", "january", "feb", "february", "mar", "march", "apr", "april", "may",
            "jun", "june", "jul", "july", "aug", "august", "sep", "sept", "september",
            "oct", "october", "nov", "november", "dec", "december"
        };
        return months.find(asciiLower(value)) != months.end();
    }

    static size_t consumeDate(const AtomList& atoms, size_t start) {
        const size_t first_end = consumeNumber(atoms, start);
        if (first_end > start) {
            // Chinese forms: 2026年7月16日, 7月16号.
            if (isToken(atoms, first_end, "年")) {
                const size_t month_end = consumeNumber(atoms, first_end + 1);
                if (month_end > first_end + 1 && isToken(atoms, month_end, "月")) {
                    const size_t day_end = consumeNumber(atoms, month_end + 1);
                    if (day_end > month_end + 1 &&
                        (isToken(atoms, day_end, "日") || isToken(atoms, day_end, "号") ||
                         isToken(atoms, day_end, "號")))
                        return day_end + 1;
                    return month_end + 1;
                }
            }
            if (isToken(atoms, first_end, "月")) {
                const size_t day_end = consumeNumber(atoms, first_end + 1);
                if (day_end > first_end + 1 &&
                    (isToken(atoms, day_end, "日") || isToken(atoms, day_end, "号") ||
                     isToken(atoms, day_end, "號")))
                    return day_end + 1;
            }
            // Numeric forms: 2026-07-16, 07/16/2026, 2026.07.16.
            if (first_end < atoms.size()) {
                const std::string separator = atoms.at(first_end)->image;
                if (separator == "-" || separator == "/" || separator == ".") {
                    const size_t second_end = consumeNumber(atoms, first_end + 1);
                    if (second_end > first_end + 1 && isToken(atoms, second_end, separator.c_str())) {
                        const size_t third_end = consumeNumber(atoms, second_end + 1);
                        if (third_end > second_end + 1) return third_end;
                    }
                }
            }
            // English forms: 16 July 2026.
            if (first_end < atoms.size() && isMonthName(atoms.at(first_end)->image)) {
                size_t end = consumeNumber(atoms, first_end + 1);
                return end > first_end + 1 ? end : first_end + 1;
            }
        }
        // English forms: July 16, 2026.
        if (start < atoms.size() && isMonthName(atoms.at(start)->image)) {
            size_t day_end = consumeNumber(atoms, start + 1);
            if (day_end == start + 1) return start;
            if (isToken(atoms, day_end, ",")) ++day_end;
            const size_t year_end = consumeNumber(atoms, day_end);
            return year_end > day_end ? year_end : day_end;
        }
        return start;
    }

    static size_t consumeTime(const AtomList& atoms, size_t start) {
        size_t number_end = consumeNumber(atoms, start);
        if (number_end == start) return start;
        // 14:30[:59] [am|pm].
        if (isToken(atoms, number_end, ":") || isToken(atoms, number_end, "：")) {
            size_t minute_end = consumeNumber(atoms, number_end + 1);
            if (minute_end == number_end + 1) return start;
            if ((isToken(atoms, minute_end, ":") || isToken(atoms, minute_end, "："))) {
                const size_t second_end = consumeNumber(atoms, minute_end + 1);
                if (second_end > minute_end + 1) minute_end = second_end;
            }
            if (minute_end < atoms.size()) {
                const std::string suffix = asciiLower(atoms.at(minute_end)->image);
                if (suffix == "am" || suffix == "pm") ++minute_end;
            }
            return minute_end;
        }
        // 9点30分15秒, 9时30分, 9點.
        if (isToken(atoms, number_end, "点") || isToken(atoms, number_end, "點") ||
            isToken(atoms, number_end, "时") || isToken(atoms, number_end, "時")) {
            size_t end = number_end + 1;
            const size_t minute_end = consumeNumber(atoms, end);
            if (minute_end > end && isToken(atoms, minute_end, "分")) {
                end = minute_end + 1;
                const size_t second_end = consumeNumber(atoms, end);
                if (second_end > end && isToken(atoms, second_end, "秒")) end = second_end + 1;
            }
            return end;
        }
        if (number_end < atoms.size()) {
            const std::string suffix = asciiLower(atoms.at(number_end)->image);
            if (suffix == "am" || suffix == "pm") return number_end + 1;
        }
        return start;
    }

    static size_t consumeQuantity(const AtomList& atoms, size_t start) {
        const size_t number_end = consumeNumber(atoms, start);
        if (number_end == start || number_end >= atoms.size()) return start;
        static const std::set<std::string> units = {
            "%", "‰", "℃", "℉", "°c", "°f",
            "个", "個", "只", "条", "條", "件", "位", "名", "人", "次", "份", "套", "台",
            "辆", "輛", "本", "册", "冊", "页", "頁", "张", "張", "片", "枚", "颗", "顆",
            "公斤", "千克", "克", "毫克", "吨", "噸", "斤", "两", "兩",
            "公里", "千米", "米", "厘米", "毫米", "平方米", "平方公里", "亩", "畝",
            "升", "毫升", "度", "元", "块", "塊", "美元", "人民币", "人民幣",
            "秒", "分钟", "分鐘", "小时", "小時", "天", "周", "週", "星期", "个月", "個月", "年",
            "kb", "mb", "gb", "tb", "kib", "mib", "gib", "hz", "khz", "mhz", "ghz",
            "mm", "cm", "m", "km", "mg", "g", "kg", "ml", "l", "w", "kw", "v", "a",
            "s", "ms", "min", "h", "hr", "hrs", "day", "days", "week", "weeks",
            "month", "months", "year", "years", "usd", "cny", "rmb", "eur"
        };
        size_t best = start;
        std::string candidate;
        for (size_t end = number_end; end < atoms.size() && end < number_end + 4; ++end) {
            candidate.append(atoms.at(end)->image);
            if (units.find(candidate) != units.end() ||
                units.find(asciiLower(candidate)) != units.end())
                best = end + 1;
        }
        return best;
    }

    static void addCandidate(const AtomList& atoms, SegPath& path, Cursor& cursor,
                             size_t start, size_t end, const char* label) {
        if (end <= start + 1) return;
        auto word = std::make_shared<Word>(atoms, start, end);
        word->addLabel(label);
        cursor = path.addCell(word, cursor);
    }

   public:
    int initalize(const std::map<std::string, std::string>&,
                  std::map<std::string, std::shared_ptr<SegmentPlugin>>&) override {
        return EXIT_SUCCESS;
    }

    void addWords(const AtomList& atoms, SegPath& path) const override {
        Cursor cursor = path.Head();
        for (size_t start = 0; start < atoms.size(); ++start) {
            const auto& atom = atoms.at(start);
            const bool signed_number =
                (atom->image == "-" || atom->image == "+") &&
                start + 1 < atoms.size() &&
                (atoms.at(start + 1)->char_type == char_type::NUM ||
                 isChineseNumber(atoms.at(start + 1)->image));
            const bool numeric_start = atom->char_type == char_type::NUM ||
                                       isChineseNumber(atom->image) || signed_number;
            const bool month_start = atom->char_type == char_type::ENG &&
                                     isMonthName(atom->image);
            if (!numeric_start && !month_start) continue;

            const size_t date_end = consumeDate(atoms, start);
            const size_t time_end = numeric_start ? consumeTime(atoms, start) : start;
            const size_t quantity_end = numeric_start ? consumeQuantity(atoms, start) : start;
            addCandidate(atoms, path, cursor, start, date_end, "DATE");
            addCandidate(atoms, path, cursor, start, time_end, "DATE");
            addCandidate(atoms, path, cursor, start, quantity_end, "DIGIT");

            if (date_end > start) {
                size_t time_start = date_end;
                if (isToken(atoms, time_start, "T") || isToken(atoms, time_start, "t"))
                    ++time_start;
                const size_t date_time_end = consumeTime(atoms, time_start);
                if (date_time_end > time_start)
                    addCandidate(atoms, path, cursor, start, date_time_end, "DATE");
            }
        }
    }
};
REGISTER_Recognizer(TemporalQuantityRecongnizer);

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
