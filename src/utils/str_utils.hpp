/*
 * File: StrUtils.hpp
 * Project: utils
 * 提供一些基础的字符串处理工具
 * File Created: Saturday, 11th December 2021 7:55:02 pm
 * Author: Xu En (xuen@mokar.com)
 * -----
 * Last Modified: Saturday, 11th December 2021 7:55:13 pm
 * Modified By: Xu En (xuen@mokahr.com)
 * -----
 * Copyright 2021 - 2021 Your Company, Moka
 */

#ifndef SRC_UTILS_STR_UTILS_HPP_
#define SRC_UTILS_STR_UTILS_HPP_

#include <json/json.h>
#include <algorithm>
#include <cctype>
#include <functional>
#include <iostream>
#include <locale>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace darts {
class StringIter {
   public:
    /**
     * @brief iter this string is success
     *
     *
     * @param hit
     * @return int 0 if successful else 1
     */
    virtual void iter(std::function<bool(const std::string&, size_t)> hit) = 0;
    virtual ~StringIter() {}
};

class StringIterPairs {
   public:
    /**
     * @brief iter kv pairs
     *
     * @param hit
     * @return int 0 if successful, otherwise 1
     */
    virtual int iter(std::function<void(StringIter&, const int64_t*, size_t)> hit) = 0;
    virtual ~StringIterPairs() {}
};

class U32Iter {
   public:
    /**
     * @brief iter string
     *
     * @param hit
     */
    virtual void iter(std::function<void(int64_t, const char*, size_t)> hit) = 0;
    virtual ~U32Iter() {}
};

/**
 * @brief 字符串连接
 *
 * @param v
 * @param delt
 * @return std::string
 */
static inline std::string join(const std::vector<std::string>& v, const std::string& delt) {
    std::stringstream ss;
    std::vector<std::string>::const_iterator it = v.begin();
    if (it != v.end()) {
        ss << *it;
        ++it;
    }
    for (; it != v.end(); ++it) {
        ss << delt << *it;
    }
    return ss.str();
}

static inline std::string join(const std::set<std::string>& v, const std::string& delt) {
    std::stringstream ss;
    std::set<std::string>::const_iterator it = v.begin();
    if (it != v.end()) {
        ss << *it;
        ++it;
    }
    for (; it != v.end(); ++it) {
        ss << delt << *it;
    }
    return ss.str();
}

/**
 * @brief trim from start (in place)
 *
 * @param s
 */
static inline std::string& ltrim(std::string& s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int c) { return !std::isspace(c); }));
    return s;
}

/**
 * @brief
 *
 * @param s
 */
static inline std::string& rtrim(std::string& s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](int c) { return !std::isspace(c); }).base(), s.end());
    return s;
}

// trim from both ends (in place)
static inline std::string& trim(std::string& s) { return rtrim(ltrim(s)); }

static inline std::string ltrim_copy(const char* cs) {
    std::string s(cs);
    return ltrim(s);
}

static inline std::string rtrim_copy(const char* cs) {
    std::string s(cs);
    return rtrim(s);
}

static inline std::string trim_copy(const char* cs) {
    std::string s(cs);
    return trim(s);
}

static inline std::string& tolower(std::string& data) {
    std::transform(data.begin(), data.end(), data.begin(), [](unsigned char c) { return std::tolower(c); });
    return data;
}

static inline std::string& toupper(std::string& data) {
    std::transform(data.begin(), data.end(), data.begin(), [](unsigned char c) { return std::toupper(c); });
    return data;
}

/**
 * @brief split a give line by a delimiter
 *
 * @param s
 * @param delimiter
 * @param res
 */
static inline void split(const std::string& s, const std::string& delimiter, std::vector<std::string>& res) {
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
        res.emplace_back(s.substr(pos_start, pos_end - pos_start));
        pos_start = pos_end + delim_len;
    }
    if (pos_start < s.length()) {
        res.emplace_back(s.substr(pos_start));
    }
}
/**
 * @brief Get the Json Root object
 *
 * @param data
 * @param root
 * @return int
 */
static inline int getJsonRoot(const std::string& data, Json::Value& root) {
    bool res;
    JSONCPP_STRING errs;
    Json::CharReaderBuilder readerBuilder;
    std::unique_ptr<Json::CharReader> const jsonReader(readerBuilder.newCharReader());
    res = jsonReader->parse(data.c_str(), data.c_str() + data.length(), &root, &errs);

    if (!res || !errs.empty()) {
        std::cerr << "ERROR: parseJson err. " << errs << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

}  // namespace darts
#endif  // SRC_UTILS_STR_UTILS_HPP_
