/*
 * File: FileUtils.hpp
 * Project: utils
 * 提供一些常用的文件处理工具
 * File Created: Saturday, 11th December 2021 7:54:38 pm
 * Author: Xu En (xuen@mokar.com)
 * -----
 * Last Modified: Saturday, 11th December 2021 7:55:40 pm
 * Modified By: Xu En (xuen@mokahr.com)
 * -----
 * Copyright 2021 - 2021 Your Company, Moka
 */

#ifndef SRC_UTILS_FILE_UTILS_HPP_
#define SRC_UTILS_FILE_UTILS_HPP_

#include <dlfcn.h>
#include <stdlib.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

/**
 * @brief get a dll path
 *
 * @return const char*
 */
const char *dllPath(void) {
    Dl_info dl_info;
    int rc = dladdr((void *)dllPath, &dl_info);
    if (!rc) {
        return "";
    }
    return (dl_info.dli_fname);
}

/**
 * @brief Get the Resource object
 *
 * @param spath
 * @return const std::string
 */
const std::string getResource(const std::string &spath) {
    namespace fs = std::filesystem;
    fs::path ori_path(spath);
    if (fs::exists(ori_path) && fs::is_regular_file(ori_path)) {
        return ori_path.string();
    }

    std::string dllpath(dllPath());
    if (!dllpath.empty()) {
        // check the dll path
        fs::path dpath1(dllpath);
        auto bin_dir = dpath1.parent_path();
        bin_dir.append(spath);
        if (fs::exists(bin_dir) && fs::is_regular_file(bin_dir)) {
            return bin_dir.string();
        }

        // check dll parent
        bin_dir = dpath1.parent_path();
        bin_dir.append("../").append(spath);
        if (fs::exists(bin_dir) && fs::is_regular_file(bin_dir)) {
            return bin_dir.string();
        }
    }

    char *env = getenv("DARTS_CONF_PATH");
    if (!env) {
        return spath;
    }
    std::string darts_env(env);
    if (!darts_env.empty()) {
        fs::path p1(darts_env);
        p1.append(spath);
        if (fs::exists(p1) && fs::is_regular_file(p1)) {
            return p1.string();
        }
    }

    return spath;
}
#endif  // SRC_UTILS_FILE_UTILS_HPP_
