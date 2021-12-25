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

#ifndef SRC_UTILS_FILEUTILS_HPP_
#define SRC_UTILS_FILEUTILS_HPP_

#include <dlfcn.h>
#include <stdlib.h>

#include <filesystem>
#include <fstream>
#include <string>

const char *dllPath(void) {
    Dl_info dl_info;
    int rc = dladdr((void *)dllPath, &dl_info);
    if (!rc) {
        return "";
    }
    return (dl_info.dli_fname);
}

const std::string getResource(const std::string &spath) {
    std::ifstream fin(spath.c_str());
    if (fin) {
        // file is exist and can read
        fin.close();
        return spath;
    }
    std::string dllpath = dllPath();
    if (!dllpath.empty()) {
        // check the dll path
        std::filesystem::path dpath1(dllpath);
        dpath1.append(spath);
        if (std::filesystem::exists(dpath1)) {
            return dpath1.string();
        }
        std::filesystem::path dpath2(dllpath);
        dpath2.append("../");
        dpath2.append(spath);
        if (std::filesystem::exists(dpath1)) {
            return dpath2.string();
        }
    }

    return spath;
}
#endif  // SRC_UTILS_FILEUTILS_HPP_
