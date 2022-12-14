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

#ifdef WIN32
#include <direct.h>
#include <io.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif
#include <dlfcn.h>
#include <stdint.h>
#include <stdlib.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#ifdef WIN32
#define ACCESS(fileName, accessMode) _access(fileName, accessMode)
#define MKDIR(path) _mkdir(path)
#else
#define ACCESS(fileName, accessMode) access(fileName, accessMode)
#define MKDIR(path) mkdir(path, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH)
#endif

/**
 * @brief Create a Directory object
 *
 * @param directoryPath
 * @return int32_t
 */
inline int32_t createDirectory(const std::string& directoryPath) {
    namespace fs = std::filesystem;
    // check ori path
    fs::path ori_path(directoryPath);
    if (fs::exists(ori_path)) {
        if (fs::is_directory(ori_path)) return 0;
        if (fs::is_regular_file(ori_path)) {
            std::cerr << ori_path << " is a file exits!" << std::endl;
            return 1;
        }
    }

    // get abs path
    std::string dir = fs::absolute(ori_path).string();
    // mkdir
    for (size_t i = 2; i < dir.size(); ++i) {
        if (dir[i] == '\\' || dir[i] == '/') {
            auto tmpdir = dir.substr(0, i);
            if (ACCESS(tmpdir.c_str(), 0)) {
                int32_t ret = MKDIR(tmpdir.c_str());
                if (ret != 0) {
                    return ret;
                }
            }
        }
    }
    if (ACCESS(dir.c_str(), 0)) {
        return MKDIR(dir.c_str());
    }
    return 0;
}

/**
 * @brief get a dll path
 *
 * @return const char*
 */
inline const char* dllPath(void) {
    Dl_info dl_info;
    int rc = dladdr((void*)dllPath, &dl_info);
    if (!rc) {
        return "";
    }
    return (dl_info.dli_fname);
}

/**
 * @brief Get the Resource object
 *
 * @param spath ori source path
 * @param isDir if this resouce is dir otherwise is file
 * @return const std::string
 */
inline const std::string getResource(const std::string& spath, bool isDir = false) {
    namespace fs = std::filesystem;
    fs::path ori_path(spath);
    if (fs::exists(ori_path)) {
        if ((!isDir) && fs::is_regular_file(ori_path)) {
            return ori_path.string();
        }
        if (isDir && fs::is_directory(ori_path)) {
            return ori_path.string();
        }
    }

    std::string dllpath(dllPath());
    if (!dllpath.empty()) {
        // check the dll path
        fs::path dpath1(dllpath);
        auto bin_dir = dpath1.parent_path();
        bin_dir.append(spath);

        if (fs::exists(bin_dir)) {
            if ((!isDir) && fs::is_regular_file(bin_dir)) {
                return bin_dir.string();
            }
            if (isDir && fs::is_directory(bin_dir)) {
                return bin_dir.string();
            }
        }

        // check dll parent
        bin_dir = dpath1.parent_path();
        bin_dir.append("../").append(spath);
        if (fs::exists(bin_dir)) {
            if ((!isDir) && fs::is_regular_file(bin_dir)) {
                return bin_dir.string();
            }
            if (isDir && fs::is_directory(bin_dir)) {
                return bin_dir.string();
            }
        }
    }

    char* env = getenv("DARTS_CONF_PATH");
    if (!env) {
        return spath;
    }
    std::string darts_env(env);
    if (!darts_env.empty()) {
        fs::path p1(darts_env);
        p1.append(spath);
        if (fs::exists(p1)) {
            if ((!isDir) && fs::is_regular_file(p1)) {
                return p1.string();
            }
            if (isDir && fs::is_directory(p1)) {
                return p1.string();
            }
        }
    }

    return spath;
}

/**
 * @brief Get the File Text object
 *
 * @param path ori path
 * @param out out string
 * @return int if success
 */
inline int getFileText(const std::string& path, std::string& out) {
    std::ifstream in(path);
    if (!in.is_open()) {
        std::cerr << "ERROR: open data " << path << " file failed " << std::endl;
        return EXIT_FAILURE;
    }
    std::stringstream sin;
    sin << in.rdbuf();
    in.close();
    out.append(sin.str());
    return EXIT_SUCCESS;
}

#endif  // SRC_UTILS_FILE_UTILS_HPP_
