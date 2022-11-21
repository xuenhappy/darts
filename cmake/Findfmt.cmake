find_package(PkgConfig REQUIRED)
pkg_check_modules(fmt_PKGCONF fmt)

find_path(
  FMT_INCLUDE_DIR
  NAMES fmt/core.h
  PATH_SUFFIXES fmt
  PATHS ${fmt_PKGCONF_INCLUDE_DIRS})

find_library(
  FMT_LIBRARY
  NAMES fmt
  PATHS ${fmt_PKGCONF_LIBRARY_DIRS})
