find_package(PkgConfig REQUIRED)
pkg_check_modules(Jsoncpp_PKGCONF jsoncpp)

find_path(
  Jsoncpp_INCLUDE_DIR
  NAMES json/json.h
  PATH_SUFFIXES jsoncpp
  PATHS ${Jsoncpp_PKGCONF_INCLUDE_DIRS} # /usr/include/jsoncpp/json
)

find_library(
  Jsoncpp_LIBRARY
  NAMES jsoncpp
  PATHS ${Jsoncpp_PKGCONF_LIBRARY_DIRS})
