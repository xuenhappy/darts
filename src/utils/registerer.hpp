/*
 * File: registerer.hpp
 * Project: utils
 * File Created: Saturday, 1st January 2022 5:16:52 pm
 * Author: Xu En (xuen@mokar.com)
 * -----
 * Last Modified: Saturday, 1st January 2022 5:17:32 pm
 * Modified By: Xu En (xuen@mokahr.com)
 * -----
 * Copyright 2021 - 2022 Your Company, Moka
 */


#ifndef SRC_UTILS_REGISTERER_HPP_
#define SRC_UTILS_REGISTERER_HPP_

#include <map>
#include <string>

namespace registerer {

// idea from boost any but make it more simple and don't use type_info.
class Any {
   public:
    Any() : content_(NULL) {}

    template <typename ValueType>
    Any(const ValueType &value)  // NOLINT
        : content_(new Holder<ValueType>(value)) {}

    Any(const Any &other) : content_(other.content_ ? other.content_->clone() : NULL) {}

    ~Any() { delete content_; }

    template <typename ValueType>
    ValueType *any_cast() {
        return content_ ? &static_cast<Holder<ValueType> *>(content_)->held_ : NULL;  // NOLINT
    }

   private:
    class PlaceHolder {
       public:
        virtual ~PlaceHolder() {}
        virtual PlaceHolder *clone() const = 0;
    };

    template <typename ValueType>
    class Holder : public PlaceHolder {
       public:
        explicit Holder(const ValueType &value) : held_(value) {}
        virtual PlaceHolder *clone() const { return new Holder(held_); }

        ValueType held_;
    };

    PlaceHolder *content_;
};

class ObjectFactory {
   public:
    ObjectFactory() {}
    virtual ~ObjectFactory() {}
    virtual Any NewInstance() { return Any(); }
    virtual Any GetSingletonInstance() { return Any(); }

   private:
};

typedef std::map<std::string, ObjectFactory *> FactoryMap;
typedef std::map<std::string, FactoryMap> BaseClassMap;


BaseClassMap &global_factory_map() {
    static BaseClassMap *factory_map = new BaseClassMap();
    return *factory_map;
}

}  // namespace registerer

#define REGISTER_REGISTERER(base_class)                                        \
    class base_class##Registerer {                                             \
        typedef ::registerer::Any Any;                                         \
        typedef ::registerer::FactoryMap FactoryMap;                           \
                                                                               \
       public:                                                                 \
        static base_class *GetInstanceByName(const ::std::string &name) {      \
            FactoryMap &map = ::registerer::global_factory_map()[#base_class]; \
            FactoryMap::iterator iter = map.find(name);                        \
            if (iter == map.end()) {                                           \
                return NULL;                                                   \
            }                                                                  \
            Any object = iter->second->NewInstance();                          \
            return *(object.any_cast<base_class *>());                         \
        }                                                                      \
        static bool IsValid(const ::std::string &name) {                       \
            FactoryMap &map = ::registerer::global_factory_map()[#base_class]; \
            return map.find(name) != map.end();                                \
        }                                                                      \
    };

#define REGISTER_CLASS(clazz, name)                                                 \
    namespace registerer {                                                          \
    class ObjectFactory##name : public ::registerer::ObjectFactory {                \
       public:                                                                      \
        ::registerer::Any NewInstance() { return ::registerer::Any(new name()); }   \
    };                                                                              \
    void __attribute__((constructor)) register_factory_##name() {                   \
        ::registerer::FactoryMap &map = ::registerer::global_factory_map()[#clazz]; \
        if (map.find(#name) == map.end()) map[#name] = new ObjectFactory##name();   \
    }                                                                               \
    }

#endif  // SRC_UTILS_REGISTERER_HPP_
