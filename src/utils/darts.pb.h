// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: darts.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_darts_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_darts_2eproto

#include <limits>
#include <string>

#include <google/protobuf/port_def.inc>
#if PROTOBUF_VERSION < 3017000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers. Please update
#error your headers.
#endif
#if 3017003 < PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers. Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/port_undef.inc>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_table_driven.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/metadata_lite.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/map.h>  // IWYU pragma: export
#include <google/protobuf/map_entry.h>
#include <google/protobuf/map_field_inl.h>
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_darts_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_darts_2eproto {
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTableField entries[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::AuxiliaryParseTableField aux[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTable schema[3]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::FieldMetadata field_metadata[];
  static const ::PROTOBUF_NAMESPACE_ID::internal::SerializationTable serialization_table[];
  static const ::PROTOBUF_NAMESPACE_ID::uint32 offsets[];
};
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_darts_2eproto;
namespace darts {
class DRegexDat;
struct DRegexDatDefaultTypeInternal;
extern DRegexDatDefaultTypeInternal _DRegexDat_default_instance_;
class DRegexDat_AItem;
struct DRegexDat_AItemDefaultTypeInternal;
extern DRegexDat_AItemDefaultTypeInternal _DRegexDat_AItem_default_instance_;
class DRegexDat_CodeMapEntry_DoNotUse;
struct DRegexDat_CodeMapEntry_DoNotUseDefaultTypeInternal;
extern DRegexDat_CodeMapEntry_DoNotUseDefaultTypeInternal _DRegexDat_CodeMapEntry_DoNotUse_default_instance_;
}  // namespace darts
PROTOBUF_NAMESPACE_OPEN
template<> ::darts::DRegexDat* Arena::CreateMaybeMessage<::darts::DRegexDat>(Arena*);
template<> ::darts::DRegexDat_AItem* Arena::CreateMaybeMessage<::darts::DRegexDat_AItem>(Arena*);
template<> ::darts::DRegexDat_CodeMapEntry_DoNotUse* Arena::CreateMaybeMessage<::darts::DRegexDat_CodeMapEntry_DoNotUse>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace darts {

// ===================================================================

class DRegexDat_AItem final :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:darts.DRegexDat.AItem) */ {
 public:
  inline DRegexDat_AItem() : DRegexDat_AItem(nullptr) {}
  ~DRegexDat_AItem() override;
  explicit constexpr DRegexDat_AItem(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  DRegexDat_AItem(const DRegexDat_AItem& from);
  DRegexDat_AItem(DRegexDat_AItem&& from) noexcept
    : DRegexDat_AItem() {
    *this = ::std::move(from);
  }

  inline DRegexDat_AItem& operator=(const DRegexDat_AItem& from) {
    CopyFrom(from);
    return *this;
  }
  inline DRegexDat_AItem& operator=(DRegexDat_AItem&& from) noexcept {
    if (this == &from) return *this;
    if (GetOwningArena() == from.GetOwningArena()) {
      InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }

  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* descriptor() {
    return GetDescriptor();
  }
  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* GetDescriptor() {
    return default_instance().GetMetadata().descriptor;
  }
  static const ::PROTOBUF_NAMESPACE_ID::Reflection* GetReflection() {
    return default_instance().GetMetadata().reflection;
  }
  static const DRegexDat_AItem& default_instance() {
    return *internal_default_instance();
  }
  static inline const DRegexDat_AItem* internal_default_instance() {
    return reinterpret_cast<const DRegexDat_AItem*>(
               &_DRegexDat_AItem_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(DRegexDat_AItem& a, DRegexDat_AItem& b) {
    a.Swap(&b);
  }
  inline void Swap(DRegexDat_AItem* other) {
    if (other == this) return;
    if (GetOwningArena() == other->GetOwningArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(DRegexDat_AItem* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetOwningArena() == other->GetOwningArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline DRegexDat_AItem* New() const final {
    return new DRegexDat_AItem();
  }

  DRegexDat_AItem* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<DRegexDat_AItem>(arena);
  }
  using ::PROTOBUF_NAMESPACE_ID::Message::CopyFrom;
  void CopyFrom(const DRegexDat_AItem& from);
  using ::PROTOBUF_NAMESPACE_ID::Message::MergeFrom;
  void MergeFrom(const DRegexDat_AItem& from);
  private:
  static void MergeImpl(::PROTOBUF_NAMESPACE_ID::Message*to, const ::PROTOBUF_NAMESPACE_ID::Message&from);
  public:
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  const char* _InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) final;
  ::PROTOBUF_NAMESPACE_ID::uint8* _InternalSerialize(
      ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(DRegexDat_AItem* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "darts.DRegexDat.AItem";
  }
  protected:
  explicit DRegexDat_AItem(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                       bool is_message_owned = false);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  static const ClassData _class_data_;
  const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*GetClassData() const final;

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kItemFieldNumber = 1,
  };
  // repeated int64 item = 1;
  int item_size() const;
  private:
  int _internal_item_size() const;
  public:
  void clear_item();
  private:
  ::PROTOBUF_NAMESPACE_ID::int64 _internal_item(int index) const;
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >&
      _internal_item() const;
  void _internal_add_item(::PROTOBUF_NAMESPACE_ID::int64 value);
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >*
      _internal_mutable_item();
  public:
  ::PROTOBUF_NAMESPACE_ID::int64 item(int index) const;
  void set_item(int index, ::PROTOBUF_NAMESPACE_ID::int64 value);
  void add_item(::PROTOBUF_NAMESPACE_ID::int64 value);
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >&
      item() const;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >*
      mutable_item();

  // @@protoc_insertion_point(class_scope:darts.DRegexDat.AItem)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 > item_;
  mutable std::atomic<int> _item_cached_byte_size_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  friend struct ::TableStruct_darts_2eproto;
};
// -------------------------------------------------------------------

class DRegexDat_CodeMapEntry_DoNotUse : public ::PROTOBUF_NAMESPACE_ID::internal::MapEntry<DRegexDat_CodeMapEntry_DoNotUse, 
    std::string, ::PROTOBUF_NAMESPACE_ID::int32,
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::TYPE_STRING,
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::TYPE_INT32> {
public:
  typedef ::PROTOBUF_NAMESPACE_ID::internal::MapEntry<DRegexDat_CodeMapEntry_DoNotUse, 
    std::string, ::PROTOBUF_NAMESPACE_ID::int32,
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::TYPE_STRING,
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::TYPE_INT32> SuperType;
  DRegexDat_CodeMapEntry_DoNotUse();
  explicit constexpr DRegexDat_CodeMapEntry_DoNotUse(
      ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);
  explicit DRegexDat_CodeMapEntry_DoNotUse(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  void MergeFrom(const DRegexDat_CodeMapEntry_DoNotUse& other);
  static const DRegexDat_CodeMapEntry_DoNotUse* internal_default_instance() { return reinterpret_cast<const DRegexDat_CodeMapEntry_DoNotUse*>(&_DRegexDat_CodeMapEntry_DoNotUse_default_instance_); }
  static bool ValidateKey(std::string* s) {
    return ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(s->data(), static_cast<int>(s->size()), ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::PARSE, "darts.DRegexDat.CodeMapEntry.key");
 }
  static bool ValidateValue(void*) { return true; }
  using ::PROTOBUF_NAMESPACE_ID::Message::MergeFrom;
  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;
};

// -------------------------------------------------------------------

class DRegexDat final :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:darts.DRegexDat) */ {
 public:
  inline DRegexDat() : DRegexDat(nullptr) {}
  ~DRegexDat() override;
  explicit constexpr DRegexDat(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  DRegexDat(const DRegexDat& from);
  DRegexDat(DRegexDat&& from) noexcept
    : DRegexDat() {
    *this = ::std::move(from);
  }

  inline DRegexDat& operator=(const DRegexDat& from) {
    CopyFrom(from);
    return *this;
  }
  inline DRegexDat& operator=(DRegexDat&& from) noexcept {
    if (this == &from) return *this;
    if (GetOwningArena() == from.GetOwningArena()) {
      InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }

  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* descriptor() {
    return GetDescriptor();
  }
  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* GetDescriptor() {
    return default_instance().GetMetadata().descriptor;
  }
  static const ::PROTOBUF_NAMESPACE_ID::Reflection* GetReflection() {
    return default_instance().GetMetadata().reflection;
  }
  static const DRegexDat& default_instance() {
    return *internal_default_instance();
  }
  static inline const DRegexDat* internal_default_instance() {
    return reinterpret_cast<const DRegexDat*>(
               &_DRegexDat_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    2;

  friend void swap(DRegexDat& a, DRegexDat& b) {
    a.Swap(&b);
  }
  inline void Swap(DRegexDat* other) {
    if (other == this) return;
    if (GetOwningArena() == other->GetOwningArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(DRegexDat* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetOwningArena() == other->GetOwningArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline DRegexDat* New() const final {
    return new DRegexDat();
  }

  DRegexDat* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<DRegexDat>(arena);
  }
  using ::PROTOBUF_NAMESPACE_ID::Message::CopyFrom;
  void CopyFrom(const DRegexDat& from);
  using ::PROTOBUF_NAMESPACE_ID::Message::MergeFrom;
  void MergeFrom(const DRegexDat& from);
  private:
  static void MergeImpl(::PROTOBUF_NAMESPACE_ID::Message*to, const ::PROTOBUF_NAMESPACE_ID::Message&from);
  public:
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  const char* _InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) final;
  ::PROTOBUF_NAMESPACE_ID::uint8* _InternalSerialize(
      ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(DRegexDat* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "darts.DRegexDat";
  }
  protected:
  explicit DRegexDat(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                       bool is_message_owned = false);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  static const ClassData _class_data_;
  const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*GetClassData() const final;

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  typedef DRegexDat_AItem AItem;

  // accessors -------------------------------------------------------

  enum : int {
    kCheckFieldNumber = 2,
    kBaseFieldNumber = 3,
    kFailFieldNumber = 4,
    kLFieldNumber = 5,
    kVFieldNumber = 6,
    kOutPutFieldNumber = 7,
    kCodeMapFieldNumber = 8,
    kMaxLenFieldNumber = 1,
  };
  // repeated int64 Check = 2;
  int check_size() const;
  private:
  int _internal_check_size() const;
  public:
  void clear_check();
  private:
  ::PROTOBUF_NAMESPACE_ID::int64 _internal_check(int index) const;
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >&
      _internal_check() const;
  void _internal_add_check(::PROTOBUF_NAMESPACE_ID::int64 value);
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >*
      _internal_mutable_check();
  public:
  ::PROTOBUF_NAMESPACE_ID::int64 check(int index) const;
  void set_check(int index, ::PROTOBUF_NAMESPACE_ID::int64 value);
  void add_check(::PROTOBUF_NAMESPACE_ID::int64 value);
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >&
      check() const;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >*
      mutable_check();

  // repeated int64 Base = 3;
  int base_size() const;
  private:
  int _internal_base_size() const;
  public:
  void clear_base();
  private:
  ::PROTOBUF_NAMESPACE_ID::int64 _internal_base(int index) const;
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >&
      _internal_base() const;
  void _internal_add_base(::PROTOBUF_NAMESPACE_ID::int64 value);
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >*
      _internal_mutable_base();
  public:
  ::PROTOBUF_NAMESPACE_ID::int64 base(int index) const;
  void set_base(int index, ::PROTOBUF_NAMESPACE_ID::int64 value);
  void add_base(::PROTOBUF_NAMESPACE_ID::int64 value);
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >&
      base() const;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >*
      mutable_base();

  // repeated int64 Fail = 4;
  int fail_size() const;
  private:
  int _internal_fail_size() const;
  public:
  void clear_fail();
  private:
  ::PROTOBUF_NAMESPACE_ID::int64 _internal_fail(int index) const;
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >&
      _internal_fail() const;
  void _internal_add_fail(::PROTOBUF_NAMESPACE_ID::int64 value);
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >*
      _internal_mutable_fail();
  public:
  ::PROTOBUF_NAMESPACE_ID::int64 fail(int index) const;
  void set_fail(int index, ::PROTOBUF_NAMESPACE_ID::int64 value);
  void add_fail(::PROTOBUF_NAMESPACE_ID::int64 value);
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >&
      fail() const;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >*
      mutable_fail();

  // repeated int64 L = 5;
  int l_size() const;
  private:
  int _internal_l_size() const;
  public:
  void clear_l();
  private:
  ::PROTOBUF_NAMESPACE_ID::int64 _internal_l(int index) const;
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >&
      _internal_l() const;
  void _internal_add_l(::PROTOBUF_NAMESPACE_ID::int64 value);
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >*
      _internal_mutable_l();
  public:
  ::PROTOBUF_NAMESPACE_ID::int64 l(int index) const;
  void set_l(int index, ::PROTOBUF_NAMESPACE_ID::int64 value);
  void add_l(::PROTOBUF_NAMESPACE_ID::int64 value);
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >&
      l() const;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >*
      mutable_l();

  // repeated .darts.DRegexDat.AItem V = 6;
  int v_size() const;
  private:
  int _internal_v_size() const;
  public:
  void clear_v();
  ::darts::DRegexDat_AItem* mutable_v(int index);
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::darts::DRegexDat_AItem >*
      mutable_v();
  private:
  const ::darts::DRegexDat_AItem& _internal_v(int index) const;
  ::darts::DRegexDat_AItem* _internal_add_v();
  public:
  const ::darts::DRegexDat_AItem& v(int index) const;
  ::darts::DRegexDat_AItem* add_v();
  const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::darts::DRegexDat_AItem >&
      v() const;

  // repeated .darts.DRegexDat.AItem OutPut = 7;
  int output_size() const;
  private:
  int _internal_output_size() const;
  public:
  void clear_output();
  ::darts::DRegexDat_AItem* mutable_output(int index);
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::darts::DRegexDat_AItem >*
      mutable_output();
  private:
  const ::darts::DRegexDat_AItem& _internal_output(int index) const;
  ::darts::DRegexDat_AItem* _internal_add_output();
  public:
  const ::darts::DRegexDat_AItem& output(int index) const;
  ::darts::DRegexDat_AItem* add_output();
  const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::darts::DRegexDat_AItem >&
      output() const;

  // map<string, int32> CodeMap = 8;
  int codemap_size() const;
  private:
  int _internal_codemap_size() const;
  public:
  void clear_codemap();
  private:
  const ::PROTOBUF_NAMESPACE_ID::Map< std::string, ::PROTOBUF_NAMESPACE_ID::int32 >&
      _internal_codemap() const;
  ::PROTOBUF_NAMESPACE_ID::Map< std::string, ::PROTOBUF_NAMESPACE_ID::int32 >*
      _internal_mutable_codemap();
  public:
  const ::PROTOBUF_NAMESPACE_ID::Map< std::string, ::PROTOBUF_NAMESPACE_ID::int32 >&
      codemap() const;
  ::PROTOBUF_NAMESPACE_ID::Map< std::string, ::PROTOBUF_NAMESPACE_ID::int32 >*
      mutable_codemap();

  // int32 MaxLen = 1;
  void clear_maxlen();
  ::PROTOBUF_NAMESPACE_ID::int32 maxlen() const;
  void set_maxlen(::PROTOBUF_NAMESPACE_ID::int32 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int32 _internal_maxlen() const;
  void _internal_set_maxlen(::PROTOBUF_NAMESPACE_ID::int32 value);
  public:

  // @@protoc_insertion_point(class_scope:darts.DRegexDat)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 > check_;
  mutable std::atomic<int> _check_cached_byte_size_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 > base_;
  mutable std::atomic<int> _base_cached_byte_size_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 > fail_;
  mutable std::atomic<int> _fail_cached_byte_size_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 > l_;
  mutable std::atomic<int> _l_cached_byte_size_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::darts::DRegexDat_AItem > v_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::darts::DRegexDat_AItem > output_;
  ::PROTOBUF_NAMESPACE_ID::internal::MapField<
      DRegexDat_CodeMapEntry_DoNotUse,
      std::string, ::PROTOBUF_NAMESPACE_ID::int32,
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::TYPE_STRING,
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::TYPE_INT32> codemap_;
  ::PROTOBUF_NAMESPACE_ID::int32 maxlen_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  friend struct ::TableStruct_darts_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// DRegexDat_AItem

// repeated int64 item = 1;
inline int DRegexDat_AItem::_internal_item_size() const {
  return item_.size();
}
inline int DRegexDat_AItem::item_size() const {
  return _internal_item_size();
}
inline void DRegexDat_AItem::clear_item() {
  item_.Clear();
}
inline ::PROTOBUF_NAMESPACE_ID::int64 DRegexDat_AItem::_internal_item(int index) const {
  return item_.Get(index);
}
inline ::PROTOBUF_NAMESPACE_ID::int64 DRegexDat_AItem::item(int index) const {
  // @@protoc_insertion_point(field_get:darts.DRegexDat.AItem.item)
  return _internal_item(index);
}
inline void DRegexDat_AItem::set_item(int index, ::PROTOBUF_NAMESPACE_ID::int64 value) {
  item_.Set(index, value);
  // @@protoc_insertion_point(field_set:darts.DRegexDat.AItem.item)
}
inline void DRegexDat_AItem::_internal_add_item(::PROTOBUF_NAMESPACE_ID::int64 value) {
  item_.Add(value);
}
inline void DRegexDat_AItem::add_item(::PROTOBUF_NAMESPACE_ID::int64 value) {
  _internal_add_item(value);
  // @@protoc_insertion_point(field_add:darts.DRegexDat.AItem.item)
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >&
DRegexDat_AItem::_internal_item() const {
  return item_;
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >&
DRegexDat_AItem::item() const {
  // @@protoc_insertion_point(field_list:darts.DRegexDat.AItem.item)
  return _internal_item();
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >*
DRegexDat_AItem::_internal_mutable_item() {
  return &item_;
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >*
DRegexDat_AItem::mutable_item() {
  // @@protoc_insertion_point(field_mutable_list:darts.DRegexDat.AItem.item)
  return _internal_mutable_item();
}

// -------------------------------------------------------------------

// -------------------------------------------------------------------

// DRegexDat

// int32 MaxLen = 1;
inline void DRegexDat::clear_maxlen() {
  maxlen_ = 0;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 DRegexDat::_internal_maxlen() const {
  return maxlen_;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 DRegexDat::maxlen() const {
  // @@protoc_insertion_point(field_get:darts.DRegexDat.MaxLen)
  return _internal_maxlen();
}
inline void DRegexDat::_internal_set_maxlen(::PROTOBUF_NAMESPACE_ID::int32 value) {
  
  maxlen_ = value;
}
inline void DRegexDat::set_maxlen(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _internal_set_maxlen(value);
  // @@protoc_insertion_point(field_set:darts.DRegexDat.MaxLen)
}

// repeated int64 Check = 2;
inline int DRegexDat::_internal_check_size() const {
  return check_.size();
}
inline int DRegexDat::check_size() const {
  return _internal_check_size();
}
inline void DRegexDat::clear_check() {
  check_.Clear();
}
inline ::PROTOBUF_NAMESPACE_ID::int64 DRegexDat::_internal_check(int index) const {
  return check_.Get(index);
}
inline ::PROTOBUF_NAMESPACE_ID::int64 DRegexDat::check(int index) const {
  // @@protoc_insertion_point(field_get:darts.DRegexDat.Check)
  return _internal_check(index);
}
inline void DRegexDat::set_check(int index, ::PROTOBUF_NAMESPACE_ID::int64 value) {
  check_.Set(index, value);
  // @@protoc_insertion_point(field_set:darts.DRegexDat.Check)
}
inline void DRegexDat::_internal_add_check(::PROTOBUF_NAMESPACE_ID::int64 value) {
  check_.Add(value);
}
inline void DRegexDat::add_check(::PROTOBUF_NAMESPACE_ID::int64 value) {
  _internal_add_check(value);
  // @@protoc_insertion_point(field_add:darts.DRegexDat.Check)
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >&
DRegexDat::_internal_check() const {
  return check_;
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >&
DRegexDat::check() const {
  // @@protoc_insertion_point(field_list:darts.DRegexDat.Check)
  return _internal_check();
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >*
DRegexDat::_internal_mutable_check() {
  return &check_;
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >*
DRegexDat::mutable_check() {
  // @@protoc_insertion_point(field_mutable_list:darts.DRegexDat.Check)
  return _internal_mutable_check();
}

// repeated int64 Base = 3;
inline int DRegexDat::_internal_base_size() const {
  return base_.size();
}
inline int DRegexDat::base_size() const {
  return _internal_base_size();
}
inline void DRegexDat::clear_base() {
  base_.Clear();
}
inline ::PROTOBUF_NAMESPACE_ID::int64 DRegexDat::_internal_base(int index) const {
  return base_.Get(index);
}
inline ::PROTOBUF_NAMESPACE_ID::int64 DRegexDat::base(int index) const {
  // @@protoc_insertion_point(field_get:darts.DRegexDat.Base)
  return _internal_base(index);
}
inline void DRegexDat::set_base(int index, ::PROTOBUF_NAMESPACE_ID::int64 value) {
  base_.Set(index, value);
  // @@protoc_insertion_point(field_set:darts.DRegexDat.Base)
}
inline void DRegexDat::_internal_add_base(::PROTOBUF_NAMESPACE_ID::int64 value) {
  base_.Add(value);
}
inline void DRegexDat::add_base(::PROTOBUF_NAMESPACE_ID::int64 value) {
  _internal_add_base(value);
  // @@protoc_insertion_point(field_add:darts.DRegexDat.Base)
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >&
DRegexDat::_internal_base() const {
  return base_;
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >&
DRegexDat::base() const {
  // @@protoc_insertion_point(field_list:darts.DRegexDat.Base)
  return _internal_base();
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >*
DRegexDat::_internal_mutable_base() {
  return &base_;
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >*
DRegexDat::mutable_base() {
  // @@protoc_insertion_point(field_mutable_list:darts.DRegexDat.Base)
  return _internal_mutable_base();
}

// repeated int64 Fail = 4;
inline int DRegexDat::_internal_fail_size() const {
  return fail_.size();
}
inline int DRegexDat::fail_size() const {
  return _internal_fail_size();
}
inline void DRegexDat::clear_fail() {
  fail_.Clear();
}
inline ::PROTOBUF_NAMESPACE_ID::int64 DRegexDat::_internal_fail(int index) const {
  return fail_.Get(index);
}
inline ::PROTOBUF_NAMESPACE_ID::int64 DRegexDat::fail(int index) const {
  // @@protoc_insertion_point(field_get:darts.DRegexDat.Fail)
  return _internal_fail(index);
}
inline void DRegexDat::set_fail(int index, ::PROTOBUF_NAMESPACE_ID::int64 value) {
  fail_.Set(index, value);
  // @@protoc_insertion_point(field_set:darts.DRegexDat.Fail)
}
inline void DRegexDat::_internal_add_fail(::PROTOBUF_NAMESPACE_ID::int64 value) {
  fail_.Add(value);
}
inline void DRegexDat::add_fail(::PROTOBUF_NAMESPACE_ID::int64 value) {
  _internal_add_fail(value);
  // @@protoc_insertion_point(field_add:darts.DRegexDat.Fail)
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >&
DRegexDat::_internal_fail() const {
  return fail_;
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >&
DRegexDat::fail() const {
  // @@protoc_insertion_point(field_list:darts.DRegexDat.Fail)
  return _internal_fail();
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >*
DRegexDat::_internal_mutable_fail() {
  return &fail_;
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >*
DRegexDat::mutable_fail() {
  // @@protoc_insertion_point(field_mutable_list:darts.DRegexDat.Fail)
  return _internal_mutable_fail();
}

// repeated int64 L = 5;
inline int DRegexDat::_internal_l_size() const {
  return l_.size();
}
inline int DRegexDat::l_size() const {
  return _internal_l_size();
}
inline void DRegexDat::clear_l() {
  l_.Clear();
}
inline ::PROTOBUF_NAMESPACE_ID::int64 DRegexDat::_internal_l(int index) const {
  return l_.Get(index);
}
inline ::PROTOBUF_NAMESPACE_ID::int64 DRegexDat::l(int index) const {
  // @@protoc_insertion_point(field_get:darts.DRegexDat.L)
  return _internal_l(index);
}
inline void DRegexDat::set_l(int index, ::PROTOBUF_NAMESPACE_ID::int64 value) {
  l_.Set(index, value);
  // @@protoc_insertion_point(field_set:darts.DRegexDat.L)
}
inline void DRegexDat::_internal_add_l(::PROTOBUF_NAMESPACE_ID::int64 value) {
  l_.Add(value);
}
inline void DRegexDat::add_l(::PROTOBUF_NAMESPACE_ID::int64 value) {
  _internal_add_l(value);
  // @@protoc_insertion_point(field_add:darts.DRegexDat.L)
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >&
DRegexDat::_internal_l() const {
  return l_;
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >&
DRegexDat::l() const {
  // @@protoc_insertion_point(field_list:darts.DRegexDat.L)
  return _internal_l();
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >*
DRegexDat::_internal_mutable_l() {
  return &l_;
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >*
DRegexDat::mutable_l() {
  // @@protoc_insertion_point(field_mutable_list:darts.DRegexDat.L)
  return _internal_mutable_l();
}

// repeated .darts.DRegexDat.AItem V = 6;
inline int DRegexDat::_internal_v_size() const {
  return v_.size();
}
inline int DRegexDat::v_size() const {
  return _internal_v_size();
}
inline void DRegexDat::clear_v() {
  v_.Clear();
}
inline ::darts::DRegexDat_AItem* DRegexDat::mutable_v(int index) {
  // @@protoc_insertion_point(field_mutable:darts.DRegexDat.V)
  return v_.Mutable(index);
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::darts::DRegexDat_AItem >*
DRegexDat::mutable_v() {
  // @@protoc_insertion_point(field_mutable_list:darts.DRegexDat.V)
  return &v_;
}
inline const ::darts::DRegexDat_AItem& DRegexDat::_internal_v(int index) const {
  return v_.Get(index);
}
inline const ::darts::DRegexDat_AItem& DRegexDat::v(int index) const {
  // @@protoc_insertion_point(field_get:darts.DRegexDat.V)
  return _internal_v(index);
}
inline ::darts::DRegexDat_AItem* DRegexDat::_internal_add_v() {
  return v_.Add();
}
inline ::darts::DRegexDat_AItem* DRegexDat::add_v() {
  ::darts::DRegexDat_AItem* _add = _internal_add_v();
  // @@protoc_insertion_point(field_add:darts.DRegexDat.V)
  return _add;
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::darts::DRegexDat_AItem >&
DRegexDat::v() const {
  // @@protoc_insertion_point(field_list:darts.DRegexDat.V)
  return v_;
}

// repeated .darts.DRegexDat.AItem OutPut = 7;
inline int DRegexDat::_internal_output_size() const {
  return output_.size();
}
inline int DRegexDat::output_size() const {
  return _internal_output_size();
}
inline void DRegexDat::clear_output() {
  output_.Clear();
}
inline ::darts::DRegexDat_AItem* DRegexDat::mutable_output(int index) {
  // @@protoc_insertion_point(field_mutable:darts.DRegexDat.OutPut)
  return output_.Mutable(index);
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::darts::DRegexDat_AItem >*
DRegexDat::mutable_output() {
  // @@protoc_insertion_point(field_mutable_list:darts.DRegexDat.OutPut)
  return &output_;
}
inline const ::darts::DRegexDat_AItem& DRegexDat::_internal_output(int index) const {
  return output_.Get(index);
}
inline const ::darts::DRegexDat_AItem& DRegexDat::output(int index) const {
  // @@protoc_insertion_point(field_get:darts.DRegexDat.OutPut)
  return _internal_output(index);
}
inline ::darts::DRegexDat_AItem* DRegexDat::_internal_add_output() {
  return output_.Add();
}
inline ::darts::DRegexDat_AItem* DRegexDat::add_output() {
  ::darts::DRegexDat_AItem* _add = _internal_add_output();
  // @@protoc_insertion_point(field_add:darts.DRegexDat.OutPut)
  return _add;
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::darts::DRegexDat_AItem >&
DRegexDat::output() const {
  // @@protoc_insertion_point(field_list:darts.DRegexDat.OutPut)
  return output_;
}

// map<string, int32> CodeMap = 8;
inline int DRegexDat::_internal_codemap_size() const {
  return codemap_.size();
}
inline int DRegexDat::codemap_size() const {
  return _internal_codemap_size();
}
inline void DRegexDat::clear_codemap() {
  codemap_.Clear();
}
inline const ::PROTOBUF_NAMESPACE_ID::Map< std::string, ::PROTOBUF_NAMESPACE_ID::int32 >&
DRegexDat::_internal_codemap() const {
  return codemap_.GetMap();
}
inline const ::PROTOBUF_NAMESPACE_ID::Map< std::string, ::PROTOBUF_NAMESPACE_ID::int32 >&
DRegexDat::codemap() const {
  // @@protoc_insertion_point(field_map:darts.DRegexDat.CodeMap)
  return _internal_codemap();
}
inline ::PROTOBUF_NAMESPACE_ID::Map< std::string, ::PROTOBUF_NAMESPACE_ID::int32 >*
DRegexDat::_internal_mutable_codemap() {
  return codemap_.MutableMap();
}
inline ::PROTOBUF_NAMESPACE_ID::Map< std::string, ::PROTOBUF_NAMESPACE_ID::int32 >*
DRegexDat::mutable_codemap() {
  // @@protoc_insertion_point(field_mutable_map:darts.DRegexDat.CodeMap)
  return _internal_mutable_codemap();
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__
// -------------------------------------------------------------------

// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

}  // namespace darts

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_darts_2eproto
