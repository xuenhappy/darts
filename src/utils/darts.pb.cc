// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: darts.proto

#include "darts.pb.h"

#include <algorithm>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/wire_format_lite.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>

PROTOBUF_PRAGMA_INIT_SEG
namespace darts {
constexpr DRegexDat_AItem::DRegexDat_AItem(
  ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized)
  : item_()
  , _item_cached_byte_size_(0){}
struct DRegexDat_AItemDefaultTypeInternal {
  constexpr DRegexDat_AItemDefaultTypeInternal()
    : _instance(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized{}) {}
  ~DRegexDat_AItemDefaultTypeInternal() {}
  union {
    DRegexDat_AItem _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT DRegexDat_AItemDefaultTypeInternal _DRegexDat_AItem_default_instance_;
constexpr DRegexDat_CodeMapEntry_DoNotUse::DRegexDat_CodeMapEntry_DoNotUse(
  ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized){}
struct DRegexDat_CodeMapEntry_DoNotUseDefaultTypeInternal {
  constexpr DRegexDat_CodeMapEntry_DoNotUseDefaultTypeInternal()
    : _instance(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized{}) {}
  ~DRegexDat_CodeMapEntry_DoNotUseDefaultTypeInternal() {}
  union {
    DRegexDat_CodeMapEntry_DoNotUse _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT DRegexDat_CodeMapEntry_DoNotUseDefaultTypeInternal _DRegexDat_CodeMapEntry_DoNotUse_default_instance_;
constexpr DRegexDat::DRegexDat(
  ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized)
  : check_()
  , _check_cached_byte_size_(0)
  , base_()
  , _base_cached_byte_size_(0)
  , fail_()
  , _fail_cached_byte_size_(0)
  , l_()
  , _l_cached_byte_size_(0)
  , v_()
  , output_()
  , codemap_(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized{})
  , maxlen_(0){}
struct DRegexDatDefaultTypeInternal {
  constexpr DRegexDatDefaultTypeInternal()
    : _instance(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized{}) {}
  ~DRegexDatDefaultTypeInternal() {}
  union {
    DRegexDat _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT DRegexDatDefaultTypeInternal _DRegexDat_default_instance_;
}  // namespace darts
static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_darts_2eproto[3];
static constexpr ::PROTOBUF_NAMESPACE_ID::EnumDescriptor const** file_level_enum_descriptors_darts_2eproto = nullptr;
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_darts_2eproto = nullptr;

const ::PROTOBUF_NAMESPACE_ID::uint32 TableStruct_darts_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::darts::DRegexDat_AItem, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::darts::DRegexDat_AItem, item_),
  PROTOBUF_FIELD_OFFSET(::darts::DRegexDat_CodeMapEntry_DoNotUse, _has_bits_),
  PROTOBUF_FIELD_OFFSET(::darts::DRegexDat_CodeMapEntry_DoNotUse, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::darts::DRegexDat_CodeMapEntry_DoNotUse, key_),
  PROTOBUF_FIELD_OFFSET(::darts::DRegexDat_CodeMapEntry_DoNotUse, value_),
  0,
  1,
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::darts::DRegexDat, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::darts::DRegexDat, maxlen_),
  PROTOBUF_FIELD_OFFSET(::darts::DRegexDat, check_),
  PROTOBUF_FIELD_OFFSET(::darts::DRegexDat, base_),
  PROTOBUF_FIELD_OFFSET(::darts::DRegexDat, fail_),
  PROTOBUF_FIELD_OFFSET(::darts::DRegexDat, l_),
  PROTOBUF_FIELD_OFFSET(::darts::DRegexDat, v_),
  PROTOBUF_FIELD_OFFSET(::darts::DRegexDat, output_),
  PROTOBUF_FIELD_OFFSET(::darts::DRegexDat, codemap_),
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, sizeof(::darts::DRegexDat_AItem)},
  { 6, 13, sizeof(::darts::DRegexDat_CodeMapEntry_DoNotUse)},
  { 15, -1, sizeof(::darts::DRegexDat)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::darts::_DRegexDat_AItem_default_instance_),
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::darts::_DRegexDat_CodeMapEntry_DoNotUse_default_instance_),
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::darts::_DRegexDat_default_instance_),
};

const char descriptor_table_protodef_darts_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n\013darts.proto\022\005darts\"\223\002\n\tDRegexDat\022\016\n\006Ma"
  "xLen\030\001 \001(\005\022\r\n\005Check\030\002 \003(\003\022\014\n\004Base\030\003 \003(\003\022"
  "\014\n\004Fail\030\004 \003(\003\022\t\n\001L\030\005 \003(\003\022!\n\001V\030\006 \003(\0132\026.da"
  "rts.DRegexDat.AItem\022&\n\006OutPut\030\007 \003(\0132\026.da"
  "rts.DRegexDat.AItem\022.\n\007CodeMap\030\010 \003(\0132\035.d"
  "arts.DRegexDat.CodeMapEntry\032\025\n\005AItem\022\014\n\004"
  "item\030\001 \003(\003\032.\n\014CodeMapEntry\022\013\n\003key\030\001 \001(\t\022"
  "\r\n\005value\030\002 \001(\005:\0028\001b\006proto3"
  ;
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_darts_2eproto_once;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_darts_2eproto = {
  false, false, 306, descriptor_table_protodef_darts_2eproto, "darts.proto", 
  &descriptor_table_darts_2eproto_once, nullptr, 0, 3,
  schemas, file_default_instances, TableStruct_darts_2eproto::offsets,
  file_level_metadata_darts_2eproto, file_level_enum_descriptors_darts_2eproto, file_level_service_descriptors_darts_2eproto,
};
PROTOBUF_ATTRIBUTE_WEAK const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable* descriptor_table_darts_2eproto_getter() {
  return &descriptor_table_darts_2eproto;
}

// Force running AddDescriptors() at dynamic initialization time.
PROTOBUF_ATTRIBUTE_INIT_PRIORITY static ::PROTOBUF_NAMESPACE_ID::internal::AddDescriptorsRunner dynamic_init_dummy_darts_2eproto(&descriptor_table_darts_2eproto);
namespace darts {

// ===================================================================

class DRegexDat_AItem::_Internal {
 public:
};

DRegexDat_AItem::DRegexDat_AItem(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena, is_message_owned),
  item_(arena) {
  SharedCtor();
  if (!is_message_owned) {
    RegisterArenaDtor(arena);
  }
  // @@protoc_insertion_point(arena_constructor:darts.DRegexDat.AItem)
}
DRegexDat_AItem::DRegexDat_AItem(const DRegexDat_AItem& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      item_(from.item_) {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  // @@protoc_insertion_point(copy_constructor:darts.DRegexDat.AItem)
}

inline void DRegexDat_AItem::SharedCtor() {
}

DRegexDat_AItem::~DRegexDat_AItem() {
  // @@protoc_insertion_point(destructor:darts.DRegexDat.AItem)
  if (GetArenaForAllocation() != nullptr) return;
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

inline void DRegexDat_AItem::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
}

void DRegexDat_AItem::ArenaDtor(void* object) {
  DRegexDat_AItem* _this = reinterpret_cast< DRegexDat_AItem* >(object);
  (void)_this;
}
void DRegexDat_AItem::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void DRegexDat_AItem::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void DRegexDat_AItem::Clear() {
// @@protoc_insertion_point(message_clear_start:darts.DRegexDat.AItem)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  item_.Clear();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* DRegexDat_AItem::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // repeated int64 item = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 10)) {
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::PackedInt64Parser(_internal_mutable_item(), ptr, ctx);
          CHK_(ptr);
        } else if (static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 8) {
          _internal_add_item(::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr));
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      default: {
      handle_unusual:
        if ((tag == 0) || ((tag & 7) == 4)) {
          CHK_(ptr);
          ctx->SetLastTag(tag);
          goto success;
        }
        ptr = UnknownFieldParse(tag,
            _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(),
            ptr, ctx);
        CHK_(ptr != nullptr);
        continue;
      }
    }  // switch
  }  // while
success:
  return ptr;
failure:
  ptr = nullptr;
  goto success;
#undef CHK_
}

::PROTOBUF_NAMESPACE_ID::uint8* DRegexDat_AItem::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:darts.DRegexDat.AItem)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // repeated int64 item = 1;
  {
    int byte_size = _item_cached_byte_size_.load(std::memory_order_relaxed);
    if (byte_size > 0) {
      target = stream->WriteInt64Packed(
          1, _internal_item(), byte_size, target);
    }
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:darts.DRegexDat.AItem)
  return target;
}

size_t DRegexDat_AItem::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:darts.DRegexDat.AItem)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // repeated int64 item = 1;
  {
    size_t data_size = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      Int64Size(this->item_);
    if (data_size > 0) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
            static_cast<::PROTOBUF_NAMESPACE_ID::int32>(data_size));
    }
    int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(data_size);
    _item_cached_byte_size_.store(cached_size,
                                    std::memory_order_relaxed);
    total_size += data_size;
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    return ::PROTOBUF_NAMESPACE_ID::internal::ComputeUnknownFieldsSize(
        _internal_metadata_, total_size, &_cached_size_);
  }
  int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

const ::PROTOBUF_NAMESPACE_ID::Message::ClassData DRegexDat_AItem::_class_data_ = {
    ::PROTOBUF_NAMESPACE_ID::Message::CopyWithSizeCheck,
    DRegexDat_AItem::MergeImpl
};
const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*DRegexDat_AItem::GetClassData() const { return &_class_data_; }

void DRegexDat_AItem::MergeImpl(::PROTOBUF_NAMESPACE_ID::Message*to,
                      const ::PROTOBUF_NAMESPACE_ID::Message&from) {
  static_cast<DRegexDat_AItem *>(to)->MergeFrom(
      static_cast<const DRegexDat_AItem &>(from));
}


void DRegexDat_AItem::MergeFrom(const DRegexDat_AItem& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:darts.DRegexDat.AItem)
  GOOGLE_DCHECK_NE(&from, this);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  item_.MergeFrom(from.item_);
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
}

void DRegexDat_AItem::CopyFrom(const DRegexDat_AItem& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:darts.DRegexDat.AItem)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool DRegexDat_AItem::IsInitialized() const {
  return true;
}

void DRegexDat_AItem::InternalSwap(DRegexDat_AItem* other) {
  using std::swap;
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  item_.InternalSwap(&other->item_);
}

::PROTOBUF_NAMESPACE_ID::Metadata DRegexDat_AItem::GetMetadata() const {
  return ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(
      &descriptor_table_darts_2eproto_getter, &descriptor_table_darts_2eproto_once,
      file_level_metadata_darts_2eproto[0]);
}

// ===================================================================

DRegexDat_CodeMapEntry_DoNotUse::DRegexDat_CodeMapEntry_DoNotUse() {}
DRegexDat_CodeMapEntry_DoNotUse::DRegexDat_CodeMapEntry_DoNotUse(::PROTOBUF_NAMESPACE_ID::Arena* arena)
    : SuperType(arena) {}
void DRegexDat_CodeMapEntry_DoNotUse::MergeFrom(const DRegexDat_CodeMapEntry_DoNotUse& other) {
  MergeFromInternal(other);
}
::PROTOBUF_NAMESPACE_ID::Metadata DRegexDat_CodeMapEntry_DoNotUse::GetMetadata() const {
  return ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(
      &descriptor_table_darts_2eproto_getter, &descriptor_table_darts_2eproto_once,
      file_level_metadata_darts_2eproto[1]);
}

// ===================================================================

class DRegexDat::_Internal {
 public:
};

DRegexDat::DRegexDat(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena, is_message_owned),
  check_(arena),
  base_(arena),
  fail_(arena),
  l_(arena),
  v_(arena),
  output_(arena),
  codemap_(arena) {
  SharedCtor();
  if (!is_message_owned) {
    RegisterArenaDtor(arena);
  }
  // @@protoc_insertion_point(arena_constructor:darts.DRegexDat)
}
DRegexDat::DRegexDat(const DRegexDat& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      check_(from.check_),
      base_(from.base_),
      fail_(from.fail_),
      l_(from.l_),
      v_(from.v_),
      output_(from.output_) {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  codemap_.MergeFrom(from.codemap_);
  maxlen_ = from.maxlen_;
  // @@protoc_insertion_point(copy_constructor:darts.DRegexDat)
}

inline void DRegexDat::SharedCtor() {
maxlen_ = 0;
}

DRegexDat::~DRegexDat() {
  // @@protoc_insertion_point(destructor:darts.DRegexDat)
  if (GetArenaForAllocation() != nullptr) return;
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

inline void DRegexDat::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
}

void DRegexDat::ArenaDtor(void* object) {
  DRegexDat* _this = reinterpret_cast< DRegexDat* >(object);
  (void)_this;
  _this->codemap_. ~MapField();
}
inline void DRegexDat::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena) {
  if (arena != nullptr) {
    arena->OwnCustomDestructor(this, &DRegexDat::ArenaDtor);
  }
}
void DRegexDat::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void DRegexDat::Clear() {
// @@protoc_insertion_point(message_clear_start:darts.DRegexDat)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  check_.Clear();
  base_.Clear();
  fail_.Clear();
  l_.Clear();
  v_.Clear();
  output_.Clear();
  codemap_.Clear();
  maxlen_ = 0;
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* DRegexDat::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // int32 MaxLen = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 8)) {
          maxlen_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // repeated int64 Check = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 18)) {
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::PackedInt64Parser(_internal_mutable_check(), ptr, ctx);
          CHK_(ptr);
        } else if (static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 16) {
          _internal_add_check(::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr));
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // repeated int64 Base = 3;
      case 3:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 26)) {
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::PackedInt64Parser(_internal_mutable_base(), ptr, ctx);
          CHK_(ptr);
        } else if (static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 24) {
          _internal_add_base(::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr));
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // repeated int64 Fail = 4;
      case 4:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 34)) {
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::PackedInt64Parser(_internal_mutable_fail(), ptr, ctx);
          CHK_(ptr);
        } else if (static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 32) {
          _internal_add_fail(::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr));
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // repeated int64 L = 5;
      case 5:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 42)) {
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::PackedInt64Parser(_internal_mutable_l(), ptr, ctx);
          CHK_(ptr);
        } else if (static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 40) {
          _internal_add_l(::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr));
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // repeated .darts.DRegexDat.AItem V = 6;
      case 6:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 50)) {
          ptr -= 1;
          do {
            ptr += 1;
            ptr = ctx->ParseMessage(_internal_add_v(), ptr);
            CHK_(ptr);
            if (!ctx->DataAvailable(ptr)) break;
          } while (::PROTOBUF_NAMESPACE_ID::internal::ExpectTag<50>(ptr));
        } else goto handle_unusual;
        continue;
      // repeated .darts.DRegexDat.AItem OutPut = 7;
      case 7:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 58)) {
          ptr -= 1;
          do {
            ptr += 1;
            ptr = ctx->ParseMessage(_internal_add_output(), ptr);
            CHK_(ptr);
            if (!ctx->DataAvailable(ptr)) break;
          } while (::PROTOBUF_NAMESPACE_ID::internal::ExpectTag<58>(ptr));
        } else goto handle_unusual;
        continue;
      // map<string, int32> CodeMap = 8;
      case 8:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 66)) {
          ptr -= 1;
          do {
            ptr += 1;
            ptr = ctx->ParseMessage(&codemap_, ptr);
            CHK_(ptr);
            if (!ctx->DataAvailable(ptr)) break;
          } while (::PROTOBUF_NAMESPACE_ID::internal::ExpectTag<66>(ptr));
        } else goto handle_unusual;
        continue;
      default: {
      handle_unusual:
        if ((tag == 0) || ((tag & 7) == 4)) {
          CHK_(ptr);
          ctx->SetLastTag(tag);
          goto success;
        }
        ptr = UnknownFieldParse(tag,
            _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(),
            ptr, ctx);
        CHK_(ptr != nullptr);
        continue;
      }
    }  // switch
  }  // while
success:
  return ptr;
failure:
  ptr = nullptr;
  goto success;
#undef CHK_
}

::PROTOBUF_NAMESPACE_ID::uint8* DRegexDat::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:darts.DRegexDat)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // int32 MaxLen = 1;
  if (this->_internal_maxlen() != 0) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(1, this->_internal_maxlen(), target);
  }

  // repeated int64 Check = 2;
  {
    int byte_size = _check_cached_byte_size_.load(std::memory_order_relaxed);
    if (byte_size > 0) {
      target = stream->WriteInt64Packed(
          2, _internal_check(), byte_size, target);
    }
  }

  // repeated int64 Base = 3;
  {
    int byte_size = _base_cached_byte_size_.load(std::memory_order_relaxed);
    if (byte_size > 0) {
      target = stream->WriteInt64Packed(
          3, _internal_base(), byte_size, target);
    }
  }

  // repeated int64 Fail = 4;
  {
    int byte_size = _fail_cached_byte_size_.load(std::memory_order_relaxed);
    if (byte_size > 0) {
      target = stream->WriteInt64Packed(
          4, _internal_fail(), byte_size, target);
    }
  }

  // repeated int64 L = 5;
  {
    int byte_size = _l_cached_byte_size_.load(std::memory_order_relaxed);
    if (byte_size > 0) {
      target = stream->WriteInt64Packed(
          5, _internal_l(), byte_size, target);
    }
  }

  // repeated .darts.DRegexDat.AItem V = 6;
  for (unsigned int i = 0,
      n = static_cast<unsigned int>(this->_internal_v_size()); i < n; i++) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      InternalWriteMessage(6, this->_internal_v(i), target, stream);
  }

  // repeated .darts.DRegexDat.AItem OutPut = 7;
  for (unsigned int i = 0,
      n = static_cast<unsigned int>(this->_internal_output_size()); i < n; i++) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      InternalWriteMessage(7, this->_internal_output(i), target, stream);
  }

  // map<string, int32> CodeMap = 8;
  if (!this->_internal_codemap().empty()) {
    typedef ::PROTOBUF_NAMESPACE_ID::Map< std::string, ::PROTOBUF_NAMESPACE_ID::int32 >::const_pointer
        ConstPtr;
    typedef ConstPtr SortItem;
    typedef ::PROTOBUF_NAMESPACE_ID::internal::CompareByDerefFirst<SortItem> Less;
    struct Utf8Check {
      static void Check(ConstPtr p) {
        (void)p;
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
          p->first.data(), static_cast<int>(p->first.length()),
          ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
          "darts.DRegexDat.CodeMapEntry.key");
      }
    };

    if (stream->IsSerializationDeterministic() &&
        this->_internal_codemap().size() > 1) {
      ::std::unique_ptr<SortItem[]> items(
          new SortItem[this->_internal_codemap().size()]);
      typedef ::PROTOBUF_NAMESPACE_ID::Map< std::string, ::PROTOBUF_NAMESPACE_ID::int32 >::size_type size_type;
      size_type n = 0;
      for (::PROTOBUF_NAMESPACE_ID::Map< std::string, ::PROTOBUF_NAMESPACE_ID::int32 >::const_iterator
          it = this->_internal_codemap().begin();
          it != this->_internal_codemap().end(); ++it, ++n) {
        items[static_cast<ptrdiff_t>(n)] = SortItem(&*it);
      }
      ::std::sort(&items[0], &items[static_cast<ptrdiff_t>(n)], Less());
      for (size_type i = 0; i < n; i++) {
        target = DRegexDat_CodeMapEntry_DoNotUse::Funcs::InternalSerialize(8, items[static_cast<ptrdiff_t>(i)]->first, items[static_cast<ptrdiff_t>(i)]->second, target, stream);
        Utf8Check::Check(&(*items[static_cast<ptrdiff_t>(i)]));
      }
    } else {
      for (::PROTOBUF_NAMESPACE_ID::Map< std::string, ::PROTOBUF_NAMESPACE_ID::int32 >::const_iterator
          it = this->_internal_codemap().begin();
          it != this->_internal_codemap().end(); ++it) {
        target = DRegexDat_CodeMapEntry_DoNotUse::Funcs::InternalSerialize(8, it->first, it->second, target, stream);
        Utf8Check::Check(&(*it));
      }
    }
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:darts.DRegexDat)
  return target;
}

size_t DRegexDat::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:darts.DRegexDat)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // repeated int64 Check = 2;
  {
    size_t data_size = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      Int64Size(this->check_);
    if (data_size > 0) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
            static_cast<::PROTOBUF_NAMESPACE_ID::int32>(data_size));
    }
    int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(data_size);
    _check_cached_byte_size_.store(cached_size,
                                    std::memory_order_relaxed);
    total_size += data_size;
  }

  // repeated int64 Base = 3;
  {
    size_t data_size = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      Int64Size(this->base_);
    if (data_size > 0) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
            static_cast<::PROTOBUF_NAMESPACE_ID::int32>(data_size));
    }
    int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(data_size);
    _base_cached_byte_size_.store(cached_size,
                                    std::memory_order_relaxed);
    total_size += data_size;
  }

  // repeated int64 Fail = 4;
  {
    size_t data_size = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      Int64Size(this->fail_);
    if (data_size > 0) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
            static_cast<::PROTOBUF_NAMESPACE_ID::int32>(data_size));
    }
    int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(data_size);
    _fail_cached_byte_size_.store(cached_size,
                                    std::memory_order_relaxed);
    total_size += data_size;
  }

  // repeated int64 L = 5;
  {
    size_t data_size = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      Int64Size(this->l_);
    if (data_size > 0) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
            static_cast<::PROTOBUF_NAMESPACE_ID::int32>(data_size));
    }
    int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(data_size);
    _l_cached_byte_size_.store(cached_size,
                                    std::memory_order_relaxed);
    total_size += data_size;
  }

  // repeated .darts.DRegexDat.AItem V = 6;
  total_size += 1UL * this->_internal_v_size();
  for (const auto& msg : this->v_) {
    total_size +=
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(msg);
  }

  // repeated .darts.DRegexDat.AItem OutPut = 7;
  total_size += 1UL * this->_internal_output_size();
  for (const auto& msg : this->output_) {
    total_size +=
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(msg);
  }

  // map<string, int32> CodeMap = 8;
  total_size += 1 *
      ::PROTOBUF_NAMESPACE_ID::internal::FromIntSize(this->_internal_codemap_size());
  for (::PROTOBUF_NAMESPACE_ID::Map< std::string, ::PROTOBUF_NAMESPACE_ID::int32 >::const_iterator
      it = this->_internal_codemap().begin();
      it != this->_internal_codemap().end(); ++it) {
    total_size += DRegexDat_CodeMapEntry_DoNotUse::Funcs::ByteSizeLong(it->first, it->second);
  }

  // int32 MaxLen = 1;
  if (this->_internal_maxlen() != 0) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
        this->_internal_maxlen());
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    return ::PROTOBUF_NAMESPACE_ID::internal::ComputeUnknownFieldsSize(
        _internal_metadata_, total_size, &_cached_size_);
  }
  int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

const ::PROTOBUF_NAMESPACE_ID::Message::ClassData DRegexDat::_class_data_ = {
    ::PROTOBUF_NAMESPACE_ID::Message::CopyWithSizeCheck,
    DRegexDat::MergeImpl
};
const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*DRegexDat::GetClassData() const { return &_class_data_; }

void DRegexDat::MergeImpl(::PROTOBUF_NAMESPACE_ID::Message*to,
                      const ::PROTOBUF_NAMESPACE_ID::Message&from) {
  static_cast<DRegexDat *>(to)->MergeFrom(
      static_cast<const DRegexDat &>(from));
}


void DRegexDat::MergeFrom(const DRegexDat& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:darts.DRegexDat)
  GOOGLE_DCHECK_NE(&from, this);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  check_.MergeFrom(from.check_);
  base_.MergeFrom(from.base_);
  fail_.MergeFrom(from.fail_);
  l_.MergeFrom(from.l_);
  v_.MergeFrom(from.v_);
  output_.MergeFrom(from.output_);
  codemap_.MergeFrom(from.codemap_);
  if (from._internal_maxlen() != 0) {
    _internal_set_maxlen(from._internal_maxlen());
  }
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
}

void DRegexDat::CopyFrom(const DRegexDat& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:darts.DRegexDat)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool DRegexDat::IsInitialized() const {
  return true;
}

void DRegexDat::InternalSwap(DRegexDat* other) {
  using std::swap;
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  check_.InternalSwap(&other->check_);
  base_.InternalSwap(&other->base_);
  fail_.InternalSwap(&other->fail_);
  l_.InternalSwap(&other->l_);
  v_.InternalSwap(&other->v_);
  output_.InternalSwap(&other->output_);
  codemap_.InternalSwap(&other->codemap_);
  swap(maxlen_, other->maxlen_);
}

::PROTOBUF_NAMESPACE_ID::Metadata DRegexDat::GetMetadata() const {
  return ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(
      &descriptor_table_darts_2eproto_getter, &descriptor_table_darts_2eproto_once,
      file_level_metadata_darts_2eproto[2]);
}

// @@protoc_insertion_point(namespace_scope)
}  // namespace darts
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::darts::DRegexDat_AItem* Arena::CreateMaybeMessage< ::darts::DRegexDat_AItem >(Arena* arena) {
  return Arena::CreateMessageInternal< ::darts::DRegexDat_AItem >(arena);
}
template<> PROTOBUF_NOINLINE ::darts::DRegexDat_CodeMapEntry_DoNotUse* Arena::CreateMaybeMessage< ::darts::DRegexDat_CodeMapEntry_DoNotUse >(Arena* arena) {
  return Arena::CreateMessageInternal< ::darts::DRegexDat_CodeMapEntry_DoNotUse >(arena);
}
template<> PROTOBUF_NOINLINE ::darts::DRegexDat* Arena::CreateMaybeMessage< ::darts::DRegexDat >(Arena* arena) {
  return Arena::CreateMessageInternal< ::darts::DRegexDat >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>
