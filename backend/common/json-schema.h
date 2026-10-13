#pragma once

#include "json.h"

#include <cstdint>
#include <initializer_list>
#include <map>
#include <memory>
#include <string>
#include <vector>

// JSON schema, covering the subset that json_schema_to_grammar() can convert.

struct common_chat_schema {
  enum node_kind {
    cKindAny,
    cKindRef,
    cKindAnyOf,
    cKindAllOf,
    cKindConst,
    cKindEnum,
    cKindNull,
    cKindBoolean,
    cKindNumber,
    cKindInteger,
    cKindString,
    cKindArray,
    cKindTuple,
    cKindObject,
  };

  enum value_type {
    cTypeNull,
    cTypeBoolean,
    cTypeNumber,
    cTypeInteger,
    cTypeString,
    cTypeArray,
    cTypeObject,
  };

  enum string_format {
    cFormatNone,
    cFormatUuid,  // uuid, uuid1 .. uuid5
    cFormatDate,
    cFormatTime,
    cFormatDateTime,
  };

  class type_set {
    uint32_t vMask = 0;

  public:
    type_set() = default;

    type_set(std::initializer_list<value_type> pTypes) {
      for (auto type : pTypes) {
        mAdd(type);
      }
    }

    static type_set mAll() {
      return { cTypeNull, cTypeBoolean, cTypeNumber, cTypeInteger, cTypeString, cTypeArray, cTypeObject };
    }

    void mAdd(value_type type) { vMask |= 1u << type; }

    bool has(value_type type) const { return (vMask & (1u << type)) != 0; }

    bool mIsOnly(value_type type) const { return vMask == (1u << type); }

    bool empty() const { return vMask == 0; }

    type_set & operator|=(const type_set & pOther) {
      vMask |= pOther.vMask;
      return *this;
    }

    type_set & operator&=(const type_set & pOther) {
      vMask &= pOther.vMask;
      return *this;
    }

    bool operator==(const type_set & pOther) const { return vMask == pOther.vMask; }

    bool operator!=(const type_set & pOther) const { return vMask != pOther.vMask; }
  };

  virtual ~common_chat_schema()   = default;
  virtual node_kind mKind() const = 0;

  type_set mValueTypes() const;

  // Whether a value matching the schema may be a string, through any branch of it.
  bool mMayBeString() const;

  static const char * mKindName(node_kind pKind);
  static const char * mTypeName(value_type type);
};

using common_chat_schema_ptr = std::unique_ptr<common_chat_schema>;

struct common_chat_schema_any : common_chat_schema {
  node_kind mKind() const override { return cKindAny; }
};

// {"$ref": "#/..."}, only references into the same document are supported
struct common_chat_schema_ref : common_chat_schema {
  std::string                vRef;
  const common_chat_schema * vTarget = nullptr;  // owned by common_chat_schema_document::refs

  explicit common_chat_schema_ref(std::string pRef) : vRef(std::move(pRef)) {}

  node_kind mKind() const override { return cKindRef; }
};

// oneOf / anyOf, or a "type" array expanded to one alternative per type
struct common_chat_schema_any_of : common_chat_schema {
  std::vector<common_chat_schema_ptr> lChildren;

  node_kind mKind() const override { return cKindAnyOf; }
};

struct common_chat_schema_all_of : common_chat_schema {
  std::vector<common_chat_schema_ptr> lChildren;

  node_kind mKind() const override { return cKindAllOf; }
};

struct common_chat_schema_const : common_chat_schema {
  common_json value;

  explicit common_chat_schema_const(common_json value) : value(std::move(value)) {}

  node_kind mKind() const override { return cKindConst; }
};

struct common_chat_schema_enum : common_chat_schema {
  std::vector<common_json> lValues;

  node_kind mKind() const override { return cKindEnum; }
};

struct common_chat_schema_null : common_chat_schema {
  node_kind mKind() const override { return cKindNull; }
};

struct common_chat_schema_boolean : common_chat_schema {
  node_kind mKind() const override { return cKindBoolean; }
};

struct common_chat_schema_number : common_chat_schema {
  node_kind mKind() const override { return cKindNumber; }
};

// bounds are inclusive, exclusiveMinimum / exclusiveMaximum are folded in
struct common_chat_schema_integer : common_chat_schema {
  int64_t vMinimum = INT64_MIN;  // INT64_MIN for unbounded
  int64_t vMaximum = INT64_MAX;  // INT64_MAX for unbounded

  node_kind mKind() const override { return cKindInteger; }
};

struct common_chat_schema_string : common_chat_schema {
  std::string   vPattern;  // empty when absent
  string_format format     = cFormatNone;
  int           vMinLength = 0;
  int           vMaxLength = -1;  // -1 for unbounded

  node_kind mKind() const override { return cKindString; }
};

struct common_chat_schema_array : common_chat_schema {
  common_chat_schema_ptr vItems;          // a common_chat_schema_any when "items" is absent
  int                    vMinItems = 0;
  int                    vMaxItems = -1;  // -1 for unbounded

  node_kind mKind() const override { return cKindArray; }
};

struct common_chat_schema_tuple : common_chat_schema {
  std::vector<common_chat_schema_ptr> lItems;

  node_kind mKind() const override { return cKindTuple; }
};

struct common_chat_schema_property {
  std::string            name;
  common_chat_schema_ptr vSchema;
  bool                   vRequired = false;
};

struct common_chat_schema_object : common_chat_schema {
  std::vector<common_chat_schema_property> lProperties;            // in schema order
  common_chat_schema_ptr                   vAdditionalProperties;  // null when not allowed

  node_kind mKind() const override { return cKindObject; }
};

struct common_chat_schema_document {
  common_chat_schema_ptr                        vRoot;
  std::map<std::string, common_chat_schema_ptr> dRefs;
};

// A document shared by the PEG parsers built from its nodes, which it keeps alive
using common_chat_schema_document_ptr = std::shared_ptr<const common_chat_schema_document>;

// Throws std::runtime_error when the schema falls outside the supported subset.
common_chat_schema_document fCommonChatSchemaFromJson(const common_json & pSchema);
