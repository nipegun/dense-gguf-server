#include "json-schema.h"

#include "common.h"

#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

class common_chat_schema_builder {
  const common_json &           vRoot;
  common_chat_schema_document & vDoc;

  // the targets built here, moved into doc_ once the whole schema is built
  std::map<std::string, common_chat_schema_ptr> dRefs;

  // ref nodes get their target once every $ref is built, a cycle would otherwise need it too early
  std::vector<common_chat_schema_ref *> lPending;

  [[noreturn]] static void mFail(const std::string & pPath, const std::string & pMsg) {
    throw std::runtime_error("JSON schema error at " + pPath + ": " + pMsg);
  }

  static int mGetCount(const common_json & pSchema, const std::string & pKey, const std::string & pPath, int pDef) {
    if (!pSchema.contains(pKey)) {
      return pDef;
    }
    const common_json & value = pSchema.at(pKey);
    if (!value.mIsNumberInteger() || value.get<int>() < 0) {
      mFail(pPath, pKey + " must be a non-negative integer");
    }
    return value.get<int>();
  }

  // a fractional bound is rounded inwards, towards the integers it still admits
  static int64_t mGetBound(const common_json & pSchema,
                           const std::string & pKey,
                           const std::string & pPath,
                           bool                pRoundUp) {
    const common_json & value = pSchema.at(pKey);
    if (value.mIsNumberInteger()) {
      return value.get<int64_t>();
    }
    if (!value.is_number()) {
      mFail(pPath, pKey + " must be a number");
    }
    double vD = value.get<double>();
    return (int64_t) (pRoundUp ? std::ceil(vD) : std::floor(vD));
  }

  static common_chat_schema::string_format mGetFormat(const common_json & pSchema, const std::string & pPath) {
    if (!pSchema.contains("format")) {
      return common_chat_schema::cFormatNone;
    }
    const common_json & value = pSchema.at("format");
    if (!value.is_string()) {
      mFail(pPath, "format must be a string");
    }
    std::string format = value.get<std::string>();
    if (format == "date") {
      return common_chat_schema::cFormatDate;
    }
    if (format == "time") {
      return common_chat_schema::cFormatTime;
    }
    if (format == "date-time") {
      return common_chat_schema::cFormatDateTime;
    }
    if (format == "uuid" ||
        (format.size() == 5 && format.compare(0, 4, "uuid") == 0 && format[4] >= '1' && format[4] <= '5')) {
      return common_chat_schema::cFormatUuid;
    }
    return common_chat_schema::cFormatNone;
  }

  const common_json & mResolveRef(const std::string & pRef, const std::string & pPath) {
    const common_json * vTarget = &vRoot;
    auto                lTokens = fStringSplit(pRef.substr(1), "/");
    for (size_t vI = 1; vI < lTokens.size(); vI++) {
      const std::string & vSel = lTokens[vI];
      if (vTarget->is_object() && vTarget->contains(vSel)) {
        vTarget = &vTarget->at(vSel);
      } else if (vTarget->is_array()) {
        size_t vIdx;
        try {
          vIdx = std::stoull(vSel);
        } catch (const std::logic_error &) {
          vIdx = vTarget->size();
        }
        if (vIdx >= vTarget->size()) {
          mFail(pPath, "cannot resolve $ref " + pRef + ", " + vSel + " is out of range");
        }
        vTarget = &vTarget->at(vIdx);
      } else {
        mFail(pPath, "cannot resolve $ref " + pRef + ", " + vSel + " not found");
      }
    }
    return *vTarget;
  }

  common_chat_schema_ptr mBuildRef(const common_json & value, const std::string & pPath) {
    if (!value.is_string()) {
      mFail(pPath, "$ref must be a string");
    }
    std::string vRef = value.get<std::string>();
    if (vRef.compare(0, 2, "#/") != 0) {
      mFail(pPath, "unsupported $ref " + vRef + ", only references into the same document are supported");
    }
    if (dRefs.find(vRef) == dRefs.end()) {
      // reserve the key first, so that a cycle back to this $ref stops here
      dRefs[vRef] = nullptr;
      dRefs[vRef] = mBuildNode(mResolveRef(vRef, pPath), vRef);
    }
    auto node = std::make_unique<common_chat_schema_ref>(vRef);
    lPending.push_back(node.get());
    return node;
  }

  template <typename T>
  common_chat_schema_ptr mBuildAlternatives(const common_json & pAlts, const std::string & pPath) {
    if (!pAlts.is_array()) {
      mFail(pPath, "must be an array of schemas");
    }
    if (pAlts.empty()) {
      mFail(pPath, "must not be empty");
    }
    auto   node = std::make_unique<T>();
    size_t vI   = 0;
    for (const auto & vAlt : pAlts) {
      node->lChildren.push_back(mBuildNode(vAlt, pPath + "/" + std::to_string(vI++)));
    }
    return node;
  }

  common_chat_schema_ptr mBuildObject(const common_json & pSchema, const std::string & pPath) {
    auto node = std::make_unique<common_chat_schema_object>();

    std::unordered_set<std::string> sRequired;
    if (pSchema.contains("required") && pSchema.at("required").is_array()) {
      for (const auto & name : pSchema.at("required")) {
        if (name.is_string()) {
          sRequired.insert(name.get<std::string>());
        }
      }
    }

    if (pSchema.contains("properties")) {
      const common_json & vProperties = pSchema.at("properties");
      if (!vProperties.is_object()) {
        mFail(pPath, "properties must be an object");
      }
      for (const auto & [name, prop] : vProperties.mItems()) {
        node->lProperties.push_back(
          { name, mBuildNode(prop, pPath + "/properties/" + name), sRequired.count(name) > 0 });
      }
    }

    if (pSchema.contains("additionalProperties")) {
      const common_json & vAdditional = pSchema.at("additionalProperties");
      if (vAdditional.is_boolean()) {
        if (vAdditional.get<bool>()) {
          node->vAdditionalProperties = std::make_unique<common_chat_schema_any>();
        }
      } else if (vAdditional.is_object()) {
        node->vAdditionalProperties = mBuildNode(vAdditional, pPath + "/additionalProperties");
      } else {
        mFail(pPath, "additionalProperties must be a boolean or a schema");
      }
    } else if (!pSchema.contains("properties")) {
      // {"type": "object"} on its own accepts any object
      node->vAdditionalProperties = std::make_unique<common_chat_schema_any>();
    }

    return node;
  }

  common_chat_schema_ptr mBuildArray(const common_json & pSchema, const std::string & pPath) {
    auto node = std::make_unique<common_chat_schema_array>();
    if (pSchema.contains("items") || pSchema.contains("prefixItems")) {
      // "items" wins when both are present; as in the converter, a schema instead of an array is the item schema
      const std::string   cKey   = pSchema.contains("items") ? "items" : "prefixItems";
      const common_json & vItems = pSchema.at(cKey);
      if (vItems.is_array()) {
        auto   vTuple = std::make_unique<common_chat_schema_tuple>();
        size_t vI     = 0;
        for (const auto & vItem : vItems) {
          vTuple->lItems.push_back(mBuildNode(vItem, pPath + "/" + cKey + "/" + std::to_string(vI++)));
        }
        return vTuple;
      }
      node->vItems = mBuildNode(vItems, pPath + "/" + cKey);
    } else {
      node->vItems = std::make_unique<common_chat_schema_any>();
    }
    node->vMinItems = mGetCount(pSchema, "minItems", pPath, 0);
    node->vMaxItems = mGetCount(pSchema, "maxItems", pPath, -1);
    return node;
  }

  common_chat_schema_ptr mBuildString(const common_json & pSchema, const std::string & pPath) {
    auto node = std::make_unique<common_chat_schema_string>();
    if (pSchema.contains("pattern")) {
      const common_json & vPattern = pSchema.at("pattern");
      if (!vPattern.is_string()) {
        mFail(pPath, "pattern must be a string");
      }
      node->vPattern = vPattern.get<std::string>();
    }
    node->format     = mGetFormat(pSchema, pPath);
    node->vMinLength = mGetCount(pSchema, "minLength", pPath, 0);
    node->vMaxLength = mGetCount(pSchema, "maxLength", pPath, -1);
    return node;
  }

  common_chat_schema_ptr mBuildInteger(const common_json & pSchema, const std::string & pPath) {
    auto node = std::make_unique<common_chat_schema_integer>();
    if (pSchema.contains("minimum")) {
      node->vMinimum = mGetBound(pSchema, "minimum", pPath, /* round_up */ true);
    } else if (pSchema.contains("exclusiveMinimum")) {
      node->vMinimum = mGetBound(pSchema, "exclusiveMinimum", pPath, /* round_up */ false) + 1;
    }
    if (pSchema.contains("maximum")) {
      node->vMaximum = mGetBound(pSchema, "maximum", pPath, /* round_up */ false);
    } else if (pSchema.contains("exclusiveMaximum")) {
      node->vMaximum = mGetBound(pSchema, "exclusiveMaximum", pPath, /* round_up */ true) - 1;
    }
    return node;
  }

  common_chat_schema_ptr mBuildNode(const common_json & pSchema, const std::string & pPath) {
    if (!pSchema.is_object()) {
      mFail(pPath, "schema must be an object");
    }
    if (pSchema.contains("$ref")) {
      return mBuildRef(pSchema.at("$ref"), pPath);
    }
    if (pSchema.contains("oneOf") || pSchema.contains("anyOf")) {
      const std::string cKey = pSchema.contains("oneOf") ? "oneOf" : "anyOf";
      return mBuildAlternatives<common_chat_schema_any_of>(pSchema.at(cKey), pPath + "/" + cKey);
    }

    common_json type;
    if (pSchema.contains("type")) {
      type = pSchema.at("type");
    }
    if (type.is_array()) {
      // {"type": ["a", "b"], ...} is {"anyOf": [{"type": "a", ...}, {"type": "b", ...}]}
      if (type.empty()) {
        mFail(pPath, "type must not be empty");
      }
      auto   node = std::make_unique<common_chat_schema_any_of>();
      size_t vI   = 0;
      for (const auto & vT : type) {
        common_json vAlt = pSchema;
        vAlt["type"]     = vT;
        node->lChildren.push_back(mBuildNode(vAlt, pPath + "/type/" + std::to_string(vI++)));
      }
      return node;
    }
    if (pSchema.contains("const")) {
      return std::make_unique<common_chat_schema_const>(pSchema.at("const"));
    }
    if (pSchema.contains("enum")) {
      const common_json & vValues = pSchema.at("enum");
      if (!vValues.is_array() || vValues.empty()) {
        mFail(pPath, "enum must be a non-empty array");
      }
      auto node = std::make_unique<common_chat_schema_enum>();
      for (const auto & value : vValues) {
        node->lValues.push_back(value);
      }
      return node;
    }
    if (!type.is_null() && !type.is_string()) {
      mFail(pPath, "type must be a string or an array of strings");
    }

    const std::string cTypeName      = type.is_string() ? type.get<std::string>() : "";
    const bool        cHasProperties = pSchema.contains("properties") || (pSchema.contains("additionalProperties") &&
                                                                          pSchema.at("additionalProperties") != true);

    if (cTypeName.empty()) {
      // without a type the structural keywords decide, in the same order as the converter
      if (cHasProperties) {
        return mBuildObject(pSchema, pPath);
      }
      if (pSchema.contains("allOf")) {
        return mBuildAlternatives<common_chat_schema_all_of>(pSchema.at("allOf"), pPath + "/allOf");
      }
      if (pSchema.contains("items") || pSchema.contains("prefixItems")) {
        return mBuildArray(pSchema, pPath);
      }
      if (pSchema.contains("pattern") || pSchema.contains("minLength") || pSchema.contains("maxLength") ||
          mGetFormat(pSchema, pPath) != common_chat_schema::cFormatNone) {
        return mBuildString(pSchema, pPath);
      }
      return std::make_unique<common_chat_schema_any>();
    }
    if (cTypeName == "object") {
      if (!cHasProperties && pSchema.contains("allOf")) {
        return mBuildAlternatives<common_chat_schema_all_of>(pSchema.at("allOf"), pPath + "/allOf");
      }
      return mBuildObject(pSchema, pPath);
    }
    if (cTypeName == "string") {
      if (pSchema.contains("allOf")) {
        return mBuildAlternatives<common_chat_schema_all_of>(pSchema.at("allOf"), pPath + "/allOf");
      }
      return mBuildString(pSchema, pPath);
    }
    if (cTypeName == "array") {
      return mBuildArray(pSchema, pPath);
    }
    if (cTypeName == "integer") {
      return mBuildInteger(pSchema, pPath);
    }
    if (cTypeName == "number") {
      return std::make_unique<common_chat_schema_number>();
    }
    if (cTypeName == "boolean") {
      return std::make_unique<common_chat_schema_boolean>();
    }
    if (cTypeName == "null") {
      return std::make_unique<common_chat_schema_null>();
    }
    mFail(pPath, "unrecognized type " + cTypeName);
  }

public:
  common_chat_schema_builder(const common_json & pRoot, common_chat_schema_document & pDoc) :
    vRoot(pRoot),
    vDoc(pDoc) {}

  common_chat_schema_ptr mBuild() {
    auto node = mBuildNode(vRoot, "#");
    for (auto & vEntry : dRefs) {
      vDoc.dRefs[vEntry.first] = std::move(vEntry.second);
    }
    for (auto * vRef : lPending) {
      vRef->vTarget = vDoc.dRefs.at(vRef->vRef).get();
    }
    return node;
  }
};

common_chat_schema_document fCommonChatSchemaFromJson(const common_json & pSchema) {
  common_chat_schema_document vDoc;
  vDoc.vRoot = common_chat_schema_builder(pSchema, vDoc).mBuild();
  return vDoc;
}

static common_chat_schema::value_type fJsonType(const common_json & value) {
  if (value.is_null()) {
    return common_chat_schema::cTypeNull;
  }
  if (value.is_boolean()) {
    return common_chat_schema::cTypeBoolean;
  }
  if (value.mIsNumberInteger()) {
    return common_chat_schema::cTypeInteger;
  }
  if (value.is_number()) {
    return common_chat_schema::cTypeNumber;
  }
  if (value.is_string()) {
    return common_chat_schema::cTypeString;
  }
  if (value.is_array()) {
    return common_chat_schema::cTypeArray;
  }
  return common_chat_schema::cTypeObject;
}

static common_chat_schema::type_set fValueTypesImpl(const common_chat_schema &                       pS,
                                                    std::unordered_set<const common_chat_schema *> & pVisited) {
  switch (pS.mKind()) {
    case common_chat_schema::cKindAny:
      return common_chat_schema::type_set::mAll();
    case common_chat_schema::cKindNull:
      return { common_chat_schema::cTypeNull };
    case common_chat_schema::cKindBoolean:
      return { common_chat_schema::cTypeBoolean };
    case common_chat_schema::cKindNumber:
      return { common_chat_schema::cTypeNumber, common_chat_schema::cTypeInteger };
    case common_chat_schema::cKindInteger:
      return { common_chat_schema::cTypeInteger };
    case common_chat_schema::cKindString:
      return { common_chat_schema::cTypeString };
    case common_chat_schema::cKindArray:
    case common_chat_schema::cKindTuple:
      return { common_chat_schema::cTypeArray };
    case common_chat_schema::cKindObject:
      return { common_chat_schema::cTypeObject };
    case common_chat_schema::cKindConst:
      return { fJsonType(static_cast<const common_chat_schema_const &>(pS).value) };
    case common_chat_schema::cKindEnum:
      {
        common_chat_schema::type_set vTypes;
        for (const auto & value : static_cast<const common_chat_schema_enum &>(pS).lValues) {
          vTypes.mAdd(fJsonType(value));
        }
        return vTypes;
      }
    case common_chat_schema::cKindRef:
      {
        const auto * vTarget = static_cast<const common_chat_schema_ref &>(pS).vTarget;
        if (!vTarget || !pVisited.insert(vTarget).second) {
          // a cycle contributes no type, to be safe
          return {};
        }
        auto vTypes = fValueTypesImpl(*vTarget, pVisited);
        pVisited.erase(vTarget);
        return vTypes;
      }
    case common_chat_schema::cKindAnyOf:
      {
        common_chat_schema::type_set vTypes;
        for (const auto & vChild : static_cast<const common_chat_schema_any_of &>(pS).lChildren) {
          vTypes |= fValueTypesImpl(*vChild, pVisited);
        }
        return vTypes;
      }
    case common_chat_schema::cKindAllOf:
      {
        auto vTypes = common_chat_schema::type_set::mAll();
        for (const auto & vChild : static_cast<const common_chat_schema_all_of &>(pS).lChildren) {
          vTypes &= fValueTypesImpl(*vChild, pVisited);
        }
        return vTypes;
      }
  }
  return {};
}

common_chat_schema::type_set common_chat_schema::mValueTypes() const {
  std::unordered_set<const common_chat_schema *> sVisited;
  return fValueTypesImpl(*this, sVisited);
}

static bool fMayBeStringImpl(const common_chat_schema & pS, std::unordered_set<const common_chat_schema *> & pVisited) {
  switch (pS.mKind()) {
    case common_chat_schema::cKindString:
      return true;
    case common_chat_schema::cKindConst:
      return static_cast<const common_chat_schema_const &>(pS).value.is_string();
    case common_chat_schema::cKindEnum:
      for (const auto & vV : static_cast<const common_chat_schema_enum &>(pS).lValues) {
        if (vV.is_string()) {
          return true;
        }
      }
      return false;
    case common_chat_schema::cKindRef:
      {
        // a cycle is taken as not a string, to be safe
        const auto * vTarget = static_cast<const common_chat_schema_ref &>(pS).vTarget;
        if (!vTarget || !pVisited.insert(vTarget).second) {
          return false;
        }
        bool vResult = fMayBeStringImpl(*vTarget, pVisited);
        pVisited.erase(vTarget);
        return vResult;
      }
    case common_chat_schema::cKindAnyOf:
      for (const auto & vChild : static_cast<const common_chat_schema_any_of &>(pS).lChildren) {
        if (fMayBeStringImpl(*vChild, pVisited)) {
          return true;
        }
      }
      return false;
    case common_chat_schema::cKindAllOf:
      {
        // every child must allow a string, an any child constrains nothing
        bool vAnyString = false;
        for (const auto & vChild : static_cast<const common_chat_schema_all_of &>(pS).lChildren) {
          if (vChild->mKind() == common_chat_schema::cKindAny) {
            continue;
          }
          if (!fMayBeStringImpl(*vChild, pVisited)) {
            return false;
          }
          vAnyString = true;
        }
        return vAnyString;
      }
    default:
      return false;
  }
}

bool common_chat_schema::mMayBeString() const {
  std::unordered_set<const common_chat_schema *> sVisited;
  return fMayBeStringImpl(*this, sVisited);
}

const char * common_chat_schema::mKindName(node_kind pKind) {
  switch (pKind) {
    case cKindAny:
      return "any";
    case cKindRef:
      return "ref";
    case cKindAnyOf:
      return "anyOf";
    case cKindAllOf:
      return "allOf";
    case cKindConst:
      return "const";
    case cKindEnum:
      return "enum";
    case cKindNull:
      return "null";
    case cKindBoolean:
      return "boolean";
    case cKindNumber:
      return "number";
    case cKindInteger:
      return "integer";
    case cKindString:
      return "string";
    case cKindArray:
      return "array";
    case cKindTuple:
      return "tuple";
    case cKindObject:
      return "object";
  }
  return "?";
}

const char * common_chat_schema::mTypeName(value_type type) {
  switch (type) {
    case cTypeNull:
      return "null";
    case cTypeBoolean:
      return "boolean";
    case cTypeNumber:
      return "number";
    case cTypeInteger:
      return "integer";
    case cTypeString:
      return "string";
    case cTypeArray:
      return "array";
    case cTypeObject:
      return "object";
  }
  return "?";
}
