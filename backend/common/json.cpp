#include "json.h"

#include "ggml.h"

#define JSON_ASSERT GGML_ASSERT
#include <iterator>
#include <new>
#include <nlohmann/json.hpp>
#include <set>
#include <unordered_map>
#include <vector>

using nlohmann::ordered_json;

// a common_json is the backing value, so any value of a tree can be used as a common_json
static_assert(sizeof(ordered_json) <= sizeof(common_json), "common_json storage is too small");
static_assert(alignof(ordered_json) <= alignof(common_json), "common_json alignment is too weak");

// runs fn and gives every error of the backing library as a common_json_error
template <typename F> static decltype(auto) fGuard(F && pFn) {
  try {
    return pFn();
  } catch (const ordered_json::exception & vE) {
    throw common_json_error(vE.what());
  }
}

static ordered_json & fAsJson(common_json * pSelf) {
  return *reinterpret_cast<ordered_json *>(pSelf);
}

static const ordered_json & fAsJson(const common_json * pSelf) {
  return *reinterpret_cast<const ordered_json *>(pSelf);
}

void common_json::mWipeString() {
  auto & vJson = fAsJson(this);
  if (!vJson.is_string()) {
    return;
  }
  auto &          vText = vJson.get_ref<std::string &>();
  volatile char * vData = vText.data();
  for (size_t vIndex = 0; vIndex < vText.size(); ++vIndex) {
    vData[vIndex] = '\0';
  }
  vText.clear();
}

static common_json & fAsCommon(ordered_json & pJson) {
  return *reinterpret_cast<common_json *>(&pJson);
}

static const common_json & fAsCommon(const ordered_json & pJson) {
  return *reinterpret_cast<const common_json *>(&pJson);
}

static ordered_json to_json(const common_json_value & pVal) {
  switch (pVal.type) {
    case common_json_value::cValNull:
      return nullptr;
    case common_json_value::cValBool:
      return pVal.vValBool;
    case common_json_value::cValInt:
      return pVal.vValInt;
    case common_json_value::cValUint:
      return pVal.vValUint;
    case common_json_value::cValDouble:
      return pVal.vValDouble;
    case common_json_value::cValString:
      return pVal.vValString;
    case common_json_value::cValJson:
      // one owner means no one else can see this tree, so it is safe to move it out
      // note: this makes a value single use, same as the json_ref of the backing library
      if (pVal.vValJson.use_count() == 1) {
        return std::move(fAsJson(pVal.vValJson.get()));
      }
      return fAsJson(pVal.vValJson.get());
  }

  return nullptr;
}

common_json_value::common_json_value(const char * pVal) {
  if (pVal) {
    type       = cValString;
    vValString = pVal;
  } else {
    type = cValNull;
  }
}

common_json_value::common_json_value(const common_json & pVal) :
  type(cValJson),
  vValJson(std::make_shared<common_json>(pVal)) {}

common_json_value::common_json_value(common_json && pVal) :
  type(cValJson),
  vValJson(std::make_shared<common_json>(std::move(pVal))) {}

// the ctors and get<T>() below are explicit specializations, giving strong symbols
// an explicit instantiation is a weak symbol, dropped by some LTO builds (clang-cl)
template <typename T> static std::shared_ptr<common_json> fSetJson(const std::set<T> & pVals) {
  common_json vOut = common_json::mArray();

  for (const auto & vVal : pVals) {
    vOut.push_back(vVal);
  }

  return std::make_shared<common_json>(std::move(vOut));
}

// a set value is usable only for the types below
#define COMMON_JSON_SET(...)                                                 \
  template <>                                                                \
  common_json_value::common_json_value(const std::set<__VA_ARGS__> & vals) : \
    type(cValJson),                                                          \
    vValJson(fSetJson(vals)) {}

COMMON_JSON_SET(int)
COMMON_JSON_SET(std::string)

#undef COMMON_JSON_SET

template <typename T> static std::shared_ptr<common_json> fMapJson(const T & pVals) {
  common_json vOut = common_json::mObject();

  for (const auto & vVal : pVals) {
    vOut.set({ vVal.first, vVal.second });
  }

  return std::make_shared<common_json>(std::move(vOut));
}

// a map value is usable only for the types below
#define COMMON_JSON_MAP(...)                                                              \
  template <>                                                                             \
  common_json_value::common_json_value(const std::map<std::string, __VA_ARGS__> & vals) : \
    type(cValJson),                                                                       \
    vValJson(fMapJson(vals)) {}

COMMON_JSON_MAP(bool)
COMMON_JSON_MAP(std::string)

#undef COMMON_JSON_MAP

// an unordered map value is usable only for the types below
#define COMMON_JSON_UMAP(...)                                                                       \
  template <>                                                                                       \
  common_json_value::common_json_value(const std::unordered_map<std::string, __VA_ARGS__> & vals) : \
    type(cValJson),                                                                                 \
    vValJson(fMapJson(vals)) {}

COMMON_JSON_UMAP(size_t)

#undef COMMON_JSON_UMAP

template <typename T> static std::shared_ptr<common_json> fVecJson(const std::vector<T> & pVals) {
  common_json vOut = common_json::mArray();

  for (const auto & vVal : pVals) {
    vOut.push_back(vVal);
  }

  return std::make_shared<common_json>(std::move(vOut));
}

// a vector value is usable only for the types below
// note: std::vector<bool> is not here, its proxy reference does not convert
#define COMMON_JSON_VEC(...)                                                    \
  template <>                                                                   \
  common_json_value::common_json_value(const std::vector<__VA_ARGS__> & vals) : \
    type(cValJson),                                                             \
    vValJson(fVecJson(vals)) {}

COMMON_JSON_VEC(int)
COMMON_JSON_VEC(unsigned char)
COMMON_JSON_VEC(unsigned int)
COMMON_JSON_VEC(long)
COMMON_JSON_VEC(unsigned long)
COMMON_JSON_VEC(long long)
COMMON_JSON_VEC(unsigned long long)
COMMON_JSON_VEC(float)
COMMON_JSON_VEC(double)
COMMON_JSON_VEC(std::string)
COMMON_JSON_VEC(std::vector<float>)
COMMON_JSON_VEC(common_json)

#undef COMMON_JSON_VEC

common_json_value::common_json_value(std::initializer_list<common_json_item> pItems) :
  type(cValJson),
  vValJson(std::make_shared<common_json>(pItems)) {}

// null, same as the backing library
// operator[] turns it into an object, push_back() into an array
common_json::common_json() {
  new (aStorage) ordered_json();
}

common_json::common_json(const common_json & pOther) {
  new (aStorage) ordered_json(fAsJson(&pOther));
}

common_json::common_json(common_json && pOther) noexcept {
  new (aStorage) ordered_json(std::move(fAsJson(&pOther)));
}

common_json::common_json(std::initializer_list<common_json_item> pItems) {
  new (aStorage) ordered_json(ordered_json::object());

  for (const auto & vItem : pItems) {
    set(vItem);
  }
}

common_json::common_json(const common_json_value & pVal) {
  new (aStorage) ordered_json(to_json(pVal));
}

common_json::common_json(std::nullptr_t) {
  new (aStorage) ordered_json(nullptr);
}

common_json & common_json::operator=(common_json pOther) noexcept {
  fAsJson(this).swap(fAsJson(&pOther));

  return *this;
}

common_json::~common_json() {
  fAsJson(this).~basic_json();
}

common_json common_json::mParse(const std::string & pText) {
  try {
    // the assignment moves the parsed tree in, it does not copy
    common_json vOut;
    fAsJson(&vOut) = ordered_json::parse(pText);
    return vOut;
  } catch (const std::exception & vE) {
    throw common_json_error(vE.what());
  }
}

common_json common_json::mParseNoThrow(const std::string & pText) {
  common_json vOut;
  fAsJson(&vOut) = ordered_json::parse(pText, nullptr, false);
  return vOut;
}

bool common_json::mIsDiscarded() const {
  return fAsJson(this).is_discarded();
}

common_json common_json::mArray() {
  common_json vOut;
  fAsJson(&vOut) = ordered_json::array();
  return vOut;
}

common_json common_json::mArray(std::initializer_list<common_json_value> pVals) {
  common_json    vOut;
  ordered_json & vArr = fAsJson(&vOut);
  vArr                = ordered_json::array();

  for (const auto & vVal : pVals) {
    vArr.push_back(to_json(vVal));
  }

  return vOut;
}

common_json common_json::mObject() {
  common_json vOut;
  fAsJson(&vOut) = ordered_json::object();
  return vOut;
}

common_json common_json::mObject(std::initializer_list<common_json_item> pItems) {
  return common_json(pItems);
}

common_json common_json::mMake(const common_json_value & pVal) {
  return common_json(pVal);
}

bool common_json::is_null() const {
  return fAsJson(this).is_null();
}

bool common_json::is_object() const {
  return fAsJson(this).is_object();
}

bool common_json::is_array() const {
  return fAsJson(this).is_array();
}

bool common_json::is_string() const {
  return fAsJson(this).is_string();
}

bool common_json::is_boolean() const {
  return fAsJson(this).is_boolean();
}

bool common_json::is_number() const {
  return fAsJson(this).is_number();
}

bool common_json::mIsNumberInteger() const {
  return fAsJson(this).is_number_integer();
}

bool common_json::mIsNumberFloat() const {
  return fAsJson(this).is_number_float();
}

bool common_json::empty() const {
  return fAsJson(this).empty();
}

size_t common_json::size() const {
  return fAsJson(this).size();
}

bool common_json::contains(const std::string & pKey) const {
  return fAsJson(this).contains(pKey);
}

bool common_json::operator==(const common_json_value & pVal) const {
  // compare a tree in place, to_json() would copy it
  if (pVal.type == common_json_value::cValJson) {
    return fAsJson(this) == fAsJson(pVal.vValJson.get());
  }
  return fAsJson(this) == to_json(pVal);
}

bool common_json::operator!=(const common_json_value & pVal) const {
  return !(*this == pVal);
}

common_json & common_json::at(const std::string & pKey) {
  return fGuard([&]() -> common_json & { return fAsCommon(fAsJson(this).at(pKey)); });
}

const common_json & common_json::at(const std::string & pKey) const {
  return fGuard([&]() -> const common_json & { return fAsCommon(fAsJson(this).at(pKey)); });
}

common_json & common_json::at(size_t pIdx) {
  return fGuard([&]() -> common_json & { return fAsCommon(fAsJson(this).at(pIdx)); });
}

const common_json & common_json::at(size_t pIdx) const {
  return fGuard([&]() -> const common_json & { return fAsCommon(fAsJson(this).at(pIdx)); });
}

common_json & common_json::operator[](const std::string & pKey) {
  return fGuard([&]() -> common_json & { return fAsCommon(fAsJson(this)[pKey]); });
}

const common_json & common_json::operator[](const std::string & pKey) const {
  return fGuard([&]() -> const common_json & { return fAsCommon(fAsJson(this).at(pKey)); });
}

common_json & common_json::operator[](size_t pIdx) {
  return fGuard([&]() -> common_json & { return fAsCommon(fAsJson(this)[pIdx]); });
}

const common_json & common_json::operator[](size_t pIdx) const {
  return fGuard([&]() -> const common_json & { return fAsCommon(fAsJson(this).at(pIdx)); });
}

common_json & common_json::front() {
  return fAsCommon(fAsJson(this).front());
}

const common_json & common_json::front() const {
  return fAsCommon(fAsJson(this).front());
}

common_json & common_json::back() {
  return fAsCommon(fAsJson(this).back());
}

const common_json & common_json::back() const {
  return fAsCommon(fAsJson(this).back());
}

void common_json::clear() {
  fAsJson(this).clear();
}

void common_json::erase(const std::string & pKey) {
  fGuard([&] { fAsJson(this).erase(pKey); });
}

void common_json::erase(size_t pIdx) {
  fGuard([&] { fAsJson(this).erase(pIdx); });
}

void common_json::assign(const common_json_value & pVal) {
  fAsJson(this) = to_json(pVal);
}

void common_json::set(const common_json_item & pItem) {
  fGuard([&] { fAsJson(this)[pItem.vKey] = to_json(pItem.vVal); });
}

void common_json::push_back(const common_json_value & pVal) {
  fGuard([&] { fAsJson(this).push_back(to_json(pVal)); });
}

void common_json::push_back(std::initializer_list<common_json_item> pItems) {
  common_json vVal(pItems);

  fGuard([&] { fAsJson(this).push_back(std::move(fAsJson(&vVal))); });
}

size_t common_json::count(const std::string & pKey) const {
  return fAsJson(this).count(pKey);
}

void common_json::insert(const common_json & pVals) {
  fGuard([&] {
    ordered_json & vSelf = fAsJson(this);

    vSelf.insert(vSelf.end(), fAsJson(&pVals).begin(), fAsJson(&pVals).end());
  });
}

std::string common_json::mDump(int pIndent) const {
  return fGuard([&] { return fAsJson(this).dump(pIndent); });
}

std::string common_json::mDumpSafe(int pIndent) const {
  return fAsJson(this).dump(pIndent, ' ', false, ordered_json::error_handler_t::replace);
}

// an array is indexed directly, an object needs a walk from the start
common_json & common_json::iterator::operator*() const {
  return fGuard([&]() -> common_json & {
    ordered_json & vJ = fAsJson(node);

    if (vJ.is_object()) {
      return fAsCommon(std::next(vJ.begin(), vIdx).value());
    }
    if (vJ.is_array()) {
      return fAsCommon(vJ[vIdx]);
    }

    // a plain value gives itself once, same as the backing library
    return *node;
  });
}

std::string common_json::iterator::mKey() const {
  return fGuard([&] { return std::next(fAsJson(node).begin(), vIdx).key(); });
}

common_json::iterator common_json::begin() const {
  return iterator(const_cast<common_json *>(this), 0);
}

common_json::iterator common_json::end() const {
  return iterator(const_cast<common_json *>(this), size());
}

// the keys follow the backing library: the index for an array, "" for a plain value
common_json::items_view::entry common_json::items_view::iterator::operator*() const {
  return fGuard([&]() -> entry {
    ordered_json & vJ = fAsJson(node);

    if (vJ.is_object()) {
      auto vIt = std::next(vJ.begin(), vIdx);

      return { vIt.key(), fAsCommon(vIt.value()) };
    }
    if (vJ.is_array()) {
      return { std::to_string(vIdx), fAsCommon(vJ[vIdx]) };
    }

    return { std::string(), *node };
  });
}

common_json::items_view common_json::mItems() const {
  return items_view(const_cast<common_json *>(this), size());
}

// the backing library cannot build a common_json, so this one is just a copy
template <> common_json common_json::get<common_json>() const {
  return *this;
}

// get<T>() is usable only for the types below

#define COMMON_JSON_GET(...)                                         \
  template <> __VA_ARGS__ common_json::get<__VA_ARGS__>() const {    \
    return fGuard([&] { return fAsJson(this).get<__VA_ARGS__>(); }); \
  }

COMMON_JSON_GET(bool)
COMMON_JSON_GET(int)
COMMON_JSON_GET(unsigned int)
COMMON_JSON_GET(long)
COMMON_JSON_GET(unsigned long)
COMMON_JSON_GET(long long)
COMMON_JSON_GET(unsigned long long)
COMMON_JSON_GET(float)
COMMON_JSON_GET(double)
COMMON_JSON_GET(std::string)
COMMON_JSON_GET(std::vector<float>)
COMMON_JSON_GET(std::vector<std::string>)
COMMON_JSON_GET(std::set<std::string>)
COMMON_JSON_GET(std::vector<int>)
COMMON_JSON_GET(std::vector<size_t>)
COMMON_JSON_GET(std::unordered_map<std::string, size_t>)

#undef COMMON_JSON_GET

// must stay below the get<std::string> specialization
common_json::operator std::string() const {
  return get<std::string>();
}

std::string common_json::value(const std::string & pKey, const char * pDef) const {
  return contains(pKey) ? at(pKey).get<std::string>() : std::string(pDef);
}
