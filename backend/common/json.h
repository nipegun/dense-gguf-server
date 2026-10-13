#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

// common_json, a thin wrapper around vendor json library
// the underlay library is pimpl, we are using nlohmann::json for now
//
// many features of the library are deliberately left out, to keep this interface small and generic and to keep compile time down
//
// some main differences compared to nlohmann::json :
// - object keys keep the order in which they are added
// - errors are always throw as common_json_error
// - obj.push_back({key, val}) is intentionally unsupported to avoid confusion with push_back on a vector; write it as obj[key] = val for clarity
// - a braced pair in value position does not build, e.g. {"key", {"a", "b"}}; write array({"a", "b"}) where nlohmann made an array
//
// in doubt, search the code base for an existing usage example; do not add anything to this header unless absolutely necessary

class common_json;

// common_json_value holds a list of these, and each of them holds a value, so one must come first
struct common_json_item;

struct common_json_error : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// one value, tagged so that this header stays free of the backing library
// note: a value that holds a tree is single use, the second use gives null
struct common_json_value {
  enum value_type {
    cValNull,
    cValBool,
    cValInt,
    cValUint,
    cValDouble,
    cValString,
    cValJson,
  };

  value_type type = cValNull;

  union {
    bool     vValBool;
    int64_t  vValInt;
    uint64_t vValUint = 0;
    double   vValDouble;
  };

  std::string                  vValString;
  std::shared_ptr<common_json> vValJson;

  common_json_value(std::nullptr_t = nullptr) : type(cValNull) {}

  common_json_value(bool pVal) : type(cValBool), vValBool(pVal) {}

  common_json_value(std::string pVal) : type(cValString), vValString(std::move(pVal)) {}

  // without this a string_view lands on the common_json ctor below and recurses
  common_json_value(std::string_view pVal) : type(cValString), vValString(pVal) {}

  common_json_value(const char * pVal);
  common_json_value(const common_json & pVal);
  common_json_value(common_json && pVal);
  // only for the types instantiated in json.cpp, the rest fails at link time
  template <typename T> common_json_value(const std::vector<T> & pVals);
  // a set becomes an array, in the set's own order
  template <typename T> common_json_value(const std::set<T> & pVals);
  // a map becomes an object, keyed in the map's own order
  template <typename T> common_json_value(const std::map<std::string, T> & pVals);
  template <typename T> common_json_value(const std::unordered_map<std::string, T> & pVals);

  // nested object, e.g. {"fn", {{"name", "x"}}}
  // note: a nested pair {"a", "b"} does not build, use common_json::array({"a", "b"}) for an array
  common_json_value(std::initializer_list<common_json_item> pItems);

  template <typename T, typename std::enable_if<std::is_enum<T>::value, int>::type = 0>
  common_json_value(T pVal) : common_json_value((typename std::underlying_type<T>::type) pVal) {}

  template <typename T,
            typename std::enable_if<std::is_integral<T>::value && !std::is_same<T, bool>::value, int>::type = 0>
  common_json_value(T pVal) : type(std::is_signed<T>::value ? cValInt : cValUint) {
    if (std::is_signed<T>::value) {
      vValInt = (int64_t) pVal;
    } else {
      vValUint = (uint64_t) pVal;
    }
  }

  template <typename T, typename std::enable_if<std::is_floating_point<T>::value, int>::type = 0>
  common_json_value(T pVal) : type(cValDouble), vValDouble((double) pVal) {}
};

struct common_json_item {
  std::string       vKey;
  common_json_value vVal;

  template <typename T>
  common_json_item(std::string pKey, T && pVal) : vKey(std::move(pKey)), vVal(std::forward<T>(pVal)) {}

  // a braced list cannot deduce T, so it needs its own overload
  common_json_item(std::string pKey, std::initializer_list<common_json_item> pItems) :
    vKey(std::move(pKey)),
    vVal(pItems) {}
};

// the types common_json_value holds on its own
// anything else reaches its common_json ctor and recurses forever
template <typename T>
struct common_json_is_value
  : std::integral_constant<bool,
                           std::is_enum<T>::value || std::is_arithmetic<T>::value ||
                             std::is_same<T, std::nullptr_t>::value || std::is_same<T, std::string>::value ||
                             std::is_same<T, std::string_view>::value || std::is_same<T, char *>::value ||
                             std::is_same<T, const char *>::value || std::is_same<T, common_json>::value> {};

template <typename T, typename A> struct common_json_is_value<std::vector<T, A>> : std::true_type {};

template <typename T, typename C, typename A> struct common_json_is_value<std::set<T, C, A>> : std::true_type {};

template <typename V, typename C, typename A>
struct common_json_is_value<std::map<std::string, V, C, A>> : std::true_type {};

template <typename V, typename H, typename E, typename A>
struct common_json_is_value<std::unordered_map<std::string, V, H, E, A>> : std::true_type {};

class common_json {
public:
  common_json();
  common_json(const common_json & pOther);
  common_json(common_json && pOther) noexcept;
  common_json(std::initializer_list<common_json_item> pItems);
  common_json(const common_json_value & pVal);

  // direct, a value would need two conversions in a row
  common_json(std::nullptr_t);

  // one step, so that "abc" or a vector can go straight into a common_json
  template <typename T,
            typename std::enable_if<!std::is_same<typename std::decay<T>::type, common_json>::value &&
                                      !std::is_same<typename std::decay<T>::type, common_json_value>::value,
                                    int>::type = 0>
  common_json(T && pVal) : common_json(common_json_value(std::forward<T>(pVal))) {
    static_assert(common_json_is_value<typename std::decay<T>::type>::value,
                  "no common_json_value ctor holds this type, add one instead of letting it recurse");
  }

  // by value, same as the backing library
  // the right side is copied before the left side can invalidate it, e.g. msg["a"] = msg.at("b")
  common_json & operator=(common_json pOther) noexcept;

  ~common_json();

  // throws common_json_error if the text is not valid JSON
  static common_json mParse(const std::string & pText);

  // gives a discarded value instead of throwing, check it with is_discarded()
  static common_json mParseNoThrow(const std::string & pText);

  bool mIsDiscarded() const;

  static common_json mArray();
  static common_json mArray(std::initializer_list<common_json_value> pVals);
  static common_json mObject();
  static common_json mObject(std::initializer_list<common_json_item> pItems);

  // holds a single value, e.g. make("abc").dump() gives "\"abc\""
  static common_json mMake(const common_json_value & pVal);

  bool is_null() const;
  bool is_object() const;
  bool is_array() const;
  bool is_string() const;
  bool is_boolean() const;
  bool is_number() const;
  bool mIsNumberInteger() const;
  bool mIsNumberFloat() const;

  // Overwrite a sensitive string before releasing its storage (used by the password helper).
  void mWipeString();

  bool   empty() const;
  size_t size() const;

  bool contains(const std::string & pKey) const;

  bool operator==(const common_json_value & pVal) const;
  bool operator!=(const common_json_value & pVal) const;

  // at() throws common_json_error if the key is missing, operator[] adds a null value instead
  // note: a const operator[] cannot add, it throws like at()
  common_json &       at(const std::string & pKey);
  const common_json & at(const std::string & pKey) const;
  common_json &       at(size_t pIdx);
  const common_json & at(size_t pIdx) const;

  common_json &       operator[](const std::string & pKey);
  const common_json & operator[](const std::string & pKey) const;

  common_json & operator[](const char * pKey) { return (*this)[std::string(pKey)]; }

  const common_json & operator[](const char * pKey) const { return (*this)[std::string(pKey)]; }

  common_json & operator[](int pIdx) { return (*this)[mToIdx(pIdx)]; }

  const common_json & operator[](int pIdx) const { return (*this)[mToIdx(pIdx)]; }

  common_json &       operator[](size_t pIdx);
  const common_json & operator[](size_t pIdx) const;

  common_json &       front();
  const common_json & front() const;
  common_json &       back();
  const common_json & back() const;

  void clear();

  void erase(const std::string & pKey);
  void erase(size_t pIdx);

  // only for the types instantiated in json.cpp, the rest fails at link time
  template <typename T> T get() const;

  // implicit get<T>() for plain values, so they can be assigned to their C++ type directly
  // note: kept to this short list on purpose, a wider one makes j["key"] ambiguous
  // note: a numeric one would make "str = json;" ambiguous, a number converts to char too
  operator std::string() const;

  template <typename T> T value(const std::string & pKey, T pDef) const {
    return contains(pKey) ? at(pKey).get<T>() : pDef;
  }

  std::string value(const std::string & pKey, const char * pDef) const;

  // a JSON default needs no get<T>(), it is already the right type
  common_json value(const std::string & pKey, const common_json & pDef) const {
    return contains(pKey) ? at(pKey) : pDef;
  }

  void assign(const common_json_value & pVal);
  void set(const common_json_item & pItem);
  void push_back(const common_json_value & pVal);

  // appends one object, e.g. push_back({{"a", 1}})
  void push_back(std::initializer_list<common_json_item> pItems);

  // 1 if the key is there, 0 if not
  size_t count(const std::string & pKey) const;

  // appends every value of another array; inserting an array into itself throws
  void insert(const common_json & pVals);

  // a common_json goes through the copy assignment above, everything else becomes a value
  template <typename T,
            typename std::enable_if<!std::is_same<typename std::decay<T>::type, common_json>::value, int>::type = 0>
  common_json & operator=(T && pVal) {
    assign(common_json_value(std::forward<T>(pVal)));
    return *this;
  }

  std::string mDump(int pIndent = -1) const;

  // same as dump(), but bad UTF-8 gets replaced instead of throwing
  std::string mDumpSafe(int pIndent = -1) const;

  // walks an array by index, or an object in insertion order
  // a plain value gives itself once, same as the backing library
  class iterator {
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type        = common_json;
    using difference_type   = std::ptrdiff_t;
    using pointer           = common_json *;
    using reference         = common_json &;

    iterator(common_json * node, size_t pIdx) : node(node), vIdx(pIdx) {}

    common_json & operator*() const;

    common_json & value() const { return **this; }

    std::string mKey() const;

    iterator & operator++() {
      vIdx++;
      return *this;
    }

    bool operator!=(const iterator & pOther) const { return vIdx != pOther.vIdx; }

    bool operator==(const iterator & pOther) const { return vIdx == pOther.vIdx; }

  private:
    common_json * node;
    size_t        vIdx;
  };

  iterator begin() const;
  iterator end() const;

  // allows: for (const auto & [key, val] : obj.items())
  class items_view {
  public:
    // the members are public, so an entry also works with structured bindings
    struct entry {
      std::string   vK;
      common_json & vV;

      const std::string & mKey() const { return vK; }

      common_json & value() const { return vV; }
    };

    items_view(common_json * node, size_t n) : node(node), n(n) {}

    class iterator {
    public:
      iterator(common_json * node, size_t pIdx) : node(node), vIdx(pIdx) {}

      entry operator*() const;

      iterator & operator++() {
        vIdx++;
        return *this;
      }

      bool operator!=(const iterator & pOther) const { return vIdx != pOther.vIdx; }

    private:
      common_json * node;
      size_t        vIdx;
    };

    iterator begin() const { return iterator(node, 0); }

    iterator end() const { return iterator(node, n); }

  private:
    common_json * node;
    size_t        n;
  };

  items_view mItems() const;

private:
  // a negative index must not turn into a huge size_t
  static size_t mToIdx(int pIdx) {
    if (pIdx < 0) {
      throw common_json_error("negative array index");
    }
    return (size_t) pIdx;
  }

  // the backing value is built here, json.cpp checks that it fits
  // it cannot be a pointer: a value inside a tree would then not be a common_json
  // at() could then only give back a copy instead of a real reference
  alignas(8) unsigned char aStorage[32];
};

using common_json_entry = common_json::items_view::entry;
