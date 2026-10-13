/*
This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <http://unlicense.org>
*/

#ifndef PUBLIC_DOMAIN_BASE64_HPP_
#define PUBLIC_DOMAIN_BASE64_HPP_

#include <cstdint>
#include <iterator>
#include <stdexcept>
#include <string>

class base64_error : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

class base64 {
public:
  enum class alphabet {
    /** the alphabet is detected automatically */
    cAuto,
    /** the standard base64 alphabet is used */
    cStandard,
    /** like `standard` except that the characters `+` and `/` are replaced by `-` and `_` respectively*/
    cUrlFilenameSafe
  };

  enum class decoding_behavior {
    /** if the input is not padded, the remaining bits are ignored */
    cModerate,
    /** if a padding character is encounter decoding is finished */
    cLoose
  };

  /**
     Encodes all the elements from `in_begin` to `in_end` to `out`.

     @warning The source and destination cannot overlap. The destination must be able to hold at least
     `required_encode_size(std::distance(in_begin, in_end))`, otherwise the behavior depends on the output iterator.

     @tparam Input_iterator the source; the returned elements are cast to `std::uint8_t` and should not be greater than
     8 bits
     @tparam Output_iterator the destination; the elements written to it are from the type `char`
     @param in_begin the beginning of the source
     @param in_end the ending of the source
     @param out the destination iterator
     @param alphabet which alphabet should be used
     @returns the iterator to the next element past the last element copied
     @throws see `Input_iterator` and `Output_iterator`
    */
  template <typename Input_iterator, typename Output_iterator>
  static Output_iterator mEncode(Input_iterator  pInBegin,
                                 Input_iterator  pInEnd,
                                 Output_iterator pOut,
                                 alphabet        pAlphabet = alphabet::cStandard) {
    constexpr auto cPad   = '=';
    const char *   vAlpha = pAlphabet == alphabet::cUrlFilenameSafe ?
                              "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_" :
                              "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    while (pInBegin != pInEnd) {
      std::uint8_t vI0 = 0, vI1 = 0, vI2 = 0;

      // first character
      vI0 = static_cast<std::uint8_t>(*pInBegin);
      ++pInBegin;

      *pOut = vAlpha[vI0 >> 2 & 0x3f];
      ++pOut;

      // part of first character and second
      if (pInBegin != pInEnd) {
        vI1 = static_cast<std::uint8_t>(*pInBegin);
        ++pInBegin;

        *pOut = vAlpha[((vI0 & 0x3) << 4) | (vI1 >> 4 & 0x0f)];
        ++pOut;
      } else {
        *pOut = vAlpha[(vI0 & 0x3) << 4];
        ++pOut;

        // last padding
        *pOut = cPad;
        ++pOut;

        // last padding
        *pOut = cPad;
        ++pOut;

        break;
      }

      // part of second character and third
      if (pInBegin != pInEnd) {
        vI2 = static_cast<std::uint8_t>(*pInBegin);
        ++pInBegin;

        *pOut = vAlpha[((vI1 & 0xf) << 2) | (vI2 >> 6 & 0x03)];
        ++pOut;
      } else {
        *pOut = vAlpha[(vI1 & 0xf) << 2];
        ++pOut;

        // last padding
        *pOut = cPad;
        ++pOut;

        break;
      }

      // rest of third
      *pOut = vAlpha[vI2 & 0x3f];
      ++pOut;
    }

    return pOut;
  }

  /**
     Encodes a string.

     @param str the string that should be encoded
     @param alphabet which alphabet should be used
     @returns the encoded base64 string
     @throws see base64::encode()
    */
  static std::string mEncode(const std::string & str, alphabet pAlphabet = alphabet::cStandard) {
    std::string vResult;

    vResult.reserve(mRequiredEncodeSize(str.length()) + 1);

    mEncode(str.begin(), str.end(), std::back_inserter(vResult), pAlphabet);

    return vResult;
  }

  /**
     Encodes a char array.

     @param buffer the char array
     @param size the size of the array
     @param alphabet which alphabet should be used
     @returns the encoded string
    */
  static std::string mEncode(const char * pBuffer, std::size_t size, alphabet pAlphabet = alphabet::cStandard) {
    std::string vResult;

    vResult.reserve(mRequiredEncodeSize(size) + 1);

    mEncode(pBuffer, pBuffer + size, std::back_inserter(vResult), pAlphabet);

    return vResult;
  }

  /**
     Decodes all the elements from `in_begin` to `in_end` to `out`. `in_begin` may point to the same location as `out`,
     in other words: inplace decoding is possible.

     @warning The destination must be able to hold at least `required_decode_size(std::distance(in_begin, in_end))`,
     otherwise the behavior depends on the output iterator.

     @tparam Input_iterator the source; the returned elements are cast to `char`
     @tparam Output_iterator the destination; the elements written to it are from the type `std::uint8_t`
     @param in_begin the beginning of the source
     @param in_end the ending of the source
     @param out the destination iterator
     @param alphabet which alphabet should be used
     @param behavior the behavior when an error was detected
     @returns the iterator to the next element past the last element copied
     @throws base64_error depending on the set behavior
     @throws see `Input_iterator` and `Output_iterator`
    */
  template <typename Input_iterator, typename Output_iterator>
  static Output_iterator mDecode(Input_iterator    pInBegin,
                                 Input_iterator    pInEnd,
                                 Output_iterator   pOut,
                                 alphabet          pAlphabet = alphabet::cAuto,
                                 decoding_behavior pBehavior = decoding_behavior::cModerate) {
    //constexpr auto pad = '=';
    std::uint8_t vLast = 0;
    auto         vBits = 0;

    while (pInBegin != pInEnd) {
      auto vC = *pInBegin;
      ++pInBegin;

      if (vC == '=') {
        break;
      }

      auto vPart = mBase64Value(pAlphabet, vC);

      // enough bits for one byte
      if (vBits + 6 >= 8) {
        *pOut = (vLast << (8 - vBits)) | (vPart >> (vBits - 2));
        ++pOut;

        vBits -= 2;
      } else {
        vBits += 6;
      }

      vLast = vPart;
    }

    // check padding
    if (pBehavior != decoding_behavior::cLoose) {
      while (pInBegin != pInEnd) {
        auto vC = *pInBegin;
        ++pInBegin;

        if (vC != '=') {
          throw base64_error("invalid base64 character.");
        }
      }
    }

    return pOut;
  }

  /**
     Decodes a string.

     @param str the base64 encoded string
     @param alphabet which alphabet should be used
     @param behavior the behavior when an error was detected
     @returns the decoded string
     @throws see base64::decode()
    */
  static std::string mDecode(const std::string & str,
                             alphabet            pAlphabet = alphabet::cAuto,
                             decoding_behavior   pBehavior = decoding_behavior::cModerate) {
    std::string vResult;

    vResult.reserve(mMaxDecodeSize(str.length()));

    mDecode(str.begin(), str.end(), std::back_inserter(vResult), pAlphabet, pBehavior);

    return vResult;
  }

  /**
     Decodes a string.

     @param buffer the base64 encoded buffer
     @param size the size of the buffer
     @param alphabet which alphabet should be used
     @param behavior the behavior when an error was detected
     @returns the decoded string
     @throws see base64::decode()
    */
  static std::string mDecode(const char *      pBuffer,
                             std::size_t       size,
                             alphabet          pAlphabet = alphabet::cAuto,
                             decoding_behavior pBehavior = decoding_behavior::cModerate) {
    std::string vResult;

    vResult.reserve(mMaxDecodeSize(size));

    mDecode(pBuffer, pBuffer + size, std::back_inserter(vResult), pAlphabet, pBehavior);

    return vResult;
  }

  /**
     Decodes a string inplace.

     @param[in,out] str the base64 encoded string
     @param alphabet which alphabet should be used
     @param behavior the behavior when an error was detected
     @throws base64::decode_inplace()
    */
  static void mDecodeInplace(std::string &     str,
                             alphabet          pAlphabet = alphabet::cAuto,
                             decoding_behavior pBehavior = decoding_behavior::cModerate) {
    str.resize(mDecode(str.begin(), str.end(), str.begin(), pAlphabet, pBehavior) - str.begin());
  }

  /**
     Decodes a char array inplace.

     @param[in,out] str the string array
     @param size the length of the array
     @param alphabet which alphabet should be used
     @param behavior the behavior when an error was detected
     @returns the pointer to the next element past the last element decoded
     @throws base64::decode_inplace()
    */
  static char * mDecodeInplace(char *            str,
                               std::size_t       size,
                               alphabet          pAlphabet = alphabet::cAuto,
                               decoding_behavior pBehavior = decoding_behavior::cModerate) {
    return mDecode(str, str + size, str, pAlphabet, pBehavior);
  }

  /**
     Returns the required decoding size for a given size. The value is calculated with the following formula:

     $$
     \lceil \frac{size}{4} \rceil \cdot 3
     $$

     @param size the size of the encoded input
     @returns the size of the resulting decoded buffer; this the absolute maximum
    */
  static std::size_t mMaxDecodeSize(std::size_t size) noexcept { return (size / 4 + (size % 4 ? 1 : 0)) * 3; }

  /**
     Returns the required encoding size for a given size. The value is calculated with the following formula:

     $$
     \lceil \frac{size}{3} \rceil \cdot 4
     $$

     @param size the size of the decoded input
     @returns the size of the resulting encoded buffer
    */
  static std::size_t mRequiredEncodeSize(std::size_t size) noexcept { return (size / 3 + (size % 3 ? 1 : 0)) * 4; }

private:
  static std::uint8_t mBase64Value(alphabet & pAlphabet, char pC) {
    if (pC >= 'A' && pC <= 'Z') {
      return pC - 'A';
    } else if (pC >= 'a' && pC <= 'z') {
      return pC - 'a' + 26;
    } else if (pC >= '0' && pC <= '9') {
      return pC - '0' + 52;
    }

    // comes down to alphabet
    if (pAlphabet == alphabet::cStandard) {
      if (pC == '+') {
        return 62;
      } else if (pC == '/') {
        return 63;
      }
    } else if (pAlphabet == alphabet::cUrlFilenameSafe) {
      if (pC == '-') {
        return 62;
      } else if (pC == '_') {
        return 63;
      }
    }  // auto detect
    else {
      if (pC == '+') {
        pAlphabet = alphabet::cStandard;

        return 62;
      } else if (pC == '/') {
        pAlphabet = alphabet::cStandard;

        return 63;
      } else if (pC == '-') {
        pAlphabet = alphabet::cUrlFilenameSafe;

        return 62;
      } else if (pC == '_') {
        pAlphabet = alphabet::cUrlFilenameSafe;

        return 63;
      }
    }

    throw base64_error("invalid base64 character.");
  }
};

#endif  // !PUBLIC_DOMAIN_BASE64_HPP_
