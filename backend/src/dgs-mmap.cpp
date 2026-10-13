#include "dgs-mmap.h"

#include "dgs-impl.h"
#include "ggml.h"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#ifdef __has_include
#  if __has_include(<unistd.h>)
#    include <fcntl.h>
#    include <sys/stat.h>
#    include <unistd.h>
#    if defined(_POSIX_MAPPED_FILES)
#      include <sys/mman.h>
#    endif
#    if defined(_POSIX_MEMLOCK_RANGE)
#      include <sys/resource.h>
#    endif
#  endif
#endif

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#  ifndef PATH_MAX
#    define PATH_MAX MAX_PATH
#  endif
#  include <io.h>
#endif

#if defined(__APPLE__)
#  include <TargetConditionals.h>
#endif

#ifdef _WIN32
#  define dgs_mmap_ftell _ftelli64
#  define dgs_mmap_fseek _fseeki64
#else
#  define dgs_mmap_ftell ftello
#  define dgs_mmap_fseek fseeko
#endif

// TODO: consider moving to dgs-impl.h if needed in more places
#if defined(_WIN32)
static std::string dgs_format_win_err(DWORD err) {
  LPSTR  buf;
  size_t size =
    FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL,
                   err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR) &buf, 0, NULL);
  if (!size) {
    return "FormatMessageA failed";
  }
  std::string ret(buf, size);
  LocalFree(buf);
  return ret;
}
#endif

// dgs_file

struct dgs_file::impl {
#if defined(_WIN32)
  HANDLE fp_win32;

  std::string GetErrorMessageWin32(DWORD error_code) const {
    std::string ret;
    LPSTR       lpMsgBuf = NULL;
    DWORD       bufLen =
      FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL,
                     error_code, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR) &lpMsgBuf, 0, NULL);
    if (!bufLen) {
      ret = format("Win32 error code: %lx", error_code);
    } else {
      ret = lpMsgBuf;
      LocalFree(lpMsgBuf);
    }

    return ret;
  }

  impl(const char * fname, const char * mode, [[maybe_unused]] const bool use_direct_io = false) {
    fp = ggml_fopen(fname, mode);
    if (fp == NULL) {
      throw std::runtime_error(format("failed to open %s: %s", fname, strerror(errno)));
    }
    fp_win32 = (HANDLE) _get_osfhandle(_fileno(fp));
    seek(0, SEEK_END);
    size = tell();
    seek(0, SEEK_SET);
  }

  impl(FILE * file) : owns_fp(false) {
    fp       = file;
    fp_win32 = (HANDLE) _get_osfhandle(_fileno(fp));
    seek(0, SEEK_END);
    size = tell();
    seek(0, SEEK_SET);
  }

  size_t tell() const {
    LARGE_INTEGER li;
    li.QuadPart = 0;
    BOOL ret    = SetFilePointerEx(fp_win32, li, &li, FILE_CURRENT);
    if (!ret) {
      throw std::runtime_error(format("read error: %s", GetErrorMessageWin32(GetLastError()).c_str()));
    }

    return li.QuadPart;
  }

  void seek(size_t offset, int whence) const {
    static_assert(SEEK_SET == FILE_BEGIN, "SEEK_SET != FILE_BEGIN");
    static_assert(SEEK_CUR == FILE_CURRENT, "SEEK_CUR != FILE_CURRENT");
    static_assert(SEEK_END == FILE_END, "SEEK_END != FILE_END");

    LARGE_INTEGER li;
    li.QuadPart = offset;
    BOOL ret    = SetFilePointerEx(fp_win32, li, NULL, whence);
    if (!ret) {
      throw std::runtime_error(format("read error: %s", GetErrorMessageWin32(GetLastError()).c_str()));
    }
  }

  void read_raw(void * ptr, size_t len) {
    size_t bytes_read = 0;
    while (bytes_read < len) {
      size_t chunk_size = std::min<size_t>(len - bytes_read, 64 * 1024 * 1024);
      DWORD  chunk_read = 0;
      BOOL   result     = ReadFile(fp_win32, reinterpret_cast<char *>(ptr) + bytes_read, chunk_size, &chunk_read, NULL);
      if (!result) {
        throw std::runtime_error(format("read error: %s", GetErrorMessageWin32(GetLastError()).c_str()));
      }
      if (chunk_read < chunk_size || chunk_read == 0) {
        throw std::runtime_error("unexpectedly reached end of file");
      }

      bytes_read += chunk_read;
    }
  }

  uint32_t read_u32() {
    uint32_t val;
    read_raw(&val, sizeof(val));
    return val;
  }

  void write_raw(const void * ptr, size_t len) const {
    size_t bytes_written = 0;
    while (bytes_written < len) {
      size_t chunk_size    = std::min<size_t>(len - bytes_written, 64 * 1024 * 1024);
      DWORD  chunk_written = 0;
      BOOL   result =
        WriteFile(fp_win32, reinterpret_cast<const char *>(ptr) + bytes_written, chunk_size, &chunk_written, NULL);
      if (!result) {
        throw std::runtime_error(format("write error: %s", GetErrorMessageWin32(GetLastError()).c_str()));
      }
      if (chunk_written < chunk_size || chunk_written == 0) {
        throw std::runtime_error("unexpectedly failed to write bytes");
      }

      bytes_written += chunk_written;
    }
  }

  void write_u32(uint32_t val) const { write_raw(&val, sizeof(val)); }

  bool has_direct_io() const { return true; }

  ~impl() {
    if (fp && owns_fp) {
      std::fclose(fp);
    }
  }
#else
  impl(const char * pFname, const char * pMode, [[maybe_unused]] const bool pUseDirectIo = false) : vFname(pFname) {
#  ifdef __linux__
    // Try unbuffered I/O for read only
    if (pUseDirectIo && std::strcmp(pMode, "rb") == 0) {
      if (mInitFd()) {
        return;
      }
      DGS_LOG_WARN("Failed to open file '%s' with error: %s. Falling back to buffered I/O", pFname, strerror(errno));
    }
#  endif
    mInitFp(pMode);
  }

#  ifdef __linux__
  bool mInitFd() {
    vFd = open(vFname.c_str(), O_RDONLY | O_DIRECT);

    if (vFd != -1) {
      struct stat vFileStats{};
      fstat(vFd, &vFileStats);

      size       = vFileStats.st_size;
      vAlignment = vFileStats.st_blksize;

      off_t vRet = lseek(vFd, 0, SEEK_SET);
      if (vRet == -1) {
        throw std::runtime_error(format("seek error: %s", strerror(errno)));
      }
      return true;
    }
    return false;
  }
#  endif

  void mInitFp(const char * pMode) {
    vFp = ggml_fopen(vFname.c_str(), pMode);
    if (vFp == NULL) {
      throw std::runtime_error(format("failed to open %s: %s", vFname.c_str(), strerror(errno)));
    }
    mSeek(0, SEEK_END);
    size = mTell();
    mSeek(0, SEEK_SET);
  }

  impl(FILE * pFile) : vFname("(file*)"), vOwnsFp(false) {
    vFp = pFile;
    mSeek(0, SEEK_END);
    size = mTell();
    mSeek(0, SEEK_SET);
  }

  size_t mTell() const {
    if (vFd == -1) {
      off_t vRet = dgs_mmap_ftell(vFp);
      if (vRet == -1) {
        throw std::runtime_error(format("ftell error: %s", strerror(errno)));
      }

      return (size_t) vRet;
    }

    off_t vPos = lseek(vFd, 0, SEEK_CUR);
    if (vPos == -1) {
      throw std::runtime_error(format("lseek error: %s", strerror(errno)));
    }
    return (size_t) vPos;
  }

  void mSeek(size_t pOffset, int pWhence) const {
    off_t vRet = 0;
    if (vFd == -1) {
      vRet = dgs_mmap_fseek(vFp, pOffset, pWhence);
    } else {
      vRet = lseek(vFd, pOffset, pWhence);
    }
    if (vRet == -1) {
      throw std::runtime_error(format("seek error: %s", strerror(errno)));
    }
  }

  void mReadRawUnsafe(void * pPtr, size_t pLen) {
    if (pLen == 0) {
      return;
    }
    errno = 0;
    if (vFd == -1) {
      const size_t cCurrOff = mTell();
      const size_t cToRead  = std::min(pLen, size - cCurrOff);

      std::size_t vRet = std::fread(pPtr, cToRead, 1, vFp);
      if (ferror(vFp)) {
        throw std::runtime_error(format("read error: %s", strerror(errno)));
      }
      if (cToRead > 0 && vRet != 1) {
        throw std::runtime_error("unexpectedly reached end of file");
      }
    } else {
      size_t vBytesRead = 0;
      while (vBytesRead < pLen) {
        const size_t cToRead = pLen - vBytesRead;
        ssize_t      vRet    = ::read(vFd, reinterpret_cast<char *>(pPtr) + vBytesRead, cToRead);

        if (vRet == -1) {
          if (errno == EINTR) {
            continue;  // Interrupted by signal, retry
          }
          // Fallback to std::fread in case the DMA controller cannot access the buffer
          if (errno == EFAULT || errno == EINVAL) {
            DGS_LOG_WARN("%s: Falling back to buffered IO due to %s\n", __func__, strerror(errno));
            auto vCurrOff = mTell();
            close(vFd);
            vFd        = -1;
            vAlignment = 1;
            mInitFp("rb");
            mSeek(vCurrOff, SEEK_SET);
            mReadRawUnsafe(pPtr, pLen);
            return;
          }
          throw std::runtime_error(format("read error: %s", strerror(errno)));
        }
        if (vRet == 0) {
          // EOF: allow if this read was only pulling alignment padding past file end
          off_t vPos = lseek(vFd, 0, SEEK_CUR);
          if (vPos != -1 && (size_t) vPos == size) {
            std::memset(reinterpret_cast<char *>(pPtr) + vBytesRead, 0, pLen - vBytesRead);
            return;
          }
          throw std::runtime_error("unexpectedly reached end of file");
        }

        vBytesRead += (size_t) vRet;
      }
    }
  }

  void mReadAlignedChunk(void * pDest, size_t size) {
    size_t vOffset              = mTell();
    off_t  vAlignedOffset       = vOffset & ~(vAlignment - 1);
    off_t  vOffsetFromAlignment = vOffset - vAlignedOffset;
    size_t vBytesToRead         = (vOffsetFromAlignment + size + vAlignment - 1) & ~(vAlignment - 1);

    // stage through a bounded buffer, so that a large tensor is not held in memory twice while it loads
    const size_t cBufferSize = std::min<size_t>(vBytesToRead, DGS_DIRECT_IO_BUFFER_SIZE);

    void * vRawBuffer = nullptr;
    int    vRet       = posix_memalign(&vRawBuffer, vAlignment, cBufferSize);
    if (vRet != 0) {
      throw std::runtime_error(format("posix_memalign failed with error %d", vRet));
    }

    struct aligned_buffer_deleter {
      void operator()(void * pP) const { free(pP); }
    };

    std::unique_ptr<void, aligned_buffer_deleter> vBuffer(vRawBuffer);

    mSeek(vAlignedOffset, SEEK_SET);

    size_t vSkip   = vOffsetFromAlignment;
    size_t vCopied = 0;
    for (size_t vDone = 0; vDone < vBytesToRead;) {
      const size_t n = std::min(cBufferSize, vBytesToRead - vDone);
      mReadRawUnsafe(vBuffer.get(), n);

      const size_t count = std::min(n - vSkip, size - vCopied);
      memcpy(reinterpret_cast<char *>(pDest) + vCopied, reinterpret_cast<char *>(vBuffer.get()) + vSkip, count);

      vCopied += count;
      vSkip = 0;
      vDone += n;
    }
  }

  void mReadRaw(void * pPtr, size_t pLen) {
    if (mHasDirectIo()) {
      mReadAlignedChunk(pPtr, pLen);
    } else {
      mReadRawUnsafe(pPtr, pLen);
    }
  }

  uint32_t mReadU32() {
    uint32_t vRet;
    mReadRaw(&vRet, sizeof(vRet));
    return vRet;
  }

  void mWriteRaw(const void * pPtr, size_t pLen) const {
    if (pLen == 0) {
      return;
    }
    errno       = 0;
    size_t vRet = std::fwrite(pPtr, pLen, 1, vFp);
    if (vRet != 1) {
      throw std::runtime_error(format("write error: %s", strerror(errno)));
    }
  }

  void mWriteU32(uint32_t pVal) const { mWriteRaw(&pVal, sizeof(pVal)); }

  bool mHasDirectIo() const { return vFd != -1 && vAlignment > 1; }

  ~impl() {
    if (vFd != -1) {
      close(vFd);
    } else if (vOwnsFp) {
      std::fclose(vFp);
    }
  }

  int         vFd = -1;
  std::string vFname;
#endif

  size_t mReadAlignment() const { return vAlignment; }

  size_t vAlignment = 1;

  FILE * vFp{};
  size_t size{};
  bool   vOwnsFp = true;
};

dgs_file::dgs_file(const char * pFname, const char * pMode, const bool pUseDirectIo) :
  vPimpl(std::make_unique<impl>(pFname, pMode, pUseDirectIo)) {}

dgs_file::dgs_file(FILE * pFile) : vPimpl(std::make_unique<impl>(pFile)) {}

dgs_file::~dgs_file() = default;

size_t dgs_file::mTell() const {
  return vPimpl->mTell();
}

size_t dgs_file::size() const {
  return vPimpl->size;
}

size_t dgs_file::mReadAlignment() const {
  return vPimpl->mReadAlignment();
}

bool dgs_file::mHasDirectIo() const {
  return vPimpl->mHasDirectIo();
}

int dgs_file::mFileId() const {
#ifdef _WIN32
  return _fileno(pimpl->fp);
#else
  if (vPimpl->vFd != -1) {
    return vPimpl->vFd;
  }
#  if defined(fileno)
  return fileno(pimpl->fp);
#  else
  return ::fileno(vPimpl->vFp);
#  endif
#endif
}

void dgs_file::mSeek(size_t pOffset, int pWhence) const {
  vPimpl->mSeek(pOffset, pWhence);
}

void dgs_file::mReadRaw(void * pPtr, size_t pLen) {
  vPimpl->mReadRaw(pPtr, pLen);
}
#ifdef _WIN32
void dgs_file::read_raw_unsafe(void * ptr, size_t len) {
  pimpl->read_raw(ptr, len);
}
#else
void dgs_file::mReadRawUnsafe(void * pPtr, size_t pLen) {
  vPimpl->mReadRawUnsafe(pPtr, pLen);
}
#endif

uint32_t dgs_file::mReadU32() {
  return vPimpl->mReadU32();
}

void dgs_file::mWriteRaw(const void * pPtr, size_t pLen) const {
  vPimpl->mWriteRaw(pPtr, pLen);
}

void dgs_file::mWriteU32(uint32_t pVal) const {
  vPimpl->mWriteU32(pVal);
}

// dgs_mmap

#if defined(_POSIX_MAPPED_FILES) || defined(_WIN32)
// merge `ranges` and return their complement within [0, limit)
static dgs_mmap::ranges fRangesComplement(dgs_mmap::ranges pRanges, size_t pLimit) {
  dgs_mmap::ranges lRes;
  std::sort(pRanges.begin(), pRanges.end());

  size_t vPos = 0;
  for (const auto & vRange : pRanges) {
    const size_t cBeg = std::min(vRange.first, pLimit);
    const size_t end  = std::min(vRange.second, pLimit);
    if (cBeg > vPos) {
      lRes.emplace_back(vPos, cBeg);
    }
    vPos = std::max(vPos, end);
  }
  if (vPos < pLimit) {
    lRes.emplace_back(vPos, pLimit);
  }

  return lRes;
}
#endif

struct dgs_mmap::impl {
#ifdef _POSIX_MAPPED_FILES
  std::vector<std::pair<size_t, size_t>> lMappedFragments;

  impl(struct dgs_file * pFile, size_t pPrefetch, bool pNuma, const dgs_mmap::ranges & pLazyRanges) {
    size       = pFile->size();
    int vFd    = pFile->mFileId();
    int vFlags = MAP_SHARED;
    if (pNuma) {
      pPrefetch = 0;
    }
#  ifdef __linux__
    if (posix_fadvise(vFd, 0, 0, POSIX_FADV_SEQUENTIAL)) {
      DGS_LOG_WARN("warning: posix_fadvise(.., POSIX_FADV_SEQUENTIAL) failed: %s\n", strerror(errno));
    }
    // MAP_POPULATE would fault in the lazy ranges too
    if (pPrefetch && pLazyRanges.empty()) {
      vFlags |= MAP_POPULATE;
    }
#  endif
    vAddr = mmap(NULL, pFile->size(), PROT_READ, vFlags, vFd, 0);
    if (vAddr == MAP_FAILED) {
      throw std::runtime_error(format("mmap failed: %s", strerror(errno)));
    }

    // page-aligned madvise over [beg, end), clamped to the file
    auto vAdvise = [&](size_t pBeg, size_t end, int pAdvice, const char * name) {
      const size_t cPageSize = sysconf(_SC_PAGESIZE);
      pBeg                   = pBeg & ~(cPageSize - 1);
      end                    = std::min((end + cPageSize - 1) & ~(cPageSize - 1), pFile->size());
      if (pBeg >= end) {
        return;
      }
      if (posix_madvise((char *) vAddr + pBeg, end - pBeg, pAdvice)) {
        DGS_LOG_WARN("warning: posix_madvise(.., %s) failed: %s\n", name, strerror(errno));
      }
    };

    if (pPrefetch > 0) {
      for (const auto & vRange : fRangesComplement(pLazyRanges, std::min(pFile->size(), pPrefetch))) {
        vAdvise(vRange.first, vRange.second, POSIX_MADV_WILLNEED, "POSIX_MADV_WILLNEED");
      }
    }
    for (const auto & vRange : pLazyRanges) {
      vAdvise(vRange.first, vRange.second, POSIX_MADV_RANDOM, "POSIX_MADV_RANDOM");
    }
    if (pNuma) {
      if (posix_madvise(vAddr, pFile->size(), POSIX_MADV_RANDOM)) {
        DGS_LOG_WARN("warning: posix_madvise(.., POSIX_MADV_RANDOM) failed: %s\n", strerror(errno));
      }
    }

    lMappedFragments.emplace_back(0, pFile->size());
  }

  static void mAlignRange(size_t * pFirst, size_t * pLast, size_t pPageSize) {
    size_t vOffsetInPage = *pFirst & (pPageSize - 1);
    size_t vOffsetToPage = vOffsetInPage == 0 ? 0 : pPageSize - vOffsetInPage;
    *pFirst += vOffsetToPage;

    *pLast = *pLast & ~(pPageSize - 1);

    if (*pLast <= *pFirst) {
      *pLast = *pFirst;
    }
  }

  void mUnmapFragment(size_t pFirst, size_t pLast) {
    int vPageSize = sysconf(_SC_PAGESIZE);
    mAlignRange(&pFirst, &pLast, vPageSize);
    size_t vLen = pLast - pFirst;

    if (vLen == 0) {
      return;
    }

    GGML_ASSERT(pFirst % vPageSize == 0);
    GGML_ASSERT(pLast % vPageSize == 0);
    GGML_ASSERT(pLast > pFirst);

    void * vNextPageStart = (uint8_t *) vAddr + pFirst;

    if (munmap(vNextPageStart, vLen)) {
      DGS_LOG_WARN("warning: munmap failed: %s\n", strerror(errno));
    }

    std::vector<std::pair<size_t, size_t>> lNewMappedFragments;
    for (const auto & vFrag : lMappedFragments) {
      if (vFrag.first < pFirst && vFrag.second > pLast) {
        lNewMappedFragments.emplace_back(vFrag.first, pFirst);
        lNewMappedFragments.emplace_back(pLast, vFrag.second);
      } else if (vFrag.first < pFirst && vFrag.second > pFirst) {
        lNewMappedFragments.emplace_back(vFrag.first, pFirst);
      } else if (vFrag.first < pLast && vFrag.second > pLast) {
        lNewMappedFragments.emplace_back(pLast, vFrag.second);
      } else if (vFrag.first >= pFirst && vFrag.second <= pLast) {
      } else {
        lNewMappedFragments.push_back(vFrag);
      }
    }
    lMappedFragments = std::move(lNewMappedFragments);
  }

  ~impl() {
    for (const auto & vFrag : lMappedFragments) {
      if (munmap((char *) vAddr + vFrag.first, vFrag.second - vFrag.first)) {
        DGS_LOG_WARN("warning: munmap failed: %s\n", strerror(errno));
      }
    }
  }
#elif defined(_WIN32)
  HANDLE hMapping = nullptr;

  impl(struct dgs_file * file, size_t prefetch, bool numa, const dgs_mmap::ranges & lazy_ranges) {
    GGML_UNUSED(numa);

    size = file->size();

    HANDLE hFile = (HANDLE) _get_osfhandle(file->file_id());

    hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);

    if (hMapping == NULL) {
      DWORD error = GetLastError();
      throw std::runtime_error(format("CreateFileMappingA failed: %s", dgs_format_win_err(error).c_str()));
    }

    addr        = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
    DWORD error = GetLastError();

    if (addr == NULL) {
      CloseHandle(hMapping);
      throw std::runtime_error(format("MapViewOfFile failed: %s", dgs_format_win_err(error).c_str()));
    }

    if (prefetch > 0) {
#  if _WIN32_WINNT >= 0x602
      BOOL(WINAPI * pPrefetchVirtualMemory)(HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
      HMODULE hKernel32 = GetModuleHandleW(L"kernel32.dll");

      pPrefetchVirtualMemory =
        (decltype(pPrefetchVirtualMemory)) (void *) GetProcAddress(hKernel32, "PrefetchVirtualMemory");

      if (pPrefetchVirtualMemory) {
        std::vector<WIN32_MEMORY_RANGE_ENTRY> entries;
        for (const auto & range : ranges_complement(lazy_ranges, std::min(size, prefetch))) {
          WIN32_MEMORY_RANGE_ENTRY entry;
          entry.VirtualAddress = (char *) addr + range.first;
          entry.NumberOfBytes  = (SIZE_T) (range.second - range.first);
          entries.push_back(entry);
        }
        if (!entries.empty() &&
            !pPrefetchVirtualMemory(GetCurrentProcess(), (ULONG_PTR) entries.size(), entries.data(), 0)) {
          DGS_LOG_WARN("warning: PrefetchVirtualMemory failed: %s\n", dgs_format_win_err(GetLastError()).c_str());
        }
      }
#  else
      DGS_LOG_DEBUG("skipping PrefetchVirtualMemory because _WIN32_WINNT < 0x602\n");
#  endif
    }
  }

  void unmap_fragment(size_t first, size_t last) {
    GGML_UNUSED(first);
    GGML_UNUSED(last);
  }

  ~impl() {
    if (hMapping) {
      if (addr) {
        if (!UnmapViewOfFile(addr)) {
          DGS_LOG_WARN("warning: UnmapViewOfFile failed: %s\n", dgs_format_win_err(GetLastError()).c_str());
        }
      }
      if (!CloseHandle(hMapping)) {
        DGS_LOG_WARN("warning: CloseHandle failed: %s\n", dgs_format_win_err(GetLastError()).c_str());
      }
    }
  }
#else
  impl(struct dgs_file * file, size_t prefetch, bool numa, const dgs_mmap::ranges & lazy_ranges) {
    GGML_UNUSED(file);
    GGML_UNUSED(prefetch);
    GGML_UNUSED(numa);
    GGML_UNUSED(lazy_ranges);

    throw std::runtime_error("mmap not supported");
  }

  void unmap_fragment(size_t first, size_t last) {
    GGML_UNUSED(first);
    GGML_UNUSED(last);

    throw std::runtime_error("mmap not supported");
  }
#endif

  void * vAddr;
  size_t size;
};

dgs_mmap::dgs_mmap(struct dgs_file * pFile, size_t pPrefetch, bool pNuma, const ranges & pLazyRanges) :
  vPimpl(std::make_unique<impl>(pFile, pPrefetch, pNuma, pLazyRanges)) {}

dgs_mmap::~dgs_mmap() = default;

size_t dgs_mmap::size() const {
  return vPimpl->size;
}

void * dgs_mmap::mAddr() const {
  return vPimpl->vAddr;
}

void dgs_mmap::mUnmapFragment(size_t pFirst, size_t pLast) {
  vPimpl->mUnmapFragment(pFirst, pLast);
}

#if defined(_POSIX_MEMLOCK_RANGE) || defined(_WIN32)
const bool dgs_mmap::cSupported = true;
#else
const bool dgs_mmap::SUPPORTED = false;
#endif

// dgs_mlock

struct dgs_mlock::impl {
#ifdef _POSIX_MEMLOCK_RANGE
  static size_t mLockGranularity() { return (size_t) sysconf(_SC_PAGESIZE); }

  bool mRawLock(const void * pAddr, size_t size) const {
    if (!mlock(pAddr, size)) {
      return true;
    }

#  ifdef __APPLE__
#    define MLOCK_SUGGESTION                                                                          \
      "Try increasing the sysctl values 'vm.user_wire_limit' and 'vm.global_user_wire_limit' and/or " \
      "decreasing 'vm.global_no_user_wire_amount'.  Also try increasing RLIMIT_MEMLOCK (ulimit -l).\n"
#  else
#    define MLOCK_SUGGESTION "Try increasing RLIMIT_MEMLOCK ('ulimit -l' as root).\n"
#  endif

    char * vErrmsg  = std::strerror(errno);
    bool   vSuggest = (errno == ENOMEM);
#  if defined(TARGET_OS_VISION) || defined(TARGET_OS_TV) || defined(_AIX) || defined(__HAIKU__)
    // visionOS/tvOS/Haiku don't support RLIMIT_MEMLOCK
    // Skip resource limit checks on these platforms
    suggest = false;
#  else
    struct rlimit vLockLimit;
    if (vSuggest && getrlimit(RLIMIT_MEMLOCK, &vLockLimit)) {
      vSuggest = false;
    }
    if (vSuggest && ((uint64_t) vLockLimit.rlim_max > (uint64_t) vLockLimit.rlim_cur + size)) {
      vSuggest = false;
    }
#  endif

    DGS_LOG_WARN("warning: failed to mlock %zu-byte buffer (after previously locking %zu bytes): %s\n%s", size,
                 this->size, vErrmsg, vSuggest ? MLOCK_SUGGESTION : "");
    return false;
  }

  static void mRawUnlock(void * pAddr, size_t size) {
    if (munlock(pAddr, size)) {
      DGS_LOG_WARN("warning: failed to munlock buffer: %s\n", std::strerror(errno));
    }
  }
#elif defined(_WIN32)
  static size_t lock_granularity() {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (size_t) si.dwPageSize;
  }

  bool raw_lock(void * ptr, size_t len) const {
    for (int tries = 1;; tries++) {
      if (VirtualLock(ptr, len)) {
        return true;
      }
      if (tries == 2) {
        DGS_LOG_WARN("warning: failed to VirtualLock %zu-byte buffer (after previously locking %zu bytes): %s\n", len,
                     size, dgs_format_win_err(GetLastError()).c_str());
        return false;
      }

      SIZE_T min_ws_size, max_ws_size;
      if (!GetProcessWorkingSetSize(GetCurrentProcess(), &min_ws_size, &max_ws_size)) {
        DGS_LOG_WARN("warning: GetProcessWorkingSetSize failed: %s\n", dgs_format_win_err(GetLastError()).c_str());
        return false;
      }
      size_t increment = len + 1048576;
      min_ws_size += increment;
      max_ws_size += increment;
      if (!SetProcessWorkingSetSize(GetCurrentProcess(), min_ws_size, max_ws_size)) {
        DGS_LOG_WARN("warning: SetProcessWorkingSetSize failed: %s\n", dgs_format_win_err(GetLastError()).c_str());
        return false;
      }
    }
  }

  static void raw_unlock(void * ptr, size_t len) {
    if (!VirtualUnlock(ptr, len)) {
      DGS_LOG_WARN("warning: failed to VirtualUnlock buffer: %s\n", dgs_format_win_err(GetLastError()).c_str());
    }
  }
#else
  static size_t lock_granularity() { return (size_t) 65536; }

  bool raw_lock(const void * addr, size_t len) const {
    DGS_LOG_WARN("warning: mlock not supported on this system\n");
    return false;
  }

  static void raw_unlock(const void * addr, size_t len) {}
#endif

  impl() : vAddr(NULL), size(0), vFailedAlready(false) {}

  void init(void * pPtr) {
    GGML_ASSERT(vAddr == NULL && size == 0);
    vAddr = pPtr;
  }

  void mGrowTo(size_t pTargetSize) {
    GGML_ASSERT(vAddr);
    if (vFailedAlready) {
      return;
    }
    size_t vGranularity = mLockGranularity();
    pTargetSize         = (pTargetSize + vGranularity - 1) & ~(vGranularity - 1);
    if (pTargetSize > size) {
      if (mRawLock((uint8_t *) vAddr + size, pTargetSize - size)) {
        size = pTargetSize;
      } else {
        vFailedAlready = true;
      }
    }
  }

  void * vAddr;
  size_t size;

  bool vFailedAlready;
};

dgs_mlock::dgs_mlock() : vPimpl(std::make_unique<impl>()) {}

dgs_mlock::~dgs_mlock() = default;

void dgs_mlock::init(void * pPtr) {
  vPimpl->init(pPtr);
}

void dgs_mlock::mGrowTo(size_t pTargetSize) {
  vPimpl->mGrowTo(pTargetSize);
}

#if defined(_POSIX_MEMLOCK_RANGE) || defined(_WIN32)
const bool dgs_mlock::cSupported = true;
#else
const bool dgs_mlock::SUPPORTED = false;
#endif

void fDgsPrefetch(dgs_memory_ranges pMr) {
#if defined(__linux__) || (defined(_WIN32) && _WIN32_WINNT >= 0x602)
  if (pMr.empty()) {
    return;
  }

#  if defined(_WIN32)
  using prefetch_virtual_memory_t = BOOL(WINAPI *)(HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
  static const auto pPrefetchVirtualMemory =
    (prefetch_virtual_memory_t) (void *) GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "PrefetchVirtualMemory");
  if (!pPrefetchVirtualMemory) {
    return;
  }

  static const long page_size = [] {
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    return (long) info.dwPageSize;
  }();
#  else
  static const long cPageSize = sysconf(_SC_PAGESIZE);
#  endif
  if (cPageSize <= 0) {
    return;
  }

  const size_t cPage = (size_t) cPageSize;
  std::sort(pMr.begin(), pMr.end(), [](const dgs_memory_range & pA, const dgs_memory_range & pB) {
    return (uintptr_t) pA.vAddr < (uintptr_t) pB.vAddr;
  });

  uintptr_t begin = 0, end = 0;
#  if defined(_WIN32)
  // collect the mr and prefetch them in one call, so the reads can be issued concurrently
  std::vector<WIN32_MEMORY_RANGE_ENTRY> entries;
  auto                                  prefetch = [&]() {
    entries.push_back({ (PVOID) begin, (SIZE_T) (end - begin) });
    return true;
  };
#  else
  auto vPrefetch = [&]() {
    if (madvise((void *) begin, end - begin, MADV_WILLNEED) != 0) {
      DGS_LOG_WARN("dgs_prefetch: madvise(MADV_WILLNEED) failed: %s\n", strerror(errno));
      return false;
    }
    return true;
  };
#  endif
  for (const auto & vRange : pMr) {
    if (!vRange.vAddr || vRange.size == 0) {
      continue;
    }
    const uintptr_t pointer = (uintptr_t) vRange.vAddr;
    const uintptr_t cFirst  = pointer / cPage * cPage;
    const uintptr_t cLast   = (pointer + vRange.size + cPage - 1) / cPage * cPage;
    if (end && cFirst > end) {
      if (!vPrefetch()) {
        return;
      }
      end = 0;
    }
    if (!end) {
      begin = cFirst;
    }
    end = std::max(end, cLast);
  }
  if (end) {
    vPrefetch();
  }
#  if defined(_WIN32)
  if (!entries.empty() && !pPrefetchVirtualMemory(GetCurrentProcess(), (ULONG_PTR) entries.size(), entries.data(), 0)) {
    DGS_LOG_WARN("dgs_prefetch: PrefetchVirtualMemory failed: %s\n", dgs_format_win_err(GetLastError()).c_str());
  }
#  endif
#else
  GGML_UNUSED(mr);
#endif
}

size_t fDgsPathMax() {
  return PATH_MAX;
}
