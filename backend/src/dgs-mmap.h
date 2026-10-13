#pragma once

#include <cstdint>
#include <cstdio>
#include <memory>
#include <utility>
#include <vector>

// staging buffer size for direct I/O reads, 64MB works well for NVMe drives
#define DGS_DIRECT_IO_BUFFER_SIZE (64 * 1024 * 1024)

struct dgs_file;
struct dgs_mmap;
struct dgs_mlock;

using dgs_files  = std::vector<std::unique_ptr<dgs_file>>;
using dgs_mmaps  = std::vector<std::unique_ptr<dgs_mmap>>;
using dgs_mlocks = std::vector<std::unique_ptr<dgs_mlock>>;

struct dgs_file {
  dgs_file(const char * pFname, const char * pMode, bool pUseDirectIo = false);
  dgs_file(FILE * pFile);
  ~dgs_file();

  size_t mTell() const;
  size_t size() const;

  int mFileId() const;  // fileno overload

  void mSeek(size_t pOffset, int pWhence) const;

  void     mReadRaw(void * pPtr, size_t pLen);
  void     mReadRawUnsafe(void * pPtr, size_t pLen);
  void     mReadAlignedChunk(void * pDest, size_t size);
  uint32_t mReadU32();

  void mWriteRaw(const void * pPtr, size_t pLen) const;
  void mWriteU32(uint32_t pVal) const;

  size_t mReadAlignment() const;
  bool   mHasDirectIo() const;
private:
  struct impl;
  std::unique_ptr<impl> vPimpl;
};

struct dgs_mmap {
  // list of [first, last) byte ranges within a file
  using ranges = std::vector<std::pair<size_t, size_t>>;

  dgs_mmap(const dgs_mmap &) = delete;
  dgs_mmap(struct dgs_file * pFile,
           size_t            pPrefetch   = (size_t) -1,
           bool              pNuma       = false,
           const ranges &    pLazyRanges = {});
  ~dgs_mmap();

  size_t size() const;
  void * mAddr() const;

  void mUnmapFragment(size_t pFirst, size_t pLast);

  static const bool cSupported;

private:
  struct impl;
  std::unique_ptr<impl> vPimpl;
};

struct dgs_mlock {
  dgs_mlock();
  ~dgs_mlock();

  void init(void * pPtr);
  void mGrowTo(size_t pTargetSize);

  static const bool cSupported;

private:
  struct impl;
  std::unique_ptr<impl> vPimpl;
};

struct dgs_memory_range {
  const void * vAddr;
  size_t       size;
};

using dgs_memory_ranges = std::vector<dgs_memory_range>;

// Prefetch the host pages covering these memory ranges.
void fDgsPrefetch(dgs_memory_ranges pMr);

size_t fDgsPathMax();
