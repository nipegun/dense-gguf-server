#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

struct ggml_tensor;

class dgs_io_write_i {
public:
  dgs_io_write_i()          = default;
  virtual ~dgs_io_write_i() = default;

  virtual void write(const void * pSrc, size_t size)                            = 0;
  virtual void mWriteTensor(ggml_tensor * pTensor, size_t pOffset, size_t size) = 0;

  // bytes written so far
  virtual size_t mNBytes() = 0;

  void mWriteString(const std::string & str);
};

class dgs_io_read_i {
public:
  dgs_io_read_i()          = default;
  virtual ~dgs_io_read_i() = default;

  virtual void read(void * pDst, size_t size)                                  = 0;
  virtual void mReadTensor(ggml_tensor * pTensor, size_t pOffset, size_t size) = 0;

  // drop tensor data that has been read but not yet applied (e.g. when a restore fails)
  virtual void mDiscard() {}

  // bytes read so far
  virtual size_t mNBytes() = 0;

  void mReadString(std::string & str);
};
