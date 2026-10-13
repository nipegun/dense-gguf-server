#include "dgs-io.h"

#include <vector>

void dgs_io_write_i::mWriteString(const std::string & str) {
  uint32_t vStrSize = str.size();

  write(&vStrSize, sizeof(vStrSize));
  write(str.data(), vStrSize);
}

void dgs_io_read_i::mReadString(std::string & str) {
  uint32_t vStrSize;
  read(&vStrSize, sizeof(vStrSize));

  std::vector<char> lBuf(vStrSize);
  read(lBuf.data(), vStrSize);

  str.assign(lBuf.data(), vStrSize);
}
