// CPU-only probe: open a file the way libcuvs does for device I/O and report whether KvikIO picked
// POSIX I/O and whether libcufile was ever loaded into the process.
#include "util/kvikio_io.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

static bool mapped(const char* name)
{
  std::ifstream maps("/proc/self/maps");
  std::string line;
  while (std::getline(maps, line)) {
    if (line.find(name) != std::string::npos) { return true; }
  }
  return false;
}

int main(int argc, char** argv)
{
  const std::string path = argc > 1 ? argv[1] : "/tmp/kvikio_compat_probe.bin";
  {
    std::ofstream f(path);
    f << "hello";
  }
  auto handle = cuvs::util::detail::open_kvikio_file_for_device_io(path, "r");
  std::printf("defaults::compat_mode=%d handle.compat_mode_requested=%d is_compat_mode_preferred=%d "
              "libcufile_mapped=%d libcuda_mapped=%d\n",
              static_cast<int>(kvikio::defaults::compat_mode()),
              static_cast<int>(handle.get_compat_mode_manager().compat_mode_requested()),
              static_cast<int>(handle.get_compat_mode_manager().is_compat_mode_preferred()),
              static_cast<int>(mapped("libcufile")),
              static_cast<int>(mapped("libcuda.so")));
  return 0;
}
