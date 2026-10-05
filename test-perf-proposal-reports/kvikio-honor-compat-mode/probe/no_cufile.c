/* LD_PRELOAD interposer: make dlopen("libcufile.so.0") fail, as on a system without cuFile. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <string.h>
void* dlopen(const char* file, int mode)
{
  static void* (*real)(const char*, int);
  if (!real) { real = (void* (*)(const char*, int))dlsym(RTLD_NEXT, "dlopen"); }
  if (file && strstr(file, "libcufile")) { return real("/nonexistent/libcufile.so.0", mode); }
  return real(file, mode);
}
