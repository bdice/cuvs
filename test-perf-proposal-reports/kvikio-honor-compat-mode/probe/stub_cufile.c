/* Stand-in for libcufile.so.0 (LD_PRELOAD): records which cuFile entry points are called and
 * fails registration, so the probe never reaches the real cuFile driver or the GPU. */
#include <stdio.h>
typedef struct { int err; int cu_err; } CUfileError_t;
static CUfileError_t fail(const char* fn)
{
  fprintf(stderr, "[stub libcufile] %s called\n", fn);
  CUfileError_t e = {5001, 0}; /* CU_FILE_DRIVER_NOT_INITIALIZED */
  return e;
}
CUfileError_t cuFileGetVersion(int* v) { *v = 1180; CUfileError_t e = {0, 0}; return e; }
#define STUB(name) CUfileError_t name(void) { return fail(#name); }
STUB(cuFileHandleRegister) STUB(cuFileHandleDeregister) STUB(cuFileRead) STUB(cuFileWrite)
STUB(cuFileBufRegister) STUB(cuFileBufDeregister) STUB(cuFileDriverOpen) STUB(cuFileDriverClose) STUB(cuFileDriverClose_v2)
STUB(cuFileDriverGetProperties) STUB(cuFileDriverSetPollMode) STUB(cuFileDriverSetMaxCacheSize)
STUB(cuFileDriverSetMaxPinnedMemSize) STUB(cuFileBatchIOSetUp) STUB(cuFileBatchIOSubmit)
STUB(cuFileBatchIOGetStatus) STUB(cuFileBatchIOCancel) STUB(cuFileBatchIODestroy)
STUB(cuFileReadAsync) STUB(cuFileWriteAsync) STUB(cuFileStreamRegister) STUB(cuFileStreamDeregister)
