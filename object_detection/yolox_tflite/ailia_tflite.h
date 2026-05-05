/**
 * \~japanese
 * @file ailia_tflite.h
 * @brief Minimal subset of the ailia TFLite Runtime C API used by the
 *        yolox_tflite sample. The official header is not shipped with the
 *        ailia-tflite-cpp submodule (which contains only the runtime
 *        binaries), so this declarations file is vendored here so that the
 *        sample compiles against libailia_tflite.{so,dylib,dll}.
 *
 *        For the full API reference, see https://docs.ailia.ai .
 */

#ifndef INCLUDED_AILIA_TFLITE
#define INCLUDED_AILIA_TFLITE

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && !defined(_WIN64)
#define AILIA_TFLITE_API __stdcall
#else
#define AILIA_TFLITE_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* status */
#define AILIA_TFLITE_STATUS_SUCCESS                (0)
#define AILIA_TFLITE_STATUS_INVALID_ARGUMENT       (-1)
#define AILIA_TFLITE_STATUS_OUT_OF_RANGE           (-2)
#define AILIA_TFLITE_STATUS_MEMORY_INSUFFICIENT    (-3)
#define AILIA_TFLITE_STATUS_BROKEN_MODEL           (-4)
#define AILIA_TFLITE_STATUS_INVALID_PARAMETER      (-5)
#define AILIA_TFLITE_STATUS_PARAMETER_NOT_FOUND    (-6)
#define AILIA_TFLITE_STATUS_UNSUPPORTED_OPCODE     (-7)
#define AILIA_TFLITE_STATUS_LICENSE_NOT_FOUND      (-8)
#define AILIA_TFLITE_STATUS_LICENSE_BROKEN         (-9)
#define AILIA_TFLITE_STATUS_LICENSE_EXPIRED        (-10)
#define AILIA_TFLITE_STATUS_INVALID_STATE          (-11)
#define AILIA_TFLITE_STATUS_OTHER_ERROR            (-128)

typedef int32_t AILIATFLiteStatus;

/* environment */
#define AILIA_TFLITE_ENV_REFERENCE          (0)
#define AILIA_TFLITE_ENV_NNAPI              (1)
#define AILIA_TFLITE_ENV_MMALIB             (2)
#define AILIA_TFLITE_ENV_MMALIB_COMPATIBLE  (3)

typedef int32_t AILIATFLiteEnvironment;

/* memory mode */
#define AILIA_TFLITE_MEMORY_MODE_DEFAULT             (0)
#define AILIA_TFLITE_MEMORY_MODE_REDUCE_INTERSTAGE   (1)

/* flags */
#define AILIA_TFLITE_FLAG_NONE                                (0)
#define AILIA_TFLITE_FLAG_INPUT_AND_OUTPUT_TENSORS_USE_SCRATCH (1)

/* tensor types (matches TfLiteType) */
#define AILIA_TFLITE_TENSOR_TYPE_FLOAT32   (0)
#define AILIA_TFLITE_TENSOR_TYPE_FLOAT16   (1)
#define AILIA_TFLITE_TENSOR_TYPE_INT32     (2)
#define AILIA_TFLITE_TENSOR_TYPE_UINT8     (3)
#define AILIA_TFLITE_TENSOR_TYPE_INT64     (4)
#define AILIA_TFLITE_TENSOR_TYPE_STRING    (5)
#define AILIA_TFLITE_TENSOR_TYPE_BOOL      (6)
#define AILIA_TFLITE_TENSOR_TYPE_INT16     (7)
#define AILIA_TFLITE_TENSOR_TYPE_COMPLEX64 (8)
#define AILIA_TFLITE_TENSOR_TYPE_INT8      (9)

typedef int32_t AILIATFLiteTensorType;

struct AILIATFLiteInstance;

/* lifecycle */
AILIATFLiteStatus AILIA_TFLITE_API ailiaTFLiteCreate(
    struct AILIATFLiteInstance** instance,
    void* tflite, size_t tflite_length,
    void* license_data, void* unused1, void* unused2, void* unused3,
    int32_t env_id, int32_t memory_mode, int32_t flags);
void              AILIA_TFLITE_API ailiaTFLiteDestroy(struct AILIATFLiteInstance* instance);

AILIATFLiteStatus AILIA_TFLITE_API ailiaTFLiteAllocateTensors(struct AILIATFLiteInstance* instance);

AILIATFLiteStatus AILIA_TFLITE_API ailiaTFLitePredict(struct AILIATFLiteInstance* instance);

/* tensor info */
AILIATFLiteStatus AILIA_TFLITE_API ailiaTFLiteGetNumberOfInputs(struct AILIATFLiteInstance* instance, int32_t* count);
AILIATFLiteStatus AILIA_TFLITE_API ailiaTFLiteGetInputTensorIndex(struct AILIATFLiteInstance* instance, int32_t* tensor_index, int32_t input_index);
AILIATFLiteStatus AILIA_TFLITE_API ailiaTFLiteGetNumberOfOutputs(struct AILIATFLiteInstance* instance, int32_t* count);
AILIATFLiteStatus AILIA_TFLITE_API ailiaTFLiteGetOutputTensorIndex(struct AILIATFLiteInstance* instance, int32_t* tensor_index, int32_t output_index);

AILIATFLiteStatus AILIA_TFLITE_API ailiaTFLiteGetTensorDimension(struct AILIATFLiteInstance* instance, int32_t* dim, int32_t tensor_index);
AILIATFLiteStatus AILIA_TFLITE_API ailiaTFLiteGetTensorShape(struct AILIATFLiteInstance* instance, int32_t* shape, int32_t tensor_index);
AILIATFLiteStatus AILIA_TFLITE_API ailiaTFLiteGetTensorType(struct AILIATFLiteInstance* instance, AILIATFLiteTensorType* tensor_type, int32_t tensor_index);
AILIATFLiteStatus AILIA_TFLITE_API ailiaTFLiteGetTensorBuffer(struct AILIATFLiteInstance* instance, void** buffer, int32_t tensor_index);

/* quantization parameters */
AILIATFLiteStatus AILIA_TFLITE_API ailiaTFLiteGetTensorQuantizationCount(struct AILIATFLiteInstance* instance, int32_t* count, int32_t tensor_index);
AILIATFLiteStatus AILIA_TFLITE_API ailiaTFLiteGetTensorQuantizationScale(struct AILIATFLiteInstance* instance, float* scale, int32_t tensor_index);
AILIATFLiteStatus AILIA_TFLITE_API ailiaTFLiteGetTensorQuantizationZeroPoint(struct AILIATFLiteInstance* instance, int64_t* zero_point, int32_t tensor_index);
AILIATFLiteStatus AILIA_TFLITE_API ailiaTFLiteGetTensorQuantizationQuantizedDimension(struct AILIATFLiteInstance* instance, int32_t* axis, int32_t tensor_index);

/* device control */
AILIATFLiteStatus AILIA_TFLITE_API ailiaTFLiteGetDeviceCount(struct AILIATFLiteInstance* instance, size_t* device_count);
AILIATFLiteStatus AILIA_TFLITE_API ailiaTFLiteGetDeviceName(struct AILIATFLiteInstance* instance, int32_t device_index, char** name);
AILIATFLiteStatus AILIA_TFLITE_API ailiaTFLiteGetDeviceExtraInfo(struct AILIATFLiteInstance* instance, int32_t device_index, char** info);
AILIATFLiteStatus AILIA_TFLITE_API ailiaTFLiteSelectDevices(struct AILIATFLiteInstance* instance, const int32_t* device_indexes, size_t device_index_count);
AILIATFLiteStatus AILIA_TFLITE_API ailiaTFLiteGetSelectedDeviceIndexes(struct AILIATFLiteInstance* instance, int32_t* device_indexes, size_t* device_index_count);

/* error detail */
const char*       AILIA_TFLITE_API ailiaTFLiteGetErrorDetail(struct AILIATFLiteInstance* instance);

#ifdef __cplusplus
}
#endif

#endif  /* INCLUDED_AILIA_TFLITE */
