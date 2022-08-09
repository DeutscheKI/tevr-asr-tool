#include <stdint.h>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"


TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
    TF_LITE_ENSURE_EQ(context, tflite::NumInputs(node), 1);
    TF_LITE_ENSURE_EQ(context, tflite::NumOutputs(node), 1);
    const TfLiteTensor* input;
    TF_LITE_ENSURE_OK(context, tflite::GetInputSafe(context, node, 0, &input));
    TfLiteTensor* output;
    TF_LITE_ENSURE_OK(context, tflite::GetOutputSafe(context, node, 0, &output));
    TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);

    return context->ResizeTensor(context, output, TfLiteIntArrayCopy(input->dims));
}

inline void BatchErf(const tflite::RuntimeShape& input_shape, const float* input_data,
                     const tflite::RuntimeShape& output_shape, float* output_data) {
    //ruy::profiler::ScopeLabel label("Logistic");
    auto input_map = tflite::optimized_ops::MapAsVector(input_data, input_shape);
    auto output_map = tflite::optimized_ops::MapAsVector(output_data, output_shape);
    output_map.array() = input_map.array().unaryExpr(Eigen::internal::scalar_erf_op<float>());
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
    const TfLiteTensor* input;
    TF_LITE_ENSURE_OK(context, tflite::GetInputSafe(context, node, 0, &input));
    TfLiteTensor* output;
    TF_LITE_ENSURE_OK(context, tflite::GetOutputSafe(context, node, 0, &output));
    switch (input->type) {
        case kTfLiteFloat32: {
            BatchErf(   tflite::GetTensorShape(input), tflite::GetTensorData<float>(input),
                        tflite::GetTensorShape(output), tflite::GetTensorData<float>(output)    );
            break;
        }
        default:
            TF_LITE_KERNEL_LOG(context, "Only float32 is supported, got %s.", TfLiteTypeGetName(input->type));
            return kTfLiteError;
    }
    return kTfLiteOk;
}

TfLiteRegistration* Register_ERF() {
  static TfLiteRegistration r = {
      nullptr,
      nullptr,
      Prepare,
      Eval,
  };
  return &r;
}
