#include <iostream>
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/hexagon/hexagon_delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/testing/util.h"
#include "tensorflow/lite/tools/benchmark/benchmark_utils.h"
#include "tensorflow/lite/tools/logging.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

#define FATAL_ERRORS(errormsg) {fprintf(stderr, "FATAL ERROR (line %d): %s\n", __LINE__, (errormsg)); exit(1);}
#define FATAL_ERROR(errormsg) FATAL_ERRORS((errormsg).c_str())

ABSL_FLAG(std::string, data_folder_path, "/usr/share/tevr_asr_tool", "Path to the data folder.");

int main(int argc, char** argv) {
    ::tflite::LogToStderr();
    absl::ParseCommandLine(argc, argv);

    const std::string &data_folder_path = absl::GetFlag(FLAGS_data_folder_path);
    const std::string &model_path = data_folder_path + std::string("/acoustic_model.tflite");
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if(model == nullptr) FATAL_ERROR(std::string("Could not load ")+model_path);

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    builder(&interpreter);
    if(interpreter != nullptr) FATAL_ERRORS("Could not create interpreter for acoustic model.");
    if(interpreter->AllocateTensors() != kTfLiteOk) FATAL_ERRORS("Could not allocate tensors for acoustic model.");

    // T* input = interpreter->typed_input_tensor<T>(i);

    if(interpreter->Invoke() != kTfLiteOk) FATAL_ERRORS("Could not invoke acoustic model.");

    // T* output = interpreter->typed_output_tensor<T>(i);


}
