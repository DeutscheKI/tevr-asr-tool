#include "absl/flags/parse.h"
#include "absl/flags/flag.h"
#include "absl/flags/internal/commandlineflag.h"
#include "absl/flags/internal/private_handle_accessor.h"
#include "absl/flags/reflection.h"
#include "absl/flags/usage_config.h"
#include "absl/memory/memory.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/hexagon/hexagon_delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/testing/util.h"
#include "tensorflow/lite/tools/benchmark/benchmark_utils.h"
#include "tensorflow/lite/tools/logging.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "kenlm/lm/ngram_query.hh"
#include "wave/file.h"
#include "tensorflow/lite/minimal_logging.h"

#define FATAL_ERRORS(errormsg) {fprintf(stderr, "FATAL ERROR (line %d): %s\n", __LINE__, (errormsg)); exit(1);}
#define FATAL_ERROR(errormsg) FATAL_ERRORS((errormsg).c_str())

ABSL_FLAG(std::string, target_file, "INVALID_PATH", "Path to the 16kHz WAV to analyze.");
ABSL_FLAG(std::string, data_folder_path, "/usr/share/tevr_asr_tool", "Path to the data folder.");

const char* tokens[]= {"", "", " ", "chen", "sche", "lich", "isch", "icht", "iche", "eine", "rden", "tion", "urde", "haft", "eich", "rung",
                       "chte", "ssen", "chaf", "nder", "tlic", "tung", "eite", "iert", "sich", "ngen", "erde", "scha", "nden", "unge", "lung",
                       "mmen", "eren", "ende", "inde", "erun", "sten", "iese", "igen", "erte", "iner", "tsch", "keit", "der", "die", "ter",
                       "und", "ein", "ist", "den", "ten", "ber", "ver", "sch", "ung", "ste", "ent", "ach", "nte", "auf", "ben", "eit", "des",
                       "ers", "aus", "das", "von", "ren", "gen", "nen", "lle", "hre", "mit", "iel", "uch", "lte", "ann", "lie", "men", "dem",
                       "and", "ind", "als", "sta", "elt", "ges", "tte", "ern", "wir", "ell", "war", "ere", "rch", "abe", "len", "ige", "ied",
                       "ger", "nnt", "wei", "ele", "och", "sse", "end", "all", "ahr", "bei", "sie", "ede", "ion", "ieg", "ege", "auc", "che",
                       "rie", "eis", "vor", "her", "ang", "f\u00fcr", "ass", "uss", "tel", "er", "in", "ge", "en", "st", "ie", "an", "te",
                       "be", "re", "zu", "ar", "es", "ra", "al", "or", "ch", "et", "ei", "un", "le", "rt", "se", "is", "ha", "we", "at", "me",
                       "ne", "ur", "he", "au", "ro", "ti", "li", "ri", "eh", "im", "ma", "tr", "ig", "el", "um", "la", "am", "de", "so", "ol",
                       "tz", "il", "on", "it", "sc", "sp", "ko", "na", "pr", "ni", "si", "fe", "wi", "ns", "ke", "ut", "da", "gr", "eu", "mi",
                       "hr", "ze", "hi", "ta", "ss", "ng", "sa", "us", "ba", "ck", "em", "kt", "ka", "ve", "fr", "bi", "wa", "ah", "gt", "di",
                       "ab", "fo", "to", "rk", "as", "ag", "gi", "hn", "s", "t", "n", "m", "r", "l", "f", "e", "a", "b", "d", "h", "k", "g",
                       "o", "i", "u", "w", "p", "z", "\u00e4", "\u00fc", "v", "\u00f6", "j", "c", "y", "x", "q", "\u00e1", "\u00ed",
                       "\u014d", "\u00f3", "\u0161", "\u00e9", "\u010d", "?" };

TfLiteRegistration* Register_ERF();

int main(int argc, char** argv) {
    ::tflite::LogToStderr();
    absl::ParseCommandLine(argc, argv);

    const std::string &target_file = absl::GetFlag(FLAGS_target_file);
    const std::string &data_folder_path = absl::GetFlag(FLAGS_data_folder_path);

    TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO, "Loading WAV ...");

    wave::File wav_file;
    wave::Error wav_error = wav_file.Open(target_file.c_str(), wave::kIn);
    if ( wav_error == wave::kInvalidFormat) FATAL_ERROR(std::string("WAV has invalid format: ") + target_file);
    if (wav_error) FATAL_ERROR(std::string("Could not open WAV file: ") + target_file);

    if( wav_file.channel_number() != 1) FATAL_ERRORS("WAV file is not mono");
    if( wav_file.sample_rate() != 16000) FATAL_ERRORS("WAV file is not 16kHz");

    std::vector<float> wave_data;
    wav_error = wav_file.Read(&wave_data);
    if (wav_error) FATAL_ERROR(std::string("Could not read WAV file ") + target_file);

    int data_length = wave_data.size();

    for(int i=0;i<4;i++) {
        const std::string &model_path = data_folder_path + std::string("/acoustic_model_0")+std::to_string(i)+std::string(".tflite");
        TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO, "Loading %s ...", model_path.c_str());
        std::unique_ptr<tflite::FlatBufferModel> acoustic_model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
        if(acoustic_model == nullptr) FATAL_ERROR(std::string("Could not load ")+model_path);

        TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO, "Building interpreter for %s ...", model_path.c_str());
        std::unique_ptr<tflite::Interpreter> acoustic_interpreter;
        tflite::ops::builtin::BuiltinOpResolver resolver;
        //tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
        resolver.AddCustom("FlexErf", Register_ERF());
        tflite::InterpreterBuilder builder(*acoustic_model, resolver);
        builder(&acoustic_interpreter);
        if(acoustic_interpreter == nullptr) FATAL_ERROR(std::string("Could not create interpreter for acoustic model ")+std::to_string(i));

        if( i == 0) {
            if( acoustic_interpreter->ResizeInputTensorStrict(0, {1, data_length}) != kTfLiteOk ) FATAL_ERRORS("Could not resize audio input tensor.");
        } else {
            if( acoustic_interpreter->ResizeInputTensorStrict(0, {1, data_length, 1280}) != kTfLiteOk ) FATAL_ERRORS("Could not resize hidden state input tensor.");
        }
        if( acoustic_interpreter->AllocateTensors() != kTfLiteOk) FATAL_ERRORS("Could not allocate tensors for acoustic model.");

        float* audio_input = acoustic_interpreter->typed_input_tensor<float>(0);
        int copy_size = data_length * (i==0 ? 1 : 1280);
        for(int t=0;t<copy_size;t++) audio_input[t] = wave_data[t];

        TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO, "Invoking %s ...", model_path.c_str());
        if( acoustic_interpreter->Invoke() != kTfLiteOk) FATAL_ERRORS("Could not invoke acoustic model.");

        TfLiteTensor* output = acoustic_interpreter->output_tensor(0);
        if(output->dims->size != 3) FATAL_ERRORS("Wrong output dims");
        if(output->dims->data[0] != 1) FATAL_ERRORS("Wrong output dim 0");
        int expected_dim2 = i==3 ? 256 : 1280;
        if(output->dims->data[2] != expected_dim2) FATAL_ERRORS("Wrong output dim 2");

        data_length = output->dims->data[1];
        wave_data.clear();
        float* logit_output = acoustic_interpreter->typed_output_tensor<float>(0);
        copy_size = data_length * expected_dim2;
        for(int t=0;t<copy_size;t++) wave_data.emplace_back(logit_output[t]);
    }

    lm::ngram::Config config;
    const std::string &lm_path = data_folder_path + std::string("/language_model.bin");
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO, "Loading language model %s ...", lm_path.c_str());
    lm::ngram::TrieModel language_model(lm_path.c_str(), config);

    int token_idx = language_model.GetVocabulary().Index(StringPiece("mückenstiche"));
    if(token_idx != 68501) FATAL_ERRORS("Language model vocabulary is wrong.");

    int last_token = 0;
    for(int t=0;t<data_length;t++) {
        int max_idx = 0;
        double max_logit = -5e17;
        for(int i=0;i<256;i++) {
            float const& v = wave_data[t * 256 + i];
            if(v > max_logit) {
                max_logit = v;
                max_idx = i;
            }
        }
        if(max_idx != last_token) {
            last_token = max_idx;
            std::cout << tokens[max_idx];
        }
    }

}
