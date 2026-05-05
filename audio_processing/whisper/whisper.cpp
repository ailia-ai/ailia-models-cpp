/*******************************************************************
*
*    DESCRIPTION:
*      Whisper with ailia Speech Sample Program
*    AUTHOR:
*      ax Inc.
*    DATE:2024/05/22
*
*******************************************************************/

#include <stdio.h>
#include <math.h>
#include <vector>
#include <string>

#include "ailia.h"
#include "ailia_audio.h"

#include "ailia_speech.h"
#include "ailia_speech_util.h"

#include "wave_reader.h"

// ======================
// Parameters
// ======================

#if defined(_WIN32) || defined(_WIN64)
#define PRINT_OUT(...) fprintf_s(stdout, __VA_ARGS__)
#define PRINT_ERR(...) fprintf_s(stderr, __VA_ARGS__)
#else
#define PRINT_OUT(...) fprintf(stdout, __VA_ARGS__)
#define PRINT_ERR(...) fprintf(stderr, __VA_ARGS__)
#endif

static int args_env_id = -1;

std::string input_file = "demo.wav";

// ======================
// Arguemnt Parser
// ======================

static void print_usage()
{
	PRINT_OUT("usage: whisper [-h] [-i FILE] [-b] [-e ENV_ID]\n");
	return;
}


static void print_help()
{
	PRINT_OUT("\n");
	PRINT_OUT("whisper model\n");
	PRINT_OUT("\n");
	PRINT_OUT("optional arguments:\n");
	PRINT_OUT("  -h, --help            show this help message and exit\n");
	PRINT_OUT("  -i FILE, --input FILE\n");
	PRINT_OUT("                        The input file.\n");
	PRINT_OUT("  -e ENV_ID, --env_id ENV_ID\n");
	PRINT_OUT("                        The backend environment id.\n");
	return;
}


static void print_error(std::string arg)
{
	PRINT_ERR("fugumt: error: unrecognized arguments: %s\n", arg.c_str());
	return;
}


static int argument_parser(int argc, char **argv)
{
	int status = 0;

	for (int i = 1; i < argc; i++) {
		std::string arg = argv[i];
		if (status == 0) {
			if (arg == "-i" || arg == "--input") {
				status = 1;
			}
			else if (arg == "-h" || arg == "--help") {
				print_usage();
				print_help();
				return -1;
			}
			else if (arg == "-e" || arg == "--env_id") {
				status = 4;
			}
			else {
				print_usage();
				print_error(arg);
				return -1;
			}
		}
		else if (arg[0] != '-') {
			switch (status) {
			case 1:
				input_file = arg;
				break;
			case 4:
				args_env_id = atoi(arg.c_str());
				break;
			default:
				print_usage();
				print_error(arg);
				return -1;
			}
			status = 0;
		}
		else {
			print_usage();
			print_error(arg);
			return -1;
		}
	}

	return AILIA_STATUS_SUCCESS;
}

// ======================
// Main functions
// ======================

int intermediate_callback(void *handle, const char *text){
	printf("\r%s", text);
	fflush(stdout);
	return 0; // 1で中断
}

int get_text(struct AILIASpeech* net, bool live_mode){
	unsigned int count = 0;
	int status = ailiaSpeechGetTextCount(net, &count);
	if (status != AILIA_STATUS_SUCCESS){
		printf("ailiaSpeechGetTextCount Error %d\n", status);
		printf("%s\n", ailiaSpeechGetErrorDetail(net));
		return -1;
	}

	if (live_mode && count > 0){
		printf("\n");
	}

	for (unsigned int idx = 0; idx < count; idx++){
		AILIASpeechText text;
		status = ailiaSpeechGetText(net, &text, AILIA_SPEECH_TEXT_VERSION, idx);
		if (status != AILIA_STATUS_SUCCESS){
			printf("ailiaSpeechGetText Error %d\n", status);
			printf("%s\n", ailiaSpeechGetErrorDetail(net));
			return -1;
		}

		float cur_time = text.time_stamp_begin;
		float next_time = text.time_stamp_end;
		printf("[%02d:%02d.%03d --> %02d:%02d.%03d] ", (int)cur_time/60%60,(int)cur_time%60, (int)(cur_time*1000)%1000, (int)next_time/60%60,(int)next_time%60, (int)(next_time*1000)%1000);
		printf("[%0.4f] ", text.confidence);
		printf("%s\n", text.text);
	}

	return AILIA_STATUS_SUCCESS;
}

int update(struct AILIASpeech* net, const float *wave_buf, int nSamples, int nChannels, int sampleRate, int &push_i, unsigned int &complete, bool translate, bool live_mode){
	int status;

	// Push pcm input to queue
	int push_size = sampleRate;
	int push_samples = std::max(0, std::min(nSamples - push_i, push_size));
	if (push_samples >= 1){
		// Push pcm data
		status = ailiaSpeechPushInputData(net, &wave_buf[nChannels * push_i], nChannels, push_samples, sampleRate);
		if (status != AILIA_STATUS_SUCCESS){
			printf("ailiaSpeechPushInputData Error %d\n", status);
			printf("%s\n", ailiaSpeechGetErrorDetail(net));
			return -1;
		}
		push_i += push_size;
	}else{
		// Finalize push pcm data
		status = ailiaSpeechFinalizeInputData(net);
		if (status != AILIA_STATUS_SUCCESS){
			printf("ailiaSpeechFinalizeInputData Error %d\n", status);
			printf("%s\n", ailiaSpeechGetErrorDetail(net));
			return -1;
		}
	}
	
	// Transcribe
	while(true){
		// Check enough pcm exists in queue
		unsigned int buffered = 0;
		status = ailiaSpeechBuffered(net, &buffered);
		if (status != AILIA_STATUS_SUCCESS){
			printf("ailiaSpeechIsBuffered Error %d\n", status);
			printf("%s\n", ailiaSpeechGetErrorDetail(net));
			return -1;
		}
		if (buffered == 0){
			break;	// Wait next data
		}

		// Process
		status = ailiaSpeechTranscribe(net);
		if (status != AILIA_STATUS_SUCCESS){
			printf("ailiaSpeechTranscribe Error %d\n", status);
		}
		if (status != AILIA_STATUS_SUCCESS){
			printf("%s\n", ailiaSpeechGetErrorDetail(net));
			if (status == AILIA_STATUS_LICENSE_NOT_FOUND){
				printf("License file not found.\n");
				printf("Please place license file.\n");
			}
			return -1;
		}
		
		// Get results
		status = get_text(net, live_mode);
		if (status != AILIA_STATUS_SUCCESS){
			return status;
		}
	}

	// Check all queued data processed
	if (push_samples == 0){
		status = ailiaSpeechComplete(net, &complete);
		if (status != AILIA_STATUS_SUCCESS){
			printf("ailiaSpeechIsBuffered Error %d\n", status);
			printf("%s\n", ailiaSpeechGetErrorDetail(net));
			return -1;
		}
	}

	return AILIA_STATUS_SUCCESS;
}

int get_model_name(std::string &encoder, std::string &decoder, int &model_id, const char *model_type){
	bool model_detected = false;
	if (strcmp(model_type,"large") == 0 || strcmp(model_type,"large_v3") == 0){
		encoder = std::string("encoder_"+std::string(model_type)+".onnx");
		decoder = std::string("decoder_"+std::string(model_type)+"_fix_kv_cache.onnx");
	}else{
		encoder = std::string("encoder_"+std::string(model_type)+".opt3.onnx");
		decoder = std::string("decoder_"+std::string(model_type)+"_fix_kv_cache.opt3.onnx");
	}
	if (strcmp(model_type,"tiny") == 0){
		model_id = AILIA_SPEECH_MODEL_TYPE_WHISPER_MULTILINGUAL_TINY;
		model_detected = true;
	}
	if (strcmp(model_type,"base") == 0){
		model_id = AILIA_SPEECH_MODEL_TYPE_WHISPER_MULTILINGUAL_BASE;
		model_detected = true;
	}
	if (strcmp(model_type,"small") == 0){
		model_id = AILIA_SPEECH_MODEL_TYPE_WHISPER_MULTILINGUAL_SMALL;
		model_detected = true;
	}
	if (strcmp(model_type,"medium") == 0){
		model_id = AILIA_SPEECH_MODEL_TYPE_WHISPER_MULTILINGUAL_MEDIUM;
		model_detected = true;
	}
	if (strcmp(model_type,"large") == 0){
		model_id = AILIA_SPEECH_MODEL_TYPE_WHISPER_MULTILINGUAL_LARGE;
		model_detected = true;
	}
	if (strcmp(model_type,"large_v3") == 0){
		model_id = AILIA_SPEECH_MODEL_TYPE_WHISPER_MULTILINGUAL_LARGE_V3;
		model_detected = true;
	}
	if (model_detected == false){
		return -1;
	}
	return AILIA_STATUS_SUCCESS;
}

int main(int argc, char **argv){
	int status = argument_parser(argc, argv);
	if (status != AILIA_STATUS_SUCCESS) {
		return -1;
	}

	const char *input_path="./demo.wav";
	const char *model_type="small";
	const char *language="auto";
	
	// Load wave file
	int sampleRate,nChannels,nSamples;
	std::vector<float> wave_buf = read_wave_file(input_path, &sampleRate, &nChannels, &nSamples);
	if(wave_buf.size()==0){
		printf("wav file not found or could not open %s\n", input_path);
		return -1;
	}
	printf("Input wave sec %f\n", (float)nSamples/sampleRate);

	struct AILIASpeech* net;

	AILIASpeechApiCallback callback = ailiaSpeechUtilGetCallback();

	bool translate = true;
	bool live_mode = false;
	
	int task_id = (translate) ? AILIA_SPEECH_TASK_TRANSLATE:AILIA_SPEECH_TASK_TRANSCRIBE;
	int flag = (live_mode) ? AILIA_SPEECH_FLAG_LIVE:AILIA_SPEECH_FLAG_NONE;
	int memory_mode = AILIA_MEMORY_REDUCE_CONSTANT | AILIA_MEMORY_REDUCE_CONSTANT_WITH_INPUT_INITIALIZER | AILIA_MEMORY_REUSE_INTERSTAGE;

	status = ailiaSpeechCreate(&net, AILIA_ENVIRONMENT_ID_AUTO, AILIA_MULTITHREAD_AUTO, memory_mode, task_id, flag, callback, AILIA_SPEECH_API_CALLBACK_VERSION);
	if (status != AILIA_STATUS_SUCCESS){
		printf("ailiaSpeechCreate Error %d\n", status);
		printf("%s\n", ailiaSpeechGetErrorDetail(net));
		return -1;
	}

	std::string encoder;
	std::string decoder;
	int model_id;
	status = get_model_name(encoder, decoder, model_id, model_type);
	if (status != AILIA_STATUS_SUCCESS){
		printf("unknown model type\n");
		ailiaSpeechDestroy(net);
		return -1;
	}

	status = ailiaSpeechOpenModelFileA(net, encoder.c_str(), decoder.c_str(), model_id);
	if (status != AILIA_STATUS_SUCCESS){
		printf("ailiaSpeechOpenModelFileA Error %d\n", status);
		printf("required file : %s and %s\n", encoder.c_str(), decoder.c_str());
		printf("%s\n", ailiaSpeechGetErrorDetail(net));
		if (status == AILIA_STATUS_LICENSE_NOT_FOUND){
			printf("License file not found.\n");
			printf("Please place license file.\n");
		}
		return -1;
	}

	status = ailiaSpeechSetLanguage(net, language);
	if (status != AILIA_STATUS_SUCCESS){
		printf("ailiaSpeechSetLanguage Error %d\n", status);
		printf("%s\n", ailiaSpeechGetErrorDetail(net));
		return -1;
	}

	if (live_mode){ // You can also use setIntermediateCallback for normal mode
		status = ailiaSpeechSetIntermediateCallback(net, &intermediate_callback, NULL);
		if (status != AILIA_STATUS_SUCCESS){
			printf("ailiaSpeechSetIntermediateCallback Error %d\n", status);
			printf("%s\n", ailiaSpeechGetErrorDetail(net));
			return -1;
		}
	}

	bool vad_enable = false;
	if (vad_enable){
		status = ailiaSpeechOpenVadFileA(net, "silero_vad.onnx", AILIA_SPEECH_VAD_TYPE_SILERO);
		if (status != AILIA_STATUS_SUCCESS){
			printf("ailiaSpeechOpenVadFileA Error %d\n", status);
		}
	}

	int push_i = 0;
	while(true){
		unsigned int complete = 0;
		status = update(net, &wave_buf[0], nSamples, nChannels, sampleRate, push_i, complete, translate, live_mode);
		if (status != AILIA_STATUS_SUCCESS){
			return -1;
		}
		if (complete == 1){
			break;
		}
	}

	ailiaSpeechDestroy(net);
	return 0;
}


