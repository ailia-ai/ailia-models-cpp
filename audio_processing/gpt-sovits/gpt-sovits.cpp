/*******************************************************************
*
*    DESCRIPTION:
*      AILIA GPT-SoVits sample
*    AUTHOR:
*
*    DATE:2024/04/30
*
*******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <string>
#include <math.h>
#include <chrono>

#undef UNICODE

#include "ailia.h"
#include "ailia_audio.h"
#include "wave_reader.h"
#include "wave_writer.h"

bool debug = false;
bool debug_token = false;


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

#define BENCHMARK_ITERS 5

#define MODEL_N 5

#define MODEL_SSL 0
#define MODEL_ENCODER 1
#define MODEL_FS_DECODER 2
#define MODEL_DECODER 3
#define MODEL_VITS 4

const char *MODEL_NAME[5] = {"cnhubert.onnx", "t2s_encoder.onnx", "t2s_fsdec.onnx", "t2s_sdec.onnx", "vits.onnx"};

static bool benchmark  = false;
static int args_env_id = -1;

std::string reference_wave = "reference_audio_captured_by_ax.wav";

#define REF_PHONES_SIZE 37
#define TEXT_PHONES_SIZE 72

const char * REF_PHONES[REF_PHONES_SIZE] = {"m", "i", "z", "u", "o", "m", "a", "r", "e", "e", "sh", "i", "a", "k", "a", "r", "a", "k", "a", "w", "a", "n", "a", "k", "U", "t", "e", "w", "a", "n", "a", "r", "a", "n", "a", "i", "."};
const char *TEXT_PHONES[TEXT_PHONES_SIZE] = {"e", "i", "e", "cl", "k", "U", "s", "u", "k", "a", "b", "u", "sh", "I", "k", "i", "g", "a", "i", "sh", "a", "d", "e", "w", "a", "e", "e", "a", "i", "n", "o", "j", "i", "ts", "u", "y", "o", "o", "k", "a", "n", "o", "t", "a", "m", "e", "n", "o", "g", "i", "j", "u", "ts", "u", "o", "k", "a", "i", "h", "a", "ts", "u", "sh", "I", "t", "e", "i", "m", "a", "s", "U", "."};

#define SYMBOLS_N 322
const char *SYMBOLS[SYMBOLS_N] = {"!", ",", "-", ".", "?", "AA", "AA0", "AA1", "AA2", "AE0", "AE1", "AE2", "AH0", "AH1", "AH2", "AO0", "AO1", "AO2", "AW0", "AW1", "AW2", "AY0", "AY1", "AY2", "B", "CH", "D", "DH", "E1", "E2", "E3", "E4", "E5", "EE", "EH0", "EH1", "EH2", "ER", "ER0", "ER1", "ER2", "EY0", "EY1", "EY2", "En1", "En2", "En3", "En4", "En5", "F", "G", "HH", "I", "IH", "IH0", "IH1", "IH2", "IY0", "IY1", "IY2", "JH", "K", "L", "M", "N", "NG", "OO", "OW0", "OW1", "OW2", "OY0", "OY1", "OY2", "P", "R", "S", "SH", "SP", "SP2", "SP3", "T", "TH", "U", "UH0", "UH1", "UH2", "UNK", "UW0", "UW1", "UW2", "V", "W", "Y", "Z", "ZH", "_", "a", "a1", "a2", "a3", "a4", "a5", "ai1", "ai2", "ai3", "ai4", "ai5", "an1", "an2", "an3", "an4", "an5", "ang1", "ang2", "ang3", "ang4", "ang5", "ao1", "ao2", "ao3", "ao4", "ao5", "b", "by", "c", "ch", "cl", "d", "dy", "e", "e1", "e2", "e3", "e4", "e5", "ei1", "ei2", "ei3", "ei4", "ei5", "en1", "en2", "en3", "en4", "en5", "eng1", "eng2", "eng3", "eng4", "eng5", "er1", "er2", "er3", "er4", "er5", "f", "g", "gy", "h", "hy", "i", "i01", "i02", "i03", "i04", "i05", "i1", "i2", "i3", "i4", "i5", "ia1", "ia2", "ia3", "ia4", "ia5", "ian1", "ian2", "ian3", "ian4", "ian5", "iang1", "iang2", "iang3", "iang4", "iang5", "iao1", "iao2", "iao3", "iao4", "iao5", "ie1", "ie2", "ie3", "ie4", "ie5", "in1", "in2", "in3", "in4", "in5", "ing1", "ing2", "ing3", "ing4", "ing5", "iong1", "iong2", "iong3", "iong4", "iong5", "ir1", "ir2", "ir3", "ir4", "ir5", "iu1", "iu2", "iu3", "iu4", "iu5", "j", "k", "ky", "l", "m", "my", "n", "ny", "o", "o1", "o2", "o3", "o4", "o5", "ong1", "ong2", "ong3", "ong4", "ong5", "ou1", "ou2", "ou3", "ou4", "ou5", "p", "py", "q", "r", "ry", "s", "sh", "t", "ts", "u", "u1", "u2", "u3", "u4", "u5", "ua1", "ua2", "ua3", "ua4", "ua5", "uai1", "uai2", "uai3", "uai4", "uai5", "uan1", "uan2", "uan3", "uan4", "uan5", "uang1", "uang2", "uang3", "uang4", "uang5", "ui1", "ui2", "ui3", "ui4", "ui5", "un1", "un2", "un3", "un4", "un5", "uo1", "uo2", "uo3", "uo4", "uo5", "v", "v1", "v2", "v3", "v4", "v5", "van1", "van2", "van3", "van4", "van5", "ve1", "ve2", "ve3", "ve4", "ve5", "vn1", "vn2", "vn3", "vn4", "vn5", "w", "x", "y", "z", "zh", "…"};


// ======================
// Arguemnt Parser
// ======================

static void print_usage()
{
	PRINT_OUT("usage: gpt-sovits [-h] [-i TEXT] [-b] [-e ENV_ID]\n");
	return;
}


static void print_help()
{
	PRINT_OUT("\n");
	PRINT_OUT("gpt-sovits model\n");
	PRINT_OUT("\n");
	PRINT_OUT("optional arguments:\n");
	PRINT_OUT("  -h, --help            show this help message and exit\n");
	PRINT_OUT("  -i FILE, --input FILE\n");
	PRINT_OUT("                        The input file.\n");
	PRINT_OUT("  -b, --benchmark       Running the inference on the same input 5 times to\n");
	PRINT_OUT("                        measure execution performance. (Cannot be used in\n");
	PRINT_OUT("                        video mode)\n");
	PRINT_OUT("  -e ENV_ID, --env_id ENV_ID\n");
	PRINT_OUT("                        The backend environment id.\n");
	return;
}


static void print_error(std::string arg)
{
	PRINT_ERR("gpt-sovits: error: unrecognized arguments: %s\n", arg.c_str());
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
			else if (arg == "-b" || arg == "--benchmark") {
				benchmark = true;
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
				reference_wave = arg;
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

void setErrorDetail(const char *func, const char *detail){
	PRINT_ERR("Error %s Detail %s\n", func, detail);
	throw(func);
}

struct AILIATensor{
	std::vector<float> data;
	AILIAShape shape;
};

void forward(AILIANetwork *ailia, std::vector<AILIATensor*> &inputs, std::vector<AILIATensor> &outputs){
	int status;

	unsigned int input_blob_cnt;
	status = ailiaGetInputBlobCount(ailia, &input_blob_cnt);
	if (status != AILIA_STATUS_SUCCESS) {
		setErrorDetail("ailiaGetInputBlobCount",ailiaGetErrorDetail(ailia));
	}

	if (input_blob_cnt != inputs.size()){
		setErrorDetail("input blob cnt and input tensor size must be same", "");
	}

	for (size_t i = 0; i < inputs.size(); i++){
		unsigned int input_blob_idx = 0;
		status = ailiaGetBlobIndexByInputIndex(ailia, &input_blob_idx, (unsigned int)i);
		if (status != AILIA_STATUS_SUCCESS) {
			setErrorDetail("ailiaGetBlobIndexByInputIndex", ailiaGetErrorDetail(ailia));
		}

		if (debug){
			PRINT_OUT("input blob shape %d %d %d %d dims %d\n",inputs[i]->shape.x,inputs[i]->shape.y,inputs[i]->shape.z,inputs[i]->shape.w,inputs[i]->shape.dim);
		}

		status = ailiaSetInputBlobShape(ailia,&inputs[i]->shape,input_blob_idx,AILIA_SHAPE_VERSION);
		if(status!=AILIA_STATUS_SUCCESS){
			setErrorDetail("ailiaSetInputBlobShape",ailiaGetErrorDetail(ailia));
		}

		status = ailiaSetInputBlobData(ailia, &(inputs[i]->data)[0], (unsigned int)(inputs[i]->data.size() * sizeof(float)), input_blob_idx);
		if (status != AILIA_STATUS_SUCCESS) {
			setErrorDetail("ailiaSetInputBlobData",ailiaGetErrorDetail(ailia));
		}
	}

	status = ailiaUpdate(ailia);
	if (status != AILIA_STATUS_SUCCESS) {
		setErrorDetail("ailiaUpdate",ailiaGetErrorDetail(ailia));
	}

	unsigned int output_blob_cnt;
	status = ailiaGetOutputBlobCount(ailia, &output_blob_cnt);
	if (status != AILIA_STATUS_SUCCESS) {
		setErrorDetail("ailiaGetOutputBlobCount",ailiaGetErrorDetail(ailia));
	}

	for (size_t i = 0; i < output_blob_cnt; i++){
		unsigned int output_blob_idx = 0;
		status = ailiaGetBlobIndexByOutputIndex(ailia, &output_blob_idx, (unsigned int)i);
		if (status != AILIA_STATUS_SUCCESS) {
			setErrorDetail("ailiaGetBlobIndexByInputIndex",ailiaGetErrorDetail(ailia));
		}

		AILIAShape output_blob_shape;
		status=ailiaGetBlobShape(ailia,&output_blob_shape,output_blob_idx,AILIA_SHAPE_VERSION);
		if(status!=AILIA_STATUS_SUCCESS){
			setErrorDetail("ailiaGetBlobShape", ailiaGetErrorDetail(ailia));
		}

		if (debug){
			PRINT_OUT("output_blob_shape %d %d %d %d dims %d\n",output_blob_shape.x,output_blob_shape.y,output_blob_shape.z,output_blob_shape.w,output_blob_shape.dim);
		}

		if (outputs.size() <= i){
			AILIATensor tensor;
			outputs.push_back(tensor);
		}
		
		AILIATensor &ref_tensor = outputs[i];
		int new_shape = output_blob_shape.x*output_blob_shape.y*output_blob_shape.z*output_blob_shape.w;
		if ((size_t)new_shape != ref_tensor.data.size()){
			ref_tensor.data.resize(new_shape);
		}
		ref_tensor.shape = output_blob_shape;

		status = ailiaGetBlobData(ailia, &ref_tensor.data[0], (unsigned int)(ref_tensor.data.size() * sizeof(float)), output_blob_idx);
		if (status != AILIA_STATUS_SUCCESS) {
			setErrorDetail("ailiaGetBlobData",ailiaGetErrorDetail(ailia));
		}
	}
}

static std::vector<float> resample(std::vector<float> pcm, int targetSampleRate, int sampleRate, int nChannels)
{
	if (nChannels == 2){
		for (size_t i = 0; i < pcm.size() / 2; i++){
			pcm[i] = (pcm[i*2] + pcm[i*2+1])/2;
		}
		pcm.resize(pcm.size() / 2);
	}

	if(sampleRate != targetSampleRate){
		int dst_n = 0;
		int status = ailiaAudioGetResampleLen(&dst_n, targetSampleRate, (int)pcm.size(), sampleRate);
		if (status != AILIA_STATUS_SUCCESS) {
			PRINT_ERR("ailiaAudioGetResampleLen failed %d\n", status);
			throw;
		}
		std::vector<float> new_audio_waveform(dst_n);
		status = ailiaAudioResample(&new_audio_waveform[0], &pcm[0], targetSampleRate, dst_n, sampleRate, (int)pcm.size());
		if (status != AILIA_STATUS_SUCCESS) {
			PRINT_ERR("ailiaAudioResample failed %d\n", status);
			throw;
		}
		return new_audio_waveform;
	}

	return pcm;
}

static std::vector<float> cleaned_text_to_sequence(const char ** data, int size){
	std::vector<float> sequence;
	for (int i = 0; i < size; i++){
		for (int s = 0; s < SYMBOLS_N; s++){
			if (strcmp(data[i], SYMBOLS[s]) == 0){
				if (debug_token){
					PRINT_OUT("%d ", s);
				}
				sequence.push_back((float)s);
				break;
			}
		}
	}
	if (debug_token){
		PRINT_OUT("\n");
	}
	return sequence;
}

static AILIATensor ssl_forward(std::vector<float> ref_audio_16k, AILIANetwork* net)
{
	std::vector<AILIATensor*> inputs;
	AILIATensor tensor;
	tensor.data = ref_audio_16k;
	tensor.shape.x = (unsigned int)ref_audio_16k.size();
	tensor.shape.y = 1;
	tensor.shape.z = 1;
	tensor.shape.w = 1;
	tensor.shape.dim = 2;
	inputs.push_back(&tensor);
	std::vector<AILIATensor> outputs;
	forward(net, inputs, outputs);
	return outputs[0];
}

int argmax(AILIATensor logits){
	float max_p = 0.0f;
	int max_i = 0;
	for (size_t i = 0; i < logits.data.size(); i++){
		if (logits.data[i] > max_p){
			max_p = logits.data[i];
			max_i = (int)i;
		}
	}
	return max_i;
}

static AILIATensor t2s_forward(AILIATensor ref_seq, AILIATensor text_seq, AILIATensor ref_bert, AILIATensor text_bert, AILIATensor ssl_content, AILIANetwork *net[MODEL_N]){
	int hz = 50;
	int max_sec = 54;
	int early_stop_num = hz * max_sec;

	std::vector<AILIATensor*> encoder_inputs;

	encoder_inputs.push_back(&ref_seq);
	encoder_inputs.push_back(&text_seq);
	encoder_inputs.push_back(&ref_bert);
	encoder_inputs.push_back(&text_bert);
	encoder_inputs.push_back(&ssl_content);

	if (debug_token){
		PRINT_OUT("encoder\n");
	}

	std::vector<AILIATensor> encoder_outputs;
	forward(net[MODEL_ENCODER], encoder_inputs, encoder_outputs);

	AILIATensor top_k;
	top_k.data = std::vector<float>(1);
	top_k.data[0] = 5;
	top_k.shape.x = 1;
	top_k.shape.y = 1;
	top_k.shape.z = 1;
	top_k.shape.w = 1;
	top_k.shape.dim = 1;

	AILIATensor top_p;
	top_p.data = std::vector<float>(1);
	top_p.data[0] = 1.0;
	top_p.shape.x = 1;
	top_p.shape.y = 1;
	top_p.shape.z = 1;
	top_p.shape.w = 1;
	top_p.shape.dim = 1;

	AILIATensor temperature;
	temperature.data = std::vector<float>(1);
	temperature.data[0] = 1.0;
	temperature.shape.x = 1;
	temperature.shape.y = 1;
	temperature.shape.z = 1;
	temperature.shape.w = 1;
	temperature.shape.dim = 1;

	AILIATensor repetition_penalty;
	repetition_penalty.data = std::vector<float>(1);
	repetition_penalty.data[0] = 1.35f;
	repetition_penalty.shape.x = 1;
	repetition_penalty.shape.y = 1;
	repetition_penalty.shape.z = 1;
	repetition_penalty.shape.w = 1;
	repetition_penalty.shape.dim = 1;

	std::vector<AILIATensor*> fs_decoder_inputs;
	fs_decoder_inputs.push_back(&encoder_outputs[0]); // x
	fs_decoder_inputs.push_back(&encoder_outputs[1]); // prompts
	fs_decoder_inputs.push_back(&top_k);
	fs_decoder_inputs.push_back(&top_p);
	fs_decoder_inputs.push_back(&temperature);
	fs_decoder_inputs.push_back(&repetition_penalty);

	int prefix_len = encoder_outputs[1].shape.x;

	if (debug_token){
		PRINT_OUT("fs_decoder\n");
	}

	std::vector<AILIATensor> fs_decoder_outputs;
	forward(net[MODEL_FS_DECODER], fs_decoder_inputs, fs_decoder_outputs);

	std::vector<AILIATensor*> decoder_inputs;
	decoder_inputs.push_back(&fs_decoder_outputs[0]); // y
	decoder_inputs.push_back(&fs_decoder_outputs[1]); // k
	decoder_inputs.push_back(&fs_decoder_outputs[2]); // v
	decoder_inputs.push_back(&fs_decoder_outputs[3]); // y_emb
	decoder_inputs.push_back(&fs_decoder_outputs[4]); // x_example
	decoder_inputs.push_back(&top_k);
	decoder_inputs.push_back(&top_p);
	decoder_inputs.push_back(&temperature);
	decoder_inputs.push_back(&repetition_penalty);

	std::vector<AILIATensor> decoder_outputs;

	const int EOS = 1024;
	int idx = 1;
	AILIATensor y = fs_decoder_outputs[0]; // output
	for (; idx < 1500; idx++){
		if (debug_token){
			PRINT_OUT("decoder step %d ", idx);
		}

		auto start2 = std::chrono::high_resolution_clock::now();
		forward(net[MODEL_DECODER], decoder_inputs, decoder_outputs);
		auto end2 = std::chrono::high_resolution_clock::now();

		decoder_inputs[0] = &decoder_outputs[0]; // y
		decoder_inputs[1] = &decoder_outputs[1]; // k
		decoder_inputs[2] = &decoder_outputs[2]; // v
		decoder_inputs[3] = &decoder_outputs[3]; // y_emb
		AILIATensor& logits = decoder_outputs[4];
		AILIATensor& samples = decoder_outputs[5];

		bool stop = false;
		if (early_stop_num != -1 && (int)(y.shape.x - prefix_len) > early_stop_num){
			stop = true;
		}
		int token = argmax(logits);
		if (token == EOS || samples.data[0] == EOS){
			stop = true;
		}

		if (debug_token){
			PRINT_OUT("token %d\n", token);
		}

		if (benchmark){
            PRINT_OUT("ailia processing time %lld ms\n",  (long long)std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count());
		}

		if (stop){
			y = decoder_outputs[0];
			break;
		}
	}

	y.data = std::vector<float>(y.data.begin() + y.data.size() - idx, y.data.begin() + y.data.size() - 1); // dont store prefix and last eos element
	y.shape.x = (unsigned int)y.data.size();
	y.shape.y = 1;
	y.shape.z = 1;
	y.shape.w = 1;
	y.shape.dim = 3;

	return y;
}

AILIATensor vits_forward(AILIATensor text_seq, AILIATensor pred_semantic, AILIATensor ref_audio, AILIANetwork *net){
	std::vector<AILIATensor*> vits_inputs;
	vits_inputs.push_back(&text_seq);
	vits_inputs.push_back(&pred_semantic);
	vits_inputs.push_back(&ref_audio);
	std::vector<AILIATensor> vits_outputs;
	forward(net, vits_inputs, vits_outputs);
	return vits_outputs[0];
}

static int recognize_from_audio(AILIANetwork* net[MODEL_N])
{
	// unused: int status = AILIA_STATUS_SUCCESS;

	int sampleRate, nChannels, nSamples;
	std::vector<float> wave = read_wave_file(reference_wave.c_str(), &sampleRate, &nChannels, &nSamples);
	if (wave.size() == 0){
		PRINT_ERR("Input file not found (%s)\n", reference_wave.c_str());
		return AILIA_STATUS_ERROR_FILE_API;
	}

	// get sequence
	if (debug_token){
		PRINT_OUT("ref_seq\n");
	}
	AILIATensor ref_seq;
	ref_seq.data = cleaned_text_to_sequence(REF_PHONES, REF_PHONES_SIZE);
	ref_seq.shape.x = (unsigned int)ref_seq.data.size();
	ref_seq.shape.y = 1;
	ref_seq.shape.z = 1;
	ref_seq.shape.w = 1;
	ref_seq.shape.dim = 2;

	if (debug_token){
		PRINT_OUT("text_seq\n");
	}
	AILIATensor text_seq;
	text_seq.data = cleaned_text_to_sequence(TEXT_PHONES, TEXT_PHONES_SIZE);
	text_seq.shape.x = (unsigned int)text_seq.data.size();
	text_seq.shape.y = 1;
	text_seq.shape.z = 1;
	text_seq.shape.w = 1;
	text_seq.shape.dim = 2;

	const int BERT_DIM = 1024;
	AILIATensor ref_bert;
	AILIATensor text_bert;

	ref_bert.data = std::vector<float>(ref_seq.data.size() * BERT_DIM);
	ref_bert.shape.x = BERT_DIM;
	ref_bert.shape.y = (unsigned int)ref_seq.data.size();
	ref_bert.shape.z = 1;
	ref_bert.shape.w = 1;
	ref_bert.shape.dim = 2;

	text_bert.data = std::vector<float>(text_seq.data.size() * BERT_DIM);
	text_bert.shape.x = BERT_DIM;
	text_bert.shape.y = (unsigned int)text_seq.data.size();
	text_bert.shape.z = 1;
	text_bert.shape.w = 1;
	text_bert.shape.dim = 2;

	// resmaple to 16k and 32k
	const int vits_hps_data_sampling_rate = 32000;
	std::vector<float> zero_wav((size_t)(vits_hps_data_sampling_rate * 0.3));
	std::vector<float> wav16k = resample(wave, 16000, sampleRate, nChannels);
	std::vector<float> ref_audio_16k = wav16k;
	ref_audio_16k.insert(ref_audio_16k.end(), zero_wav.begin(), zero_wav.end());

	AILIATensor ref_audio;	
	ref_audio.data = resample(wave, vits_hps_data_sampling_rate, sampleRate, nChannels);
	ref_audio.shape.x = (unsigned int)ref_audio.data.size();
	ref_audio.shape.y = 1;
	ref_audio.shape.z = 1;
	ref_audio.shape.w = 1;
	ref_audio.shape.dim = 2;

	// ssl
	AILIATensor ssl_content = ssl_forward(ref_audio_16k, net[MODEL_SSL]);

	// t2s
	AILIATensor pred_semantic = t2s_forward(ref_seq, text_seq, ref_bert, text_bert, ssl_content, net);
	AILIATensor audio = vits_forward(text_seq, pred_semantic, ref_audio, net[MODEL_VITS]);

	// save
	write_wave_file("output.wav", audio.data, vits_hps_data_sampling_rate);

	PRINT_OUT("Program finished successfully.\n");

	return AILIA_STATUS_SUCCESS;
}


int main(int argc, char **argv)
{
	int status = argument_parser(argc, argv);
	if (status != AILIA_STATUS_SUCCESS) {
		return -1;
	}

	// env list
	unsigned int env_count;
	status = ailiaGetEnvironmentCount(&env_count);
	if (status != AILIA_STATUS_SUCCESS) {
		PRINT_ERR("ailiaGetEnvironmentCount Failed %d", status);
		return -1;
	}

	int env_id = AILIA_ENVIRONMENT_ID_AUTO;
	for (unsigned int i = 0; i < env_count; i++) {
		AILIAEnvironment* env;
		status = ailiaGetEnvironment(&env, i, AILIA_ENVIRONMENT_VERSION);
		bool is_fp16 = (env->props & AILIA_ENVIRONMENT_PROPERTY_FP16) != 0;
		PRINT_OUT("env_id : %d type : %d name : %s", env->id, env->type, env->name);
		if (is_fp16){
			PRINT_OUT(" (Warning : FP16 backend is not worked this model)\n");
			continue;
		}
		PRINT_OUT("\n");
		if (args_env_id == env->id){
			env_id = env->id;
		}
		if (args_env_id == -1 && env_id == AILIA_ENVIRONMENT_ID_AUTO){
			if (env->type == AILIA_ENVIRONMENT_TYPE_GPU) {
				env_id = env->id;
			}
		}
	}
	if (args_env_id == -1){
		PRINT_OUT("you can select environment using -e option\n");
	}

	// net initialize
	AILIANetwork *ailia[MODEL_N];
	for (int i = 0; i < MODEL_N; i++){
		status = ailiaCreate(&ailia[i], env_id, AILIA_MULTITHREAD_AUTO);
		if (status != AILIA_STATUS_SUCCESS) {
			PRINT_ERR("ailiaCreate failed %d\n", status);
			if (status == AILIA_STATUS_LICENSE_NOT_FOUND || status==AILIA_STATUS_LICENSE_EXPIRED){
				PRINT_OUT("License file not found or expired : please place license file (AILIA.lic)\n");
			}
			return -1;
		}

		status = ailiaSetMemoryMode(ailia[i], AILIA_MEMORY_OPTIMAIZE_DEFAULT | AILIA_MEMORY_REUSE_INTERSTAGE);
		if (status != AILIA_STATUS_SUCCESS) {
			PRINT_ERR("ailiaSetMemoryMode failed %d\n", status);
			ailiaDestroy(ailia[i]);
			return -1;
		}

		AILIAEnvironment *env_ptr = nullptr;
		status = ailiaGetSelectedEnvironment(ailia[i], &env_ptr, AILIA_ENVIRONMENT_VERSION);
		if (status != AILIA_STATUS_SUCCESS) {
			PRINT_ERR("ailiaGetSelectedEnvironment failed %d\n", status);
			ailiaDestroy(ailia[i]);
			return -1;
		}

		PRINT_OUT("selected env name : %s\n", env_ptr->name);

		status = ailiaOpenWeightFile(ailia[i], MODEL_NAME[i]);
		if (status != AILIA_STATUS_SUCCESS) {
			PRINT_ERR("ailiaOpenWeightFile failed %d\n", status);
			ailiaDestroy(ailia[i]);
			return -1;
		}
	}

	auto start2 = std::chrono::high_resolution_clock::now();
	status = recognize_from_audio(ailia);
	auto end2 = std::chrono::high_resolution_clock::now();
	if (benchmark){
		PRINT_OUT("total processing time %lld ms\n",  (long long)std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count());
	}

	for (int i = 0; i < MODEL_N; i++){
		ailiaDestroy(ailia[i]);
	}

	return status;
}
