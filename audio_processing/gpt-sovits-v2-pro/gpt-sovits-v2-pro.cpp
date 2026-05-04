/*******************************************************************
*
*    DESCRIPTION:
*      GPT-SoVITS V2-Pro with ailia Voice Sample Program
*    AUTHOR:
*      ax Inc.
*    DATE:2026/05/04
*
*******************************************************************/

#include "ailia_voice.h"
#include "ailia_voice_util.h"

#include <stdio.h>
#include <string.h>
#include <vector>
#include <string>

#include "wave_reader.h"
#include "wave_writer.h"

int main(int argc, char *argv[]){
	AILIAVoiceApiCallback callback = ailiaVoiceUtilGetCallback();

	printf("Usage : gpt-sovits-v2-pro [gpt-sovits-v2-pro/gpt-sovits-v2-pro-en/gpt-sovits-v2-pro-zh] [input_text]\n");

	const char * input_text = "";
	const char * lang = "";
	const char * model = "gpt-sovits-v2-pro";

	if (argc >= 2){
		model = argv[1];
		if (!(strcmp(model, "gpt-sovits-v2-pro") == 0 ||
		      strcmp(model, "gpt-sovits-v2-pro-en") == 0 ||
		      strcmp(model, "gpt-sovits-v2-pro-zh") == 0)){
			printf("model must be gpt-sovits-v2-pro, gpt-sovits-v2-pro-en or gpt-sovits-v2-pro-zh\n");
			return -1;
		}
	}
	if (argc >= 3){
		input_text = argv[2];
	}

	if (strcmp(model, "gpt-sovits-v2-pro-en") == 0){
		if (strlen(input_text) == 0){
			input_text = u8"Hello world.";
		}
		lang = "en";
	}else if (strcmp(model, "gpt-sovits-v2-pro-zh") == 0){
		if (strlen(input_text) == 0){
			input_text = u8"你好，世界。今天天气真好。"; // 你好，世界。今天天气真好。
		}
		lang = "zh";
	}else{
		if (strlen(input_text) == 0){
			input_text = u8"こんにちは。今日は新しいAIエンジンであるアイリアSDKを紹介します。"; // こんにちは。今日は新しいAIエンジンであるアイリアSDKを紹介します。
		}
		lang = "ja";
	}

	printf("Model : %s\n", model);
	printf("Input text : %s\n", input_text);
	printf("Language : %s\n", lang);

	AILIAVoice *net;
	int env_id = AILIA_ENVIRONMENT_ID_AUTO;
	int num_thread = AILIA_MULTITHREAD_AUTO;
	int memory_mode = AILIA_MEMORY_REDUCE_CONSTANT | AILIA_MEMORY_REDUCE_CONSTANT_WITH_INPUT_INITIALIZER | AILIA_MEMORY_REUSE_INTERSTAGE;
	int status = ailiaVoiceCreate(&net, env_id, num_thread, memory_mode, AILIA_VOICE_FLAG_NONE, callback, AILIA_VOICE_API_CALLBACK_VERSION);
	if (status != AILIA_STATUS_SUCCESS){
		printf("ailiaVoiceCreate error %d\n", status);
		return -1;
	}

	// Open Open JTalk dictionary (used by all variants)
	status = ailiaVoiceOpenDictionaryFileA(net, "./open_jtalk_dic_utf_8-1.11", AILIA_VOICE_DICTIONARY_TYPE_OPEN_JTALK);
	if (status != AILIA_STATUS_SUCCESS){
		printf("ailiaVoiceOpenDictionaryFileA error %d\n", status);
		return -1;
	}

	// Open language dependent dictionaries
	if (strcmp(model, "gpt-sovits-v2-pro-en") == 0){
		status = ailiaVoiceOpenDictionaryFileA(net, "./g2p_en", AILIA_VOICE_DICTIONARY_TYPE_G2P_EN);
		if (status != AILIA_STATUS_SUCCESS){
			printf("ailiaVoiceOpenDictionaryFileA g2p_en error %d\n", status);
			return -1;
		}
	}
	if (strcmp(model, "gpt-sovits-v2-pro-zh") == 0){
		status = ailiaVoiceOpenDictionaryFileA(net, "./g2p_cn", AILIA_VOICE_DICTIONARY_TYPE_G2P_CN);
		if (status != AILIA_STATUS_SUCCESS){
			printf("ailiaVoiceOpenDictionaryFileA g2p_cn error %d\n", status);
			return -1;
		}
		status = ailiaVoiceOpenDictionaryFileA(net, "./g2pw", AILIA_VOICE_DICTIONARY_TYPE_G2PW);
		if (status != AILIA_STATUS_SUCCESS){
			printf("ailiaVoiceOpenDictionaryFileA g2pw error %d\n", status);
			return -1;
		}
	}

	// Open GPT-SoVITS V2-Pro model files
	status = ailiaVoiceOpenGPTSoVITSV2ProModelFileA(net,
		"./gpt-sovits-v2-pro/t2s_encoder.onnx",
		"./gpt-sovits-v2-pro/t2s_fsdec.onnx",
		"./gpt-sovits-v2-pro/t2s_sdec.opt.onnx",
		"./gpt-sovits-v2-pro/cnhubert.onnx",
		"./gpt-sovits-v2-pro/vits.onnx",
		"./gpt-sovits-v2-pro/sv.onnx",
		"./gpt-sovits-v2-pro/chinese-roberta.onnx",
		"./gpt-sovits-v2-pro/vocab.txt");
	if (status != AILIA_STATUS_SUCCESS){
		printf("ailiaVoiceOpenGPTSoVITSV2ProModelFileA error %d\n", status);
		if (status == AILIA_STATUS_LICENSE_NOT_FOUND){
			printf("License file not found.\n");
		}
		return -1;
	}

	// Set reference audio
	{
		int sampleRate, nChannels, nSamples;
		const char *ref_audio = "reference_audio_captured_by_ax.wav";
		const char *ref_text = u8"水をマレーシアから買わなくてはならない。"; // 水をマレーシアから買わなくてはならない。
		int ref_g2p_type = AILIA_VOICE_G2P_TYPE_GPT_SOVITS_JA;

		std::vector<float> wave = read_wave_file(ref_audio, &sampleRate, &nChannels, &nSamples);
		if (wave.size() == 0){
			printf("Reference wav file not found or could not open: %s\n", ref_audio);
			return -1;
		}

		status = ailiaVoiceGraphemeToPhoneme(net, ref_text, ref_g2p_type);
		if (status != AILIA_STATUS_SUCCESS){
			printf("ailiaVoiceGraphemeToPhoneme error %d\n", status);
			return -1;
		}
		unsigned int len = 0;
		status = ailiaVoiceGetFeatureLength(net, &len);
		if (status != AILIA_STATUS_SUCCESS){
			printf("ailiaVoiceGetFeatureLength error %d\n", status);
			return -1;
		}
		std::vector<char> ref_features;
		ref_features.resize(len);
		status = ailiaVoiceGetFeatures(net, &ref_features[0], len);
		if (status != AILIA_STATUS_SUCCESS){
			printf("ailiaVoiceGetFeatures error %d\n", status);
			return -1;
		}
		printf("Reference Features : %s\n", &ref_features[0]);

		status = ailiaVoiceSetReference(net, &wave[0], (unsigned int)(wave.size() * sizeof(float)), nChannels, sampleRate, &ref_features[0]);
		if (status != AILIA_STATUS_SUCCESS){
			printf("ailiaVoiceSetReference error %d\n", status);
			return -1;
		}
	}

	// G2P for the input text
	std::vector<char> features;
	if (strcmp(model, "gpt-sovits-v2-pro-zh") == 0){
		status = ailiaVoiceGraphemeToPhoneme(net, input_text, AILIA_VOICE_G2P_TYPE_GPT_SOVITS_ZH);
	}else if (strcmp(model, "gpt-sovits-v2-pro-en") == 0){
		status = ailiaVoiceGraphemeToPhoneme(net, input_text, AILIA_VOICE_G2P_TYPE_GPT_SOVITS_EN);
	}else{
		status = ailiaVoiceGraphemeToPhoneme(net, input_text, AILIA_VOICE_G2P_TYPE_GPT_SOVITS_JA);
	}
	if (status != AILIA_STATUS_SUCCESS){
		printf("ailiaVoiceGraphemeToPhoneme error %d\n", status);
		return -1;
	}
	unsigned int len = 0;
	status = ailiaVoiceGetFeatureLength(net, &len);
	if (status != AILIA_STATUS_SUCCESS){
		printf("ailiaVoiceGetFeatureLength error %d\n", status);
		return -1;
	}
	features.resize(len);
	status = ailiaVoiceGetFeatures(net, &features[0], len);
	if (status != AILIA_STATUS_SUCCESS){
		printf("ailiaVoiceGetFeatures error %d\n", status);
		return -1;
	}
	printf("Features : %s\n", &features[0]);

	// Inference
	status = ailiaVoiceInference(net, &features[0]);
	if (status != AILIA_STATUS_SUCCESS){
		printf("ailiaVoiceInference error %d\n", status);
		return -1;
	}

	unsigned int samples, channels, sampling_rate;
	status = ailiaVoiceGetWaveInfo(net, &samples, &channels, &sampling_rate);
	if (status != AILIA_STATUS_SUCCESS){
		printf("ailiaVoiceGetWaveInfo error %d\n", status);
		return -1;
	}

	std::vector<float> buf(samples * channels);
	status = ailiaVoiceGetWave(net, &buf[0], (unsigned int)(buf.size() * sizeof(float)));
	if (status != AILIA_STATUS_SUCCESS){
		printf("ailiaVoiceGetWave error %d\n", status);
		return -1;
	}

	printf("Wave samples : %d\nWave channels : %d\nWave sampling rate : %d\n", samples, channels, sampling_rate);

	write_wave_file("output.wav", buf, sampling_rate);

	ailiaVoiceDestroy(net);
	return 0;
}
