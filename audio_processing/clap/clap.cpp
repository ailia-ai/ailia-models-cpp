/*******************************************************************
*
*    DESCRIPTION:
*      AILIA clap sample
*    AUTHOR:
*
*    DATE:2024/01/25
*
*******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <vector>
#include <string>
#include <algorithm>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#define sleep(n) Sleep(n*1000)
#else
#include <unistd.h>
#endif

#undef UNICODE

#include "ailia.h"
#include "ailia_audio.h"
#include "ailia_tokenizer.h"

#include "utils.h"
#include "wave_reader.h"
#include "clap_utils.h"

// ======================
// Parameters
// ======================

#define CLAP_AUDIO_WEIGHT_PATH	"CLAP_audio_LAION-Audio-630K_with_fusion.onnx"
#define CLAP_AUDIO_MODEL_PATH	"CLAP_audio_LAION-Audio-630K_with_fusion.onnx.prototxt"
#define CLAP_TEXT_ROBERTAMODEL_WEIGHT_PATH	"CLAP_text_text_branch_RobertaModel_roberta-base.onnx"
#define CLAP_TEXT_ROBERTAMODEL_MODEL_PATH	"CLAP_text_text_branch_RobertaModel_roberta-base.onnx.prototxt"
#define CLAP_TEXT_PROJECTION_WEIGHT_PATH	"CLAP_text_projection_LAION-Audio-630K_with_fusion.onnx"
#define CLAP_TEXT_PROJECTION_MODEL_PATH		"CLAP_text_projection_LAION-Audio-630K_with_fusion.onnx.prototxt"

#define BENCHMARK_ITERS 5

static std::string weight_audio(CLAP_AUDIO_WEIGHT_PATH);
static std::string model_audio(CLAP_AUDIO_MODEL_PATH);
static std::string weight_text_robertamodel(CLAP_TEXT_ROBERTAMODEL_WEIGHT_PATH);
static std::string model_text_robertamodel(CLAP_TEXT_ROBERTAMODEL_MODEL_PATH);
static std::string weight_text_projection(CLAP_TEXT_PROJECTION_WEIGHT_PATH);
static std::string model_text_projection(CLAP_TEXT_PROJECTION_MODEL_PATH);

static std::string vocab_file("roberta-base/vocab.json");
static std::string merge_file("roberta-base/merges.txt");

static std::string input_wav_path("input.wav");
static std::vector<std::string> texts;
static unsigned int token_length = 77;

typedef float TYPE_IDS;
typedef float TYPE_MASK;

static bool benchmark  = false;
static int args_env_id = -1;
bool debug = false;


// ======================
// Arguemnt Parser
// ======================

static void print_usage()
{
    PRINT_OUT("usage: clap [-h] [-i WAV_FILE] [-t TEXT] [-v VOCAB_FILE] [-m MERGE_FILE] [-e ENV_ID]\n");
    return;
}

static void print_help()
{
    PRINT_OUT("\n");
    PRINT_OUT("clap classification model\n");
    PRINT_OUT("\n");
    PRINT_OUT("optional arguments:\n");
    PRINT_OUT("  -h, --help            show this help message and exit\n");
    PRINT_OUT("  -i WAV_FILE, --input WAV_FILE\n");
    PRINT_OUT("                        The input wav file.\n");
    PRINT_OUT("  -t TEXT, --text TEXT\n");
    PRINT_OUT("                        The input text. (can be called multiple times.)\n");
	PRINT_OUT("  -v VOCAB_FILE, --vocab_file VOCAB_FILE\n");
    PRINT_OUT("                        The vocab file in roberta tokenizer.\n");
	PRINT_OUT("  -m MERGE_FILE, --merge_file MERGE_FILE\n");
    PRINT_OUT("                        The merge file in roberta tokenizer.\n");
//    PRINT_OUT("  -b, --benchmark       Running the inference on the same input 5 times to\n");
//    PRINT_OUT("                        measure execution performance. (Cannot be used in\n");
//    PRINT_OUT("                        video mode)\n");
	PRINT_OUT("  -e ENV_ID, --env_id ENV_ID\n");
	PRINT_OUT("                        The backend environment id.\n");
    return;
}

static void print_error(std::string arg)
{
    PRINT_ERR("clap: error: unrecognized arguments: %s\n", arg.c_str());
    return;
}

static int argument_parser(int argc, char **argv)
{
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
		if (arg == "-i" || arg == "--input") {
			input_wav_path = argv[++i];
		}
		else if (arg == "-t" || arg == "--text") {
			texts.push_back(argv[++i]);
		}
		else if (arg == "-v" || arg == "--vocab_file") {
			vocab_file = argv[++i];
		}
		else if (arg == "-m" || arg == "--merge_file") {
			merge_file = argv[++i];
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
			args_env_id = atoi(argv[++i]);
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
// Utils
// ======================
static float cos_sim(float* a, float* b, size_t len)
{
    float dot = 0, na = 0, nb = 0;
    for(size_t i=0; i<len; i++){
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if(na < FLT_EPSILON || nb < FLT_EPSILON ) return 0;
    return dot / (sqrtf(na) * sqrtf(nb));
}

// ======================
// Audio embeddings
// ======================
static std::vector<float> audio_embedding(AILIANetwork *ailia_audio, std::string wav_file)
{
    int status;
    std::vector<float> feature;
    
    const int target_sample_rate = 48000;
    int sampleRate=0, nChannels=0, nSamples=0;
    std::vector<float> audio_waveform = read_wave_file(wav_file.c_str(), &sampleRate, &nChannels, &nSamples);
    if (debug){
        PRINT_OUT("wav sampleRate=%d, nChannels=%d, nSamples=%d : %s\n", sampleRate, nChannels, nSamples, wav_file.c_str());
    }
    
    // resample
    if(sampleRate != target_sample_rate){
        int dst_n = 0;
        status = ailiaAudioGetResampleLen(&dst_n, target_sample_rate, (int)audio_waveform.size(), sampleRate);
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaAudioGetResampleLen failed %d\n", status);
            return feature;
        }
        if (debug){
            PRINT_OUT("convert sample rate %d to %d, length %zu to %d\n", sampleRate, target_sample_rate, audio_waveform.size(), dst_n);
        }
        std::vector<float> new_audio_waveform(dst_n);
        status = ailiaAudioResample(&new_audio_waveform[0], &audio_waveform[0], target_sample_rate, dst_n, sampleRate, (int)audio_waveform.size());
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaAudioResample failed %d\n", status);
            return feature;
        }
        audio_waveform = new_audio_waveform;
    }
    
    // quantize as int16
    for(auto v=audio_waveform.begin(); v!=audio_waveform.end(); ++v){
        float x = *v;
        if(x < -1) x = -1;
        if(x > 1) x = 1;
        int16_t y = (int16_t)(32767.f * x);
        *v = (float)y / 32767.f;
    }
    
    AUDIO_CONFIG audio_config;
    std::vector<float> mel_fusion = get_audio_features(audio_waveform, 480000, "fusion", "repeatpad", audio_config);
    
    AILIAShape shape;
	unsigned int blob_idx_longer, blob_idx_mel_fusion, blob_idx_out0;
	
	// get input info
	status = ailiaFindBlobIndexByName(ailia_audio, &blob_idx_longer, "longer");
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaFindBlobIndexByName failed %d\n", status);
        return feature;
    }
	status = ailiaFindBlobIndexByName(ailia_audio, &blob_idx_mel_fusion, "mel_fusion");
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaFindBlobIndexByName failed %d\n", status);
        return feature;
    }
	status = ailiaGetBlobIndexByOutputIndex(ailia_audio, &blob_idx_out0, 0);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaGetBlobIndexByOutputIndex failed %d\n", status);
        return feature;
    }
    status = ailiaGetBlobShape(ailia_audio, &shape, blob_idx_mel_fusion, AILIA_SHAPE_VERSION);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaGetBlobShape failed %d\n", status);
        return feature;
    }
    if(debug){
        PRINT_OUT("audio input=%d,%d output=%d mel_fusion shape=[%d,%d,%d]\n", blob_idx_longer, blob_idx_mel_fusion, blob_idx_out0, shape.z, shape.y, shape.x);
    }
    if(mel_fusion.size() != (shape.x * shape.y * shape.z)){
        PRINT_ERR("Invalid length of mel_fusion : %zu must be %d\n", mel_fusion.size(), shape.x * shape.y * shape.z);
        return feature;
    }

	// set input
    float longer = 1;   // True
	status = ailiaSetInputBlobData(ailia_audio, &longer, sizeof(longer), blob_idx_longer);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaSetInputBlobData failed %d\n", status);
        return feature;
    }
	status = ailiaSetInputBlobData(ailia_audio, &mel_fusion[0], (unsigned int)(mel_fusion.size() * sizeof(float)), blob_idx_mel_fusion);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaSetInputBlobData failed %d\n", status);
        return feature;
    }

	// predict ailia_text_robertamodel
	status = ailiaUpdate(ailia_audio);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaUpdate failed %d\n", status);
        return feature;
    }
	status = ailiaGetBlobShape(ailia_audio, &shape, blob_idx_out0, AILIA_SHAPE_VERSION);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaGetBlobShape failed %d\n", status);
        return feature;
    }
    if(debug){
        PRINT_OUT("audio output shape=[%d,%d]\n", shape.y, shape.x);
    }

	// get output
	feature = std::vector<float>(shape.x);
	status = ailiaGetBlobData(ailia_audio, &feature[0], (unsigned int)(feature.size() * sizeof(float)), blob_idx_out0);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaGetBlobData failed %d\n", status);
        return feature;
    }

    return feature;
}

// ======================
// Text embeddings
// ======================

static void tokenize(AILIATokenizer* tokenizer, std::string text, 
	std::vector<TYPE_IDS>& input_ids, std::vector<TYPE_MASK>& attention_mask,
	const unsigned int token_length=77)
{
    if (debug){
        PRINT_OUT("Input Text : %s\n", text.c_str());
    }
	ailiaTokenizerEncode(tokenizer, text.c_str());
	unsigned int count;
	ailiaTokenizerGetTokenCount(tokenizer, &count);
	std::vector<int> tokens(count);
	ailiaTokenizerGetTokens(tokenizer, &tokens[0], count);

	input_ids = std::vector<TYPE_IDS>(token_length);
	attention_mask = std::vector<TYPE_MASK>(token_length);
    for (unsigned int i = 0; i < token_length; i++){
        if (i < tokens.size()){
            input_ids[i] = (TYPE_IDS)tokens[i];
			attention_mask[i] = 1;
        }else{
            input_ids[i] = 1;
			attention_mask[i] = 0;
        }
    }
    if (debug){
        PRINT_OUT("input Tokens : ");
        for (size_t i = 0; i < input_ids.size(); i++){
            PRINT_OUT("%.0f ", input_ids[i]);
        }
        PRINT_OUT("\n");
        //PRINT_OUT("input Mask   : ");
        //for (int i = 0; i < input_ids.size(); i++){
        //    PRINT_OUT("%.0f ", attention_mask[i]);
        //}
        //PRINT_OUT("\n");
    }
}

static std::vector<float> text_embedding(AILIANetwork *ailia_text_robertamodel, AILIANetwork *ailia_text_projection,
	std::vector<TYPE_IDS>& ary_input_ids, std::vector<TYPE_MASK>& ary_attention_mask, 
    unsigned int* dim_feature,
	const unsigned int num_texts, const unsigned int token_length=77)
{
    std::vector<float> features, branch;
	int status;
	AILIAShape shape;
	unsigned int blob_idx_ids, blob_idx_mask, blob_idx_out1;
	unsigned int blob_idx_x, blob_idx_text_embed;
	
	// get info of ailia_text_robertamodel
	status = ailiaFindBlobIndexByName(ailia_text_robertamodel, &blob_idx_ids, "input_ids");
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaFindBlobIndexByName failed %d\n", status);
        return features;
    }
	status = ailiaFindBlobIndexByName(ailia_text_robertamodel, &blob_idx_mask, "attention_mask");
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaFindBlobIndexByName failed %d\n", status);
        return features;
    }
	shape.dim = 2;
	shape.x = token_length;
	shape.y = num_texts;
	shape.z = 0;
	shape.w = 0;
	status = ailiaSetInputBlobShape(ailia_text_robertamodel, &shape, blob_idx_ids, AILIA_SHAPE_VERSION);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaSetInputBlobShape failed %d\n", status);
        return features;
    }
	status = ailiaSetInputBlobShape(ailia_text_robertamodel, &shape, blob_idx_mask, AILIA_SHAPE_VERSION);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaSetInputBlobShape failed %d\n", status);
        return features;
    }
	status = ailiaGetBlobIndexByOutputIndex(ailia_text_robertamodel, &blob_idx_out1, 1);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaGetBlobIndexByOutputIndex failed %d\n", status);
        return features;
    }
	status = ailiaGetBlobShape(ailia_text_robertamodel, &shape, blob_idx_out1, AILIA_SHAPE_VERSION);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaGetBlobShape failed %d\n", status);
        return features;
    }
    if(debug){
        PRINT_OUT("text_robertamodel input=%d,%d output=%d outshape=[%d,%d]\n", blob_idx_ids, blob_idx_mask, blob_idx_out1, shape.y, shape.x);
    }
	
	// set input of ailia_text_robertamodel
	status = ailiaSetInputBlobData(ailia_text_robertamodel, &ary_input_ids[0], (unsigned int)(ary_input_ids.size() * sizeof(TYPE_IDS)), blob_idx_ids);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaSetInputBlobData failed %d\n", status);
        return features;
    }
	status = ailiaSetInputBlobData(ailia_text_robertamodel, &ary_attention_mask[0], (unsigned int)(ary_attention_mask.size() * sizeof(TYPE_MASK)), blob_idx_mask);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaSetInputBlobData failed %d\n", status);
        return features;
    }

	// predict ailia_text_robertamodel
	status = ailiaUpdate(ailia_text_robertamodel);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaUpdate failed %d\n", status);
        return features;
    }
	
	// get output
	branch = std::vector<float>(shape.x * shape.y);
	status = ailiaGetBlobData(ailia_text_robertamodel, &branch[0], (unsigned int)(branch.size() * sizeof(float)), blob_idx_out1);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaGetBlobData failed %d\n", status);
        return features;
    }

	// get info of ailia_text_projection
	status = ailiaFindBlobIndexByName(ailia_text_projection, &blob_idx_x, "x");
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaFindBlobIndexByName failed %d\n", status);
        return features;
    }
	status = ailiaSetInputBlobShape(ailia_text_projection, &shape, blob_idx_x, AILIA_SHAPE_VERSION);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaSetInputBlobShape failed %d\n", status);
        return features;
    }
	status = ailiaGetBlobIndexByOutputIndex(ailia_text_projection, &blob_idx_text_embed, 0);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaGetBlobIndexByOutputIndex failed %d\n", status);
        return features;
    }
	status = ailiaGetBlobShape(ailia_text_projection, &shape, blob_idx_text_embed, AILIA_SHAPE_VERSION);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaGetBlobShape failed %d\n", status);
        return features;
    }
    if(debug){
        PRINT_OUT("text_projection input=%d output=%d outshape=[%d,%d]\n", blob_idx_x, blob_idx_text_embed, shape.y, shape.x);
    }

	// set input of ailia_text_projection
	status = ailiaSetInputBlobData(ailia_text_projection, &branch[0], (unsigned int)(branch.size() * sizeof(float)), blob_idx_x);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaSetInputBlobData failed %d\n", status);
        return features;
    }

	// predict ailia_text_projection
	status = ailiaUpdate(ailia_text_projection);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaUpdate failed %d\n", status);
        return features;
    }
	
	// get output
	features = std::vector<float>(shape.x * shape.y);
	status = ailiaGetBlobData(ailia_text_projection, &features[0], (unsigned int)(features.size() * sizeof(float)), blob_idx_text_embed);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaGetBlobData failed %d\n", status);
        return features;
    }

    if(dim_feature) *dim_feature = shape.x;
    return features;
}

// ======================
// Main functions
// ======================

static int get_env_id(void)
{
    unsigned int env_count;
	int status = ailiaGetEnvironmentCount(&env_count);
	if (status != AILIA_STATUS_SUCCESS) {
		PRINT_ERR("ailiaGetEnvironmentCount Failed %d", status);
		return -1;
	}

	int env_id = AILIA_ENVIRONMENT_ID_AUTO;
	for (unsigned int i = 0; i < env_count; i++) {
		AILIAEnvironment* env;
		status = ailiaGetEnvironment(&env, i, AILIA_ENVIRONMENT_VERSION);
		// unused: bool is_fp16 = (env->props & AILIA_ENVIRONMENT_PROPERTY_FP16) != 0;
		PRINT_OUT("env_id : %d type : %d name : %s", env->id, env->type, env->name);
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

    return env_id;
}

static int initialize_ailia(AILIANetwork **ailia, int env_id, std::string model_file, std::string weight_file){
    int status = ailiaCreate(ailia, env_id, AILIA_MULTITHREAD_AUTO);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaCreate failed %d\n", status);
        return -1;
    }
    status = ailiaOpenStreamFile(*ailia, model_file.c_str());
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaOpenStreamFile failed %d\n", status);
        PRINT_ERR("ailiaGetErrorDetail %s\n", ailiaGetErrorDetail(*ailia));
        ailiaDestroy(*ailia);
        return -1;
    }
    status = ailiaOpenWeightFile(*ailia, weight_file.c_str());
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaOpenWeightFile failed %d\n", status);
        ailiaDestroy(*ailia);
        return -1;
    }
    return AILIA_STATUS_SUCCESS;
}

int main(int argc, char **argv)
{
    int status = argument_parser(argc, argv);
    if (status != AILIA_STATUS_SUCCESS) {
        return -1;
    }

	// env list
    int env_id = get_env_id();

    // net initialize
    AILIANetwork *ailia_audio;
    status = initialize_ailia(&ailia_audio, env_id, model_audio, weight_audio);
    if (status != AILIA_STATUS_SUCCESS) {
        return -1;
    }
    AILIANetwork *ailia_text_robertamodel;
    status = initialize_ailia(&ailia_text_robertamodel, env_id, model_text_robertamodel, weight_text_robertamodel);
    if (status != AILIA_STATUS_SUCCESS) {
        return -1;
    }
    AILIANetwork *ailia_text_projection;
    status = initialize_ailia(&ailia_text_projection, env_id, model_text_projection, weight_text_projection);
    if (status != AILIA_STATUS_SUCCESS) {
        return -1;
    }
	
    // Tokenize
    if (texts.size() == 0){
		texts = {
			"applause applaud clap", 
			"The crowd is clapping.",
			"I love the contrastive learning", 
			"bell", 
			"soccer", 
			"open the door.",
			"applause",
			"dog",
			"dog barking"
		};
    }
	AILIATokenizer *tokenizer;
	status = ailiaTokenizerCreate(&tokenizer, AILIA_TOKENIZER_TYPE_ROBERTA, AILIA_TOKENIZER_FLAG_NONE);
	if (status != 0){
		PRINT_ERR("ailiaTokenizerCreate error %d\n", status);
		return -1;
	}
	status = ailiaTokenizerOpenVocabFile(tokenizer, vocab_file.c_str());
	if (status != 0){
		printf("ailiaTokenizerOpenVocabFile error %d\n", status);
		return -1;
	}
	status = ailiaTokenizerOpenMergeFile(tokenizer, merge_file.c_str());
	if (status != 0){
		printf("ailiaTokenizerOpenMergeFile error %d\n", status);
		return -1;
	}
    PRINT_OUT("Tokenize...\n");
	unsigned int num_texts = (unsigned int)texts.size();
    std::vector<TYPE_IDS> ary_input_ids(num_texts * token_length);
    std::vector<TYPE_MASK> ary_attention_mask(num_texts * token_length);
    for (size_t i = 0; i < num_texts; i++){
		std::vector<TYPE_IDS> input_ids;
		std::vector<TYPE_MASK> attention_mask;
        tokenize(tokenizer, texts[i], input_ids, attention_mask, token_length);
		memcpy(&ary_input_ids[i * token_length], &input_ids[0], sizeof(TYPE_IDS) * token_length);
		memcpy(&ary_attention_mask[i * token_length], &attention_mask[0], sizeof(TYPE_MASK) * token_length);
    }
	ailiaTokenizerDestroy(tokenizer);

    // text embedding
    PRINT_OUT("Text embedding...\n");
    unsigned int dim_text_feature = 0;
    std::vector<float> text_features = text_embedding(ailia_text_robertamodel, ailia_text_projection, 
		ary_input_ids, ary_attention_mask, &dim_text_feature, num_texts, token_length);

    // audio embedding
    PRINT_OUT("Audio embedding...\n");
    std::vector<float> audio_feature = audio_embedding(ailia_audio, input_wav_path);
    if(dim_text_feature > 0 && dim_text_feature == audio_feature.size() && text_features.size() > 0){
        PRINT_OUT("===== cosine similality between text and audio =====\n");
        PRINT_OUT("audio: %s\n", input_wav_path.c_str());
        for (size_t i = 0; i < num_texts; i++){
            float sim = cos_sim(&audio_feature[0], &text_features[i * dim_text_feature], dim_text_feature);
            PRINT_OUT("cossim=%.4f, word=%s\n", sim, texts[i].c_str());
        }
    }

    // release instance
    ailiaDestroy(ailia_audio);
    ailiaDestroy(ailia_text_robertamodel);
	ailiaDestroy(ailia_text_projection);

    PRINT_OUT("Program finished successfully.\n");
    return status;
}
