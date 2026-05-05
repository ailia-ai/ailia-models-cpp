/*******************************************************************
*
*    DESCRIPTION:
*      AILIA Sentence Transformers sample
*    AUTHOR:
*
*    DATE:2023/06/15
*
*******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <string>
#include <math.h>

#undef UNICODE

#include "ailia.h"
#include "ailia_tokenizer.h"

bool debug = false;


// ======================
// Parameters
// ======================

#define WEIGHT_PATH "paraphrase-multilingual-mpnet-base-v2.onnx"
#define MODEL_PATH  "paraphrase-multilingual-mpnet-base-v2.onnx.prototxt"

#if defined(_WIN32) || defined(_WIN64)
#define PRINT_OUT(...) fprintf_s(stdout, __VA_ARGS__)
#define PRINT_ERR(...) fprintf_s(stderr, __VA_ARGS__)
#else
#define PRINT_OUT(...) fprintf(stdout, __VA_ARGS__)
#define PRINT_ERR(...) fprintf(stderr, __VA_ARGS__)
#endif

#define BENCHMARK_ITERS 5

#define NUM_INPUTS 2
#define NUM_OUTPUTS 2
#define NUM_STATE 768

static std::string weight(WEIGHT_PATH);
static std::string model(MODEL_PATH);

static bool benchmark  = false;
static int args_env_id = -1;

std::string input_text = "nnapiの速度";


// ======================
// Arguemnt Parser
// ======================

static void print_usage()
{
	PRINT_OUT("usage: sentenace_transformers [-h] [-i TEXT] [-b] [-e ENV_ID]\n");
	return;
}


static void print_help()
{
	PRINT_OUT("\n");
	PRINT_OUT("sentenace_transformers model\n");
	PRINT_OUT("\n");
	PRINT_OUT("optional arguments:\n");
	PRINT_OUT("  -h, --help            show this help message and exit\n");
	PRINT_OUT("  -i TEXT, --input TEXT\n");
	PRINT_OUT("                        The input text.\n");
	PRINT_OUT("  -b, --benchmark       Running the inference on the same input 5 times to\n");
	PRINT_OUT("                        measure execution performance. (Cannot be used in\n");
	PRINT_OUT("                        video mode)\n");
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
				input_text = arg;
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

void softmax(float *data, int n){
	float sum=0;
	for(int i=0;i<n;i++){
		sum+=exp(data[i]);
	}
	for(int i=0;i<n;i++){
		data[i]=exp(data[i])/sum;
	}
}

std::vector<int> topk(const float *logits, int num_words, int kn){
	std::vector<int> results;
	for (int k = 0; k < kn; k++){
		float prob = -INFINITY;
		int arg_max = 0;
		for (int j = 0; j < num_words; j++){
			if (prob < logits[j]){
				bool already = false;
				for (int l = 0; l < k; l++){
					if (results[l] == j){
						already = true;
						break;
					}
				}
				if (already){
					continue;
				}
				prob = logits[j];
				arg_max = j;
			}
		}
		results.push_back(arg_max);
	}
	return results;
}

void setErrorDetail(const char *func, const char *detail){
	PRINT_ERR("Error %s Detail %s\n", func, detail);
}

std::vector<int> encode(std::string text, struct AILIATokenizer *tokenizer){
	std::vector<int> tokens(0);
	int status = ailiaTokenizerEncode(tokenizer, text.c_str());
	if (status != AILIA_STATUS_SUCCESS){
		setErrorDetail("ailiaTokenizerEncode", "");
		return tokens;
	}
	unsigned int count;
	status = ailiaTokenizerGetTokenCount(tokenizer, &count);
	if (status != AILIA_STATUS_SUCCESS){
		setErrorDetail("ailiaTokenizerGetTokenCount", "");
		return tokens;
	}
	tokens.resize(count);
	status = ailiaTokenizerGetTokens(tokenizer, &tokens[0], count);
	if (status != AILIA_STATUS_SUCCESS){
		setErrorDetail("ailiaTokenizerGetTokens", "");
		return tokens;
	}
	return tokens;
}

std::string decode(std::vector<int> &tokens, struct AILIATokenizer *tokenizer){
	int status = ailiaTokenizerDecode(tokenizer, &tokens[0], tokens.size());
	if (status != AILIA_STATUS_SUCCESS){
		setErrorDetail("ailiaTokenizerDecode", "");
		return std::string("");
	}
	unsigned int len;
	status = ailiaTokenizerGetTextLength(tokenizer, &len);
	if (status != AILIA_STATUS_SUCCESS){
		setErrorDetail("ailiaTokenizerGetTextLength", "");
		return std::string("");
	}
	if (len == 0){
		return std::string("");
	}
	std::vector<char> out_text(len);
	char * p_text = &out_text[0];
	status = ailiaTokenizerGetText(tokenizer, p_text, len);
	if (status != AILIA_STATUS_SUCCESS){
		setErrorDetail("ailiaTokenizerGetText", "");
		return std::string("");
	}
	return std::string(p_text);
}

int forward(AILIANetwork *ailia, std::vector<float> *inputs[NUM_INPUTS], std::vector<float> *outputs[NUM_OUTPUTS]){
	int status;

	for (int i = 0; i < NUM_INPUTS; i++){
		unsigned int input_blob_idx = 0;
		status = ailiaGetBlobIndexByInputIndex(ailia, &input_blob_idx, i);
		if (status != AILIA_STATUS_SUCCESS) {
			setErrorDetail("ailiaGetBlobIndexByInputIndex", ailiaGetErrorDetail(ailia));
			return status;
		}

		AILIAShape sequence_shape;
		int batch_size = 1;
		sequence_shape.x=inputs[i]->size();
		sequence_shape.y=batch_size;
		sequence_shape.z=1;
		sequence_shape.w=1;
		sequence_shape.dim=2;

		if (debug){
			printf("input blob shape %d %d %d %d dims %d\n",sequence_shape.x,sequence_shape.y,sequence_shape.z,sequence_shape.w,sequence_shape.dim);
		}

		status = ailiaSetInputBlobShape(ailia,&sequence_shape,input_blob_idx,AILIA_SHAPE_VERSION);
		if(status!=AILIA_STATUS_SUCCESS){
			setErrorDetail("ailiaSetInputBlobShape",ailiaGetErrorDetail(ailia));
			return status;
		}

		if (inputs[i]->size() > 0){
			status = ailiaSetInputBlobData(ailia, &(*inputs[i])[0], inputs[i]->size() * sizeof(float), input_blob_idx);
			if (status != AILIA_STATUS_SUCCESS) {
				setErrorDetail("ailiaSetInputBlobData",ailiaGetErrorDetail(ailia));
				return status;
			}
		}
	}

	status = ailiaUpdate(ailia);
	if (status != AILIA_STATUS_SUCCESS) {
		setErrorDetail("ailiaUpdate",ailiaGetErrorDetail(ailia));
		return status;
	}

	for (int i = 0; i < NUM_OUTPUTS; i++){
		unsigned int output_blob_idx = 0;
		status = ailiaGetBlobIndexByOutputIndex(ailia, &output_blob_idx, i);
		if (status != AILIA_STATUS_SUCCESS) {
			setErrorDetail("ailiaGetBlobIndexByInputIndex",ailiaGetErrorDetail(ailia));
			return status;
		}

		AILIAShape output_blob_shape;
		status=ailiaGetBlobShape(ailia,&output_blob_shape,output_blob_idx,AILIA_SHAPE_VERSION);
		if(status!=AILIA_STATUS_SUCCESS){
			setErrorDetail("ailiaGetBlobShape", ailiaGetErrorDetail(ailia));
			return status;
		}

		if (debug){
			printf("output_blob_shape %d %d %d %d dims %d\n",output_blob_shape.x,output_blob_shape.y,output_blob_shape.z,output_blob_shape.w,output_blob_shape.dim);
		}

		(*outputs[i]).resize(output_blob_shape.x*output_blob_shape.y*output_blob_shape.z*output_blob_shape.w);

		status =ailiaGetBlobData(ailia, &(*outputs[i])[0], outputs[i]->size() * sizeof(float), output_blob_idx);
		if (status != AILIA_STATUS_SUCCESS) {
			setErrorDetail("ailiaGetBlobData",ailiaGetErrorDetail(ailia));
			return status;
		}
	}

	return AILIA_STATUS_SUCCESS;
}

std::vector<float> mean_pool(std::vector<float> &features){
	std::vector<float> mean(NUM_STATE);
	for (int j = 0; j < NUM_STATE; j++){
		float sum = 0;
		int num_sentence = features.size() / NUM_STATE;
		for (int i = 0; i < num_sentence; i++){
			sum += features[i * NUM_STATE + j];
		}
		sum /= num_sentence;
		mean[j] = sum;
	}
	return mean;
}

std::vector<std::string> split(std::string text) {
	const int MIN_LENGTH = 5;
	std::string delimiter = u8"。";
	std::vector<std::string> sents;
	
	// Replace newline characters and split into sentences
	size_t pos = 0;
	while ((pos = text.find("\n", pos)) != std::string::npos) {
		text.replace(pos, 1, "");
	}
	
	pos = 0;
	while ((pos = text.find(delimiter, pos)) != std::string::npos) {
		text.replace(pos, delimiter.length(), delimiter + " ");
		pos += delimiter.length() + 1;
	}

	// Split sentences and filter based on minimum length
	pos = 0;
	while ((pos = text.find(delimiter, pos)) != std::string::npos) {
		std::string sentence = text.substr(0, pos + delimiter.length());
		text.erase(0, pos + delimiter.length());
		if (sentence.length() > MIN_LENGTH) {
			sents.push_back(sentence);
		}
		pos = 0;
	}
	
	return sents;
}

std::vector<std::string> open_texts(std::string parh){
	FILE *fp = fopen("sample.txt", "r");
	std::vector<char> text;
	while(!feof(fp)){
		char c = fgetc(fp);
		text.push_back(c);
	}
	text.push_back('\0');
	fclose(fp);
	return split(&text[0]);
}

std::vector<float> calc_embedding(AILIANetwork* net, struct AILIATokenizer *tokenizer, std::string text, bool prompt)
{
	std::vector<int> tokens = encode(text, tokenizer);

	std::vector<float> input_ids(tokens.size());
	std::vector<float> attention_mask(tokens.size());

	if (prompt && debug){
		PRINT_OUT("Input Tokens :\n");
	}
	for (size_t i = 0; i < tokens.size(); i++){
		input_ids[i] = (float)tokens[i];
		attention_mask[i] = 1;
		if (prompt && debug){
			PRINT_OUT("%d ", (int)input_ids[i]);
		}
	}
	if (prompt && debug){
		PRINT_OUT("\n");
	}

	std::vector<float> *inputs[NUM_INPUTS];
	inputs[0] = &input_ids;
	inputs[1] = &attention_mask;

	std::vector<float> features(tokens.size() * NUM_STATE);
	std::vector<float> temp(NUM_STATE);

	std::vector<float> *outputs[NUM_OUTPUTS];
	outputs[0] = &features;
	outputs[1] = &temp;

	std::vector<float> pool_features;
	int status = forward(net, inputs, outputs);
	if (status != AILIA_STATUS_SUCCESS){
		return pool_features;
	}

	pool_features = mean_pool(features);
	return pool_features;
}

float norm(std::vector<float> & vec1){
	float norm1 = 0;
	for (size_t i = 0; i < vec1.size(); i++){
		norm1 += vec1[i] * vec1[i];
	}
	norm1 = sqrt(norm1);
	return norm1;
}

float cos_similarity(std::vector<float> & vec1, std::vector<float> & vec2){
	float sum = 0;
	float norm1 = norm(vec1);
	float norm2 = norm(vec2);
	for (size_t i = 0; i < vec1.size(); i++){
		sum += (vec1[i] / norm1) * (vec2[i] / norm2);
	}
	return sum;
}

static int recognize_from_text(AILIANetwork* net, struct AILIATokenizer *tokenizer)
{
	// unused: int status = AILIA_STATUS_SUCCESS;

	// Open database
	std::vector<std::string> texts = open_texts(std::string("sample.txt"));

	// Embedding
	std::vector< std::vector<float> > embeddings;
	PRINT_OUT("Calculating embeddings\n");
	for (size_t i = 0; i < texts.size(); i++){
		PRINT_OUT("\r%zu/%zu", i, texts.size());
		fflush(stdout);
		std::vector<float> embedding = calc_embedding(net, tokenizer, texts[i], false);
		embeddings.push_back(embedding);
	}
	PRINT_OUT("\n");

	// Embedding Query
	std::vector<float> query_embedding = calc_embedding(net, tokenizer, input_text, true);
	if (debug){
		PRINT_OUT("Query norm %f\n", norm(query_embedding));
	}

	// Search
	float max_score = 0.0f;
	int max_i = 0;
	for (size_t i = 0; i < texts.size(); i++){
		float score = cos_similarity(query_embedding, embeddings[i]);
		if (debug){
			PRINT_OUT("%f ",score);
		}
		if (max_score < score){
			max_score = score;
			max_i = i;
		}
	}

	PRINT_OUT("Query : %s\n", input_text.c_str());
	PRINT_OUT("Result : %s\n", texts[max_i].c_str());
	PRINT_OUT("Similarity : %f\n", max_score);
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
	AILIANetwork *ailia;
	status = ailiaCreate(&ailia, env_id, AILIA_MULTITHREAD_AUTO);
	if (status != AILIA_STATUS_SUCCESS) {
		PRINT_ERR("ailiaCreate failed %d\n", status);
		if (status == AILIA_STATUS_LICENSE_NOT_FOUND || status==AILIA_STATUS_LICENSE_EXPIRED){
			PRINT_OUT("License file not found or expired : please place license file (AILIA.lic)\n");
		}
		return -1;
	}

	AILIAEnvironment *env_ptr = nullptr;
	status = ailiaGetSelectedEnvironment(ailia, &env_ptr, AILIA_ENVIRONMENT_VERSION);
	if (status != AILIA_STATUS_SUCCESS) {
		PRINT_ERR("ailiaGetSelectedEnvironment failed %d\n", status);
		ailiaDestroy(ailia);
		return -1;
	}

	PRINT_OUT("selected env name : %s\n", env_ptr->name);

	status = ailiaOpenStreamFile(ailia, model.c_str());
	if (status != AILIA_STATUS_SUCCESS) {
		PRINT_ERR("ailiaOpenStreamFile failed %d\n", status);
		PRINT_ERR("ailiaGetErrorDetail %s\n", ailiaGetErrorDetail(ailia));
		ailiaDestroy(ailia);
		return -1;
	}

	status = ailiaOpenWeightFile(ailia, weight.c_str());
	if (status != AILIA_STATUS_SUCCESS) {
		PRINT_ERR("ailiaOpenWeightFile failed %d\n", status);
		ailiaDestroy(ailia);
		return -1;
	}

	AILIATokenizer *tokenizer;
	status = ailiaTokenizerCreate(&tokenizer, AILIA_TOKENIZER_TYPE_XLM_ROBERTA, AILIA_TOKENIZER_FLAG_NONE);
	if (status != 0){
		printf("ailiaTokenizerCreate error %d\n", status);
		return -1;
	}
	status = ailiaTokenizerOpenModelFile(tokenizer, "sentencepiece.bpe.model");
	if (status != 0){
		printf("ailiaTokenizerOpenModelFile error %d\n", status);
		return -1;
	}

	status = recognize_from_text(ailia, tokenizer);

	ailiaTokenizerDestroy(tokenizer);

	ailiaDestroy(ailia);

	return status;
}
