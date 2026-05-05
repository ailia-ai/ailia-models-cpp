/*******************************************************************
*
*    DESCRIPTION:
*      AILIA G2P EN model
*    AUTHOR:
*
*    DATE:2024/06/26
*
*******************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <string>
#include <math.h>
#include <chrono>
#include <unordered_map>
#include <unordered_set>
#include <regex>
#include <iostream>
#include <map>
#include <fstream>
#include <sstream>

#undef UNICODE

#include "ailia.h"

#include "g2p_en_averaged_perceptron.h"
#include "g2p_en_expand.h"
#include "g2p_en_model.h"
#include "g2p_en_file.h"

namespace ailiaG2P {

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


// ======================
// ailia functions
// ======================

void checkError(int status, const char *func){
	if (status != 0){
		throw(func);
	}
}

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
		status = ailiaGetBlobIndexByInputIndex(ailia, &input_blob_idx, i);
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

		status = ailiaSetInputBlobData(ailia, &(inputs[i]->data)[0], inputs[i]->data.size() * sizeof(float), input_blob_idx);
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
		status = ailiaGetBlobIndexByOutputIndex(ailia, &output_blob_idx, i);
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

		status = ailiaGetBlobData(ailia, &ref_tensor.data[0], ref_tensor.data.size() * sizeof(float), output_blob_idx);
		if (status != AILIA_STATUS_SUCCESS) {
			setErrorDetail("ailiaGetBlobData",ailiaGetErrorDetail(ailia));
		}
	}
}

// ======================
// dictionary functions
// ======================

std::string toLowerCase(const std::string &text);

std::unordered_map<std::string, std::tuple<std::vector<std::string>, std::vector<std::string>, std::string>> construct_homograph_dictionary(const char * path_a, const wchar_t * path_w) {
	std::unordered_map<std::string, std::tuple<std::vector<std::string>, std::vector<std::string>, std::string>> homograph2features;

	std::vector<char> buffer;
	if (path_w == NULL){
		buffer = load_file_a(path_a);
	}else{
		buffer = load_file_w(path_w);
	}
	std::istringstream file(std::string(buffer.begin(), buffer.end()));

	std::string line;
	while (std::getline(file, line)) {
		if (line[0] == '#') continue;

		std::vector<std::string> parts;
		std::stringstream ss(line);
		std::string part;
		while (std::getline(ss, part, '|')) {
			parts.push_back(part);
		}

		std::string headword = toLowerCase(parts[0]);
		std::vector<std::string> pron1;
		std::vector<std::string> pron2;
		std::string pron1_str = parts[1];
		std::string pron2_str = parts[2];
		std::string pos1 = parts[3];

		std::stringstream pron1_ss(pron1_str);
		std::stringstream pron2_ss(pron2_str);
		std::string token;
		while (pron1_ss >> token) {
			pron1.push_back(token);
		}
		while (pron2_ss >> token) {
			pron2.push_back(token);
		}

		homograph2features[headword] = std::make_tuple(pron1, pron2, pos1);
	}
	return homograph2features;
}

std::unordered_map<std::string, std::vector<std::string>> construct_cmu_dictionary(const char * path_a, const wchar_t * path_w) {
	std::unordered_map<std::string, std::vector<std::string>> cmudict;

	std::vector<char> buffer;
	if (path_w == NULL){
		buffer = load_file_a(path_a);
	}else{
		buffer = load_file_w(path_w);
	}
	std::istringstream file(std::string(buffer.begin(), buffer.end()));

	std::string line;
	while (std::getline(file, line)) {
		if (line[0] == '#') continue;

		std::vector<std::string> lists;
		std::stringstream ss(line);
		std::string part;
		while (ss >> part) {
			lists.push_back(part);
		}

		std::string headword = toLowerCase(lists[0]);
		std::vector<std::string> pron(lists.begin() + 2, lists.end());
		if (cmudict.find(headword) == cmudict.end()) {
			cmudict[headword] = pron;
		}
	}
	return cmudict;
}

// ======================
// normalize functions
// ======================

std::string toLowerCase(const std::string &text) {
	std::string result = text;
	std::transform(result.begin(), result.end(), result.begin(), ::tolower);
	return result;
}

std::string regexReplace(const std::string &text, const std::regex &pattern, const std::string &replacement) {
	return std::regex_replace(text, pattern, replacement);
}

std::vector<std::string> split(const std::string &text) {
	std::vector<std::string> words;
	std::string word;
	for (char ch : text) {
		if (isspace(ch)) {
			if (!word.empty()) {
				words.push_back(word);
				word.clear();
			}
		} else {
			word += ch;
		}
	}
	if (!word.empty()) {
		words.push_back(word);
	}
	return words;
}

std::vector<int> tokenize(const std::string& word) {
	std::vector<std::string> graphemes;
	graphemes = {"<pad>", "<unk>", "</s>",
					"a", "b", "c", "d", "e", "f", "g",
					"h", "i", "j", "k", "l", "m",
					"n", "o", "p", "q", "r", "s",
					"t", "u", "v", "w", "x", "y", "z"};
	
	std::map<std::string, int> g2idx;
	for(size_t i = 0; i < graphemes.size(); ++i) {
		g2idx[graphemes[i]] = i;
	}

	std::vector<int> x;
	for (const auto& c : word) {
		std::string s(1, c);
		if (g2idx.find(s) != g2idx.end()) {
			x.push_back(g2idx[s]);
		} else {
			x.push_back(g2idx["<unk>"]);
		}
	}
	x.push_back(g2idx["</s>"]);
	return x;
}

// ======================
// predict functions
// ======================

std::vector<std::string> get_phonemes() {
	return {"<pad>", "<unk>", "<s>", "</s>", "AA0", "AA1", "AA2", "AE0", "AE1", "AE2", "AH0", "AH1", "AH2", "AO0", "AO1",
			"AO2", "AW0", "AW1", "AW2", "AY0", "AY1", "AY2", "B", "CH", "D", "DH", "EH0", "EH1", "EH2", "ER0", "ER1", "ER2",
			"EY0", "EY1", "EY2", "F", "G", "HH", "IH0", "IH1", "IH2", "IY0", "IY1", "IY2", "JH", "K", "L", "M", "N", "NG",
			"OW0", "OW1", "OW2", "OY0", "OY1", "OY2", "P", "R", "S", "SH", "T", "TH", "UH0", "UH1", "UH2", "UW", "UW0", 
			"UW1", "UW2", "V", "W", "Y", "Z", "ZH"};
}

std::unordered_map<int, std::string> get_idx2p(const std::vector<std::string>& phonemes) {
	std::unordered_map<int, std::string> idx2p;
	for (size_t i = 0; i < phonemes.size(); ++i) {
		idx2p[i] = phonemes[i];
	}
	return idx2p;
}

std::vector<std::string> preds_to_phonemes(const std::vector<int>& preds, const std::unordered_map<int, std::string>& idx2p) {
	std::vector<std::string> phoneme_preds;
	for (int idx : preds) {
		if (idx2p.find(idx) != idx2p.end()) {
			phoneme_preds.push_back(idx2p.at(idx));
		} else {
			phoneme_preds.push_back("<unk>");
		}
	}
	return phoneme_preds;
}

std::vector<std::string> G2PEnModel::predict(const std::string &word){
	std::vector<int> x = tokenize(word);
	if (debug){
		PRINT_OUT("tokens : ");
		for (size_t i = 0; i < x.size(); i++){
			PRINT_OUT("%d ", x[i]);
		}
		PRINT_OUT("\n");
	}

	std::vector<float> h_data(256);

	AILIATensor h_tensor;
	h_tensor.data = h_data;
	h_tensor.shape.x = 256;
	h_tensor.shape.y = 1;
	h_tensor.shape.z = 1;
	h_tensor.shape.w = 1;
	h_tensor.shape.dim = 2;

	for (size_t i = 0; i < x.size(); i++){
		std::vector<float> x_data(1);
		x_data[0] = x[i];

		AILIATensor x_tensor;
		x_tensor.data = x_data;
		x_tensor.shape.x = 1;
		x_tensor.shape.y = 1;
		x_tensor.shape.z = 1;
		x_tensor.shape.w = 1;
		x_tensor.shape.dim = 1;

		std::vector<AILIATensor*> encoder_inputs;
		encoder_inputs.push_back(&x_tensor);
		encoder_inputs.push_back(&h_tensor);

		std::vector<AILIATensor> encoder_outputs;
		forward(net[MODEL_ENCODER], encoder_inputs, encoder_outputs);

		h_tensor = encoder_outputs[0];
	}

	std::vector<int> preds;
	int pred = 2;

	for (int i = 0; i < 20; i++){
		std::vector<float> pred_data(1);
		pred_data[0] = pred;

		AILIATensor pred_tensor;
		pred_tensor.data = pred_data;
		pred_tensor.shape.x = pred_data.size();
		pred_tensor.shape.y = 1;
		pred_tensor.shape.z = 1;
		pred_tensor.shape.w = 1;
		pred_tensor.shape.dim = 1;

		std::vector<AILIATensor*> decoder_inputs;
		decoder_inputs.push_back(&pred_tensor);
		decoder_inputs.push_back(&h_tensor);

		std::vector<AILIATensor> decoder_outputs;
		forward(net[MODEL_DECODER], decoder_inputs, decoder_outputs);

		AILIATensor logits_tensor = decoder_outputs[0];
		h_tensor = decoder_outputs[1];

		float max_logits = -1;
		for (size_t i = 0; i < logits_tensor.shape.x; i++){
			if (max_logits < logits_tensor.data[i]){
				max_logits = logits_tensor.data[i];
				pred = i;
			}
		}

		if (pred == 3){
			break;
		}

		preds.push_back(pred);
	}

	if (debug){
		PRINT_OUT("output\n");
		for (size_t i = 0; i < preds.size(); i++){
			PRINT_OUT("%d ", preds[i]);
		}
		PRINT_OUT("\n");
	}

	std::vector<std::string> phonemes = get_phonemes();
	std::unordered_map<int, std::string> idx2p = get_idx2p(phonemes);
	std::vector<std::string> phoneme_preds = preds_to_phonemes(preds, idx2p);
	return phoneme_preds;
}

// ======================
// main functions
// ======================

void G2PEnModel::open(int env_id, const char *model_encoder_a, const wchar_t *model_encoder_w, const char *model_decoder_a, const wchar_t *model_decoder_w, const char *homograph_a, const wchar_t *homograph_w, const char *cmudict_a, const wchar_t *cmudict_w)
{
	int status;

	for (int i = 0; i < MODEL_N; i++){
		status = ailiaCreate(&net[i], env_id, AILIA_MULTITHREAD_AUTO);
		checkError(status, "ailiaCreate");

		status = ailiaSetMemoryMode(net[i], AILIA_MEMORY_OPTIMAIZE_DEFAULT | AILIA_MEMORY_REUSE_INTERSTAGE);
		checkError(status, "ailiaSetMemoryMode");

		AILIAEnvironment *env_ptr = nullptr;
		status = ailiaGetSelectedEnvironment(net[i], &env_ptr, AILIA_ENVIRONMENT_VERSION);
		checkError(status, "ailiaGetSelectedEnvironment");

		PRINT_OUT("selected env name : %s\n", env_ptr->name);

		if (i == 0){
			if (model_encoder_w == NULL){
				status = ailiaOpenWeightFileA(net[i], model_encoder_a);
			}else{
				status = ailiaOpenWeightFileW(net[i], model_encoder_w);
			}
		}else{
			if (model_decoder_w == NULL){
				status = ailiaOpenWeightFileA(net[i], model_decoder_a);
			}else{
				status = ailiaOpenWeightFileW(net[i], model_decoder_w);
			}
		}
		checkError(status, "ailiaOpenWeightFile");
	}

	homograph2features = construct_homograph_dictionary(homograph_a, homograph_w);
	cmudict = construct_cmu_dictionary(cmudict_a, cmudict_w);
}

void G2PEnModel::close()
{
	for (int i = 0; i < MODEL_N; i++){
		ailiaDestroy(net[i]);
	}
}

void G2PEnModel::import_from_text(const char *weight_a, const wchar_t *weight_w, const char *tagdict_a, const wchar_t *tagdict_w, const char *classes_a, const wchar_t *classes_w){
	model.import_from_text(weight_a, weight_w, tagdict_a, tagdict_w, classes_a, classes_w);
}

std::vector<std::string> G2PEnModel::compute(std::string text)
{
	text = normalize_numbers(text);

	text = toLowerCase(text);
	text = regexReplace(text, std::regex("[^ a-z'.,?!\\-]"), "");
	text = regexReplace(text, std::regex("i\\.e\\."), "that is");
	text = regexReplace(text, std::regex("e\\.g\\."), "for example");
	std::string text2 = text;
	text2 = regexReplace(text2, std::regex("\\."), " . ");
	text2 = regexReplace(text2, std::regex(","), " , ");
	text2 = regexReplace(text2, std::regex("!"), " ! ");
	text2 = regexReplace(text2, std::regex("\\?"), " ? ");

	std::vector<std::string> words = split(text2);

    std::vector<std::pair<std::string, std::string>> tokens = model.tag(words);

	std::vector<std::string> prons;
	for (const auto& token : tokens) {
		std::string word = token.first;
		std::string pos = token.second;
		std::vector<std::string> pron;

		if (!std::regex_search(word, std::regex("[a-z]"))) {
			pron.push_back(word);
		} else if (homograph2features.find(word) != homograph2features.end()) {
			std::tuple<std::vector<std::string>, std::vector<std::string>, std::string> data = homograph2features[word];
			std::vector<std::string> pron1 = std::get<0>(data);
			std::vector<std::string> pron2 = std::get<1>(data);
			std::string pos1 = std::get<2>(data);
			if (pos.find(pos1) == 0) {
				pron = pron1;
			} else {
				pron = pron2;
			}
		} else if (cmudict.find(word) != cmudict.end()) {
			pron = cmudict[word];
		} else {
			pron = predict(word);
		}

		for (size_t i = 0; i < pron.size(); i++) {
			prons.push_back(pron[i]);
		}
		prons.push_back(" ");
	}
	if (prons.size() > 0){
		prons.pop_back();
	}
	return prons;
}

}
