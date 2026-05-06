/*******************************************************************
*
*    DESCRIPTION:
*      AILIA G2P EN averaged perceptron
*    AUTHOR:
*
*    DATE:2024/06/26
*
*******************************************************************/

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <algorithm>
#include "g2p_en_averaged_perceptron.h"
#include "g2p_en_file.h"

using namespace std;

namespace ailiaG2P{

string AveragedPerceptron::predict(const unordered_map<string, int>& features) {
	unordered_map<string, double> scores;
	for (const auto& feat : features) {
		if (weights.find(feat.first) == weights.end() || feat.second == 0) continue;
		const unordered_map<string, double>& label_weights = weights[feat.first];
		for (const auto& lw : label_weights) {
			scores[lw.first] += feat.second * lw.second;
		}
	}
	string best_label = *max_element(classes.begin(), classes.end(), [&](const string& a, const string& b) {
		return scores[a] < scores[b] || (scores[a] == scores[b] && a > b);
	});
	return best_label;
}

void AveragedPerceptron::import_from_text(const char *weight_a, const wchar_t *weight_w, const char *tagdict_a, const wchar_t *tagdict_w, const char *classes_a, const wchar_t *classes_w) {
	std::vector<char> buffer_weight;
	if (weight_w == NULL){
		buffer_weight = load_file_a(weight_a);
	}else{
		buffer_weight = load_file_w(weight_w);
	}
	std::istringstream weights_file(std::string(buffer_weight.begin(), buffer_weight.end()));

	string line, feat, label;
	double weight;
	while (getline(weights_file, feat)) {
		getline(weights_file, label);
		getline(weights_file, line);
		istringstream iss(line);
		iss >> weight;
		weights[feat][label] = weight;
	}

	std::vector<char> buffer_tagdict;
	if (tagdict_w == NULL){
		buffer_tagdict = load_file_a(tagdict_a);
	}else{
		buffer_tagdict = load_file_w(tagdict_w);
	}
	std::istringstream tagdict_file(std::string(buffer_tagdict.begin(), buffer_tagdict.end()));

	string tag, v;
	while (getline(tagdict_file, tag)) {
		getline(tagdict_file, v);
		tagdict[tag] = v;
	}

	std::vector<char> buffer_classes;
	if (classes_w == NULL){
		buffer_classes = load_file_a(classes_a);
	}else{
		buffer_classes = load_file_w(classes_w);
	}
	std::istringstream classes_file(std::string(buffer_classes.begin(), buffer_classes.end()));

	while (getline(classes_file, line)) {
		classes.insert(line);
	}
}

unordered_map<string, int> AveragedPerceptron::_get_features(int i, const string& word, const vector<string>& context, const string& prev, const string& prev2) {
	unordered_map<string, int> features;

	auto add = [&features](const string& name, const vector<string>& args = {}) {
		string key = name;
		for (const auto& arg : args) key += " " + arg;
		features[key]++;
	};

	i += (int)START.size();
	add("bias");
	if (word.size() >= 3)
		add("i suffix", { word.substr(word.size() - 3) });
	else
		add("i suffix", { word });
	
	add("i pref1", { word.empty() ? "" : std::string(1, word[0]) });
	add("i-1 tag", { prev });
	add("i-2 tag", { prev2 });
	add("i tag+i-2 tag", { prev, prev2 });
	
	if (i < (int)context.size()) add("i word", { context[i] });
	if (i < (int)context.size()) add("i-1 tag+i word", { prev, context[i] });
	if (i - 1 >= 0 && i - 1 < (int)context.size()) add("i-1 word", { context[i - 1] });
	if (i - 1 >= 0 && i - 1 < (int)context.size() && context[i - 1].size() >= 3) 
		add("i-1 suffix", { context[i - 1].substr(context[i - 1].size() - 3) });
	else if (i - 1 >= 0 && i - 1 < (int)context.size())
		add("i-1 suffix", { context[i - 1] });
	
	if (i - 2 >= 0 && i - 2 < (int)context.size()) add("i-2 word", { context[i - 2] });
	if (i + 1 < (int)context.size()) add("i+1 word", { context[i + 1] });
	if (i + 1 < (int)context.size() && context[i + 1].size() >= 3) 
		add("i+1 suffix", { context[i + 1].substr(context[i + 1].size() - 3) });
	else if (i + 1 < (int)context.size())
		add("i+1 suffix", { context[i + 1] });

	return features;
}

string AveragedPerceptron::normalize(const string& word) {
	if (word.find('-') != string::npos && word[0] != '-') {
		return "!HYPHEN";
	}
	if (all_of(word.begin(), word.end(), ::isdigit) && word.size() == 4) {
		return "!YEAR";
	}
	if (!word.empty() && ::isdigit(word[0])) {
		return "!DIGITS";
	}
	string lower_word;
	transform(word.begin(), word.end(), back_inserter(lower_word), ::tolower);
	return lower_word;
}

vector<pair<string, string>> AveragedPerceptron::tag(const vector<string>& tokens) {
	string prev = START[0], prev2 = START[1];
	vector<pair<string, string>> output;

	vector<string> context = START;
	for (const auto& w : tokens) {
		context.push_back(normalize(w));
	}
	context.insert(context.end(), END.begin(), END.end());

	for (size_t i = 0; i < tokens.size(); ++i) {
		string word = tokens[i];
		string tag;
		if (tagdict.find(word) != tagdict.end()) {
			tag = tagdict.at(word);
		} else {
			auto features = _get_features((int)i, word, context, prev, prev2);
			tag = predict(features);
		}
		output.push_back({ word, tag });

		prev2 = prev;
		prev = tag;
	}

	return output;
}

void test_averaged_perceptron() {
	AveragedPerceptron model;

	model.import_from_text("averaged_perceptron_tagger_weights.txt", NULL, "averaged_perceptron_tagger_tagdict.txt", NULL, "averaged_perceptron_tagger_classes.txt", NULL);

	vector<string> words = { "i'm", "an", "activationist", "." };
	auto output = model.tag(words);

	vector<pair<string, string>> expect;
	expect.push_back(pair<string, string>({"i'm", "VB"}));
	expect.push_back(pair<string, string>({"an", "DT"}));
	expect.push_back(pair<string, string>({"activationist", "NN"}));
	expect.push_back(pair<string, string>({".", "."}));

	for (size_t i = 0; i < expect.size(); i++){
		if (output[i] != expect[i]){
			for (const auto& pair : output) {
				cout << "(" << pair.first << ", " << pair.second << "), ";
			}
			cout << endl;

			throw("verify error at test_averaged_perceptron");
		}
	}
}

}
