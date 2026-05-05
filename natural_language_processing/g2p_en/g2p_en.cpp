/*******************************************************************
*
*    DESCRIPTION:
*      AILIA G2P EN sample
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
#include <chrono>

#undef UNICODE

#include "ailia.h"
#include "g2p_en_model.h"
#include "g2p_en_expand.h"
#include "g2p_en_averaged_perceptron.h"

using namespace ailiaG2P;

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

static bool benchmark  = false;
static bool verify  = false;
static int args_env_id = -1;

std::string reference_text = "To be or not to be, that is the questionary";

// ======================
// Arguemnt Parser
// ======================

static void print_usage()
{
	PRINT_OUT("usage: g2p_en [-h] [-i TEXT] [-b] [-e ENV_ID]\n");
	return;
}


static void print_help()
{
	PRINT_OUT("\n");
	PRINT_OUT("g2p_en model\n");
	PRINT_OUT("\n");
	PRINT_OUT("optional arguments:\n");
	PRINT_OUT("  -h, --help            show this help message and exit\n");
	PRINT_OUT("  -i FILE, --input FILE\n");
	PRINT_OUT("                        The input file.\n");
	PRINT_OUT("  -b, --benchmark       Running the inference on the same input 5 times to\n");
	PRINT_OUT("                        measure execution performance. (Cannot be used in\n");
	PRINT_OUT("                        video mode)\n");
	PRINT_OUT("  -v, --verify          Check model output\n");
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
			else if (arg == "-v" || arg == "--verify") {
				verify = true;
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
				reference_text = std::string(arg);
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

void verify_output(std::vector<std::string> prons, std::vector<std::string> expect){
	if (expect.size() > 0){
		if (expect.size() != prons.size()){
			PRINT_OUT("Invalid out token length %zu vs %zu\n", expect.size(), prons.size());
			throw("verify error");
		}
		for (size_t i = 0; i < prons.size(); i++){
			if (expect[i] != prons[i]){
				PRINT_OUT("Output token mismatch %s vs %s at %zu\n", expect[i].c_str(), prons[i].c_str(), i);
				throw("verify error");
			}
		}
	}
	PRINT_OUT("verify success\n");
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
		//bool is_fp16 = (env->props & AILIA_ENVIRONMENT_PROPERTY_FP16) != 0;
		PRINT_OUT("env_id : %d type : %d name : %s", env->id, env->type, env->name);
		//if (is_fp16){
		//	PRINT_OUT(" (Warning : FP16 backend is not worked this model)\n");
		//	continue;
		//}
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

	G2PEnModel model = G2PEnModel();
	model.open(args_env_id, "g2p_encoder.onnx", NULL, "g2p_decoder.onnx", NULL, "homographs.en", NULL, "cmudict", NULL);
	
	model.import_from_text("averaged_perceptron_tagger_weights.txt", NULL, "averaged_perceptron_tagger_tagdict.txt", NULL, "averaged_perceptron_tagger_classes.txt", NULL);

	if (verify){
		verify_output(model.compute("I have $250 in my pocket."), {"AY1", " ", "HH", "AE1", "V", " ", "T", "UW1", " ", "HH", "AH1", "N", "D", "R", "AH0", "D", " ", "F", "IH1", "F", "T", "IY0", " ", "D", "AA1", "L", "ER0", "Z", " ", "IH0", "N", " ", "M", "AY1", " ", "P", "AA1", "K", "AH0", "T", " ", "."});
		verify_output(model.compute("popular pets, e.g. cats and dogs"), {"P", "AA1", "P", "Y", "AH0", "L", "ER0", " ", "P", "EH1", "T", "S", " ", ",", " ", "F", "AO1", "R", " ", "IH0", "G", "Z", "AE1", "M", "P", "AH0", "L", " ", "K", "AE1", "T", "S", " ", "AH0", "N", "D", " ", "D", "AA1", "G", "Z"});
		verify_output(model.compute("I refuse to collect the refuse around here."), {"AY1", " ", "R", "IH0", "F", "Y", "UW1", "Z", " ", "T", "UW1", " ", "K", "AH0", "L", "EH1", "K", "T", " ", "DH", "AH0", " ", "R", "EH1", "F", "Y", "UW2", "Z", " ", "ER0", "AW1", "N", "D", " ", "HH", "IY1", "R", " ", "."});
		verify_output(model.compute("I'm an activationist."), {"AY1", "M", " ", "AE1", "N", " ", "AE2", "K", "T", "IH0", "V", "EY1", "SH", "AH0", "N", "IH0", "S", "T", " ", "."});
	}else{
		PRINT_OUT("Input : \n");
		PRINT_OUT("%s\n", reference_text.c_str());

		auto start2 = std::chrono::high_resolution_clock::now();
		std::vector<std::string> prons = model.compute(reference_text);
		auto end2 = std::chrono::high_resolution_clock::now();
		if (benchmark){
			PRINT_OUT("total processing time %lld ms\n",  (long long)std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count());
		}

		PRINT_OUT("Output :\n");
		for (size_t i = 0; i < prons.size(); i++){
			PRINT_OUT("%s ", prons[i].c_str());
		}
		PRINT_OUT("\n");
	}

	PRINT_OUT("Program finished successfully.\n");

	model.close();

	// unit test
	if (verify){
		test_expand();
		test_averaged_perceptron();
	}

	return status;
}
