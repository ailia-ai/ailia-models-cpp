/*******************************************************************
*
*    DESCRIPTION:
*      AILIA fugumt sample
*    AUTHOR:
*
*    DATE:2023/06/14
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

#define ENCODE_WEIGHT_PATH "encoder_model.onnx"
#define ENCODE_MODEL_PATH "encoder_model.onnx.prototxt"
#define DECODE_WEIGHT_PATH "decoder_model.onnx"
#define DECODE_MODEL_PATH "decoder_model.onnx.prototxt"

#if defined(_WIN32) || defined(_WIN64)
#define PRINT_OUT(...) fprintf_s(stdout, __VA_ARGS__)
#define PRINT_ERR(...) fprintf_s(stderr, __VA_ARGS__)
#else
#define PRINT_OUT(...) fprintf(stdout, __VA_ARGS__)
#define PRINT_ERR(...) fprintf(stderr, __VA_ARGS__)
#endif

#define BENCHMARK_ITERS 5

#define ENCODER_NUM_INPUTS 2
#define ENCODER_NUM_OUTPUTS 1

#define DECODER_NUM_INPUTS 27
#define DECODER_NUM_OUTPUTS 25
#define DECODER_NUM_PAST_KEY 24


static std::string encoder_weight(ENCODE_WEIGHT_PATH);
static std::string encoder_model(ENCODE_MODEL_PATH);
static std::string decoder_weight(DECODE_WEIGHT_PATH);
static std::string decoder_model(DECODE_MODEL_PATH);

static bool benchmark  = false;
static int args_env_id = -1;

std::string input_text = "これは猫です";

#define MAX_LENGTH 512


// ======================
// Arguemnt Parser
// ======================

static void print_usage()
{
	PRINT_OUT("usage: fugumt [-h] [-i TEXT] [-b] [-e ENV_ID]\n");
	return;
}


static void print_help()
{
	PRINT_OUT("\n");
	PRINT_OUT("fugumt model\n");
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


int ailia_encode(AILIANetwork *ailia_encoder,
    std::vector<float> *encoder_inputs[ENCODER_NUM_INPUTS], std::vector<float> *encoder_outputs[ENCODER_NUM_OUTPUTS]) {
    int status;
    
    for (int i = 0; i < ENCODER_NUM_INPUTS; i++){
		unsigned int encoder_input_blob_idx = 0;
		status = ailiaGetBlobIndexByInputIndex(ailia_encoder, &encoder_input_blob_idx, i);
		if (status != AILIA_STATUS_SUCCESS) {
			setErrorDetail("ailiaGetBlobIndexByInputIndex", ailiaGetErrorDetail(ailia_encoder));
			return status;
		}

        AILIAShape encoder_input_blob_shape;
		int batch_size = 1;
        encoder_input_blob_shape.x=encoder_inputs[i]->size();
        encoder_input_blob_shape.y=batch_size;
        encoder_input_blob_shape.z=1;
        encoder_input_blob_shape.w=1;
        encoder_input_blob_shape.dim=2;

        if (debug){
			PRINT_OUT("encoder input blob shape %d %d %d %d dims %d\n",encoder_input_blob_shape.x,encoder_input_blob_shape.y,encoder_input_blob_shape.z,encoder_input_blob_shape.w,encoder_input_blob_shape.dim);
		}

        status = ailiaSetInputBlobShape(ailia_encoder, &encoder_input_blob_shape, encoder_input_blob_idx, AILIA_SHAPE_VERSION);
		if(status!=AILIA_STATUS_SUCCESS){
			setErrorDetail("ailiaSetInputBlobShape",ailiaGetErrorDetail(ailia_encoder));
			return status;
		}

        if (encoder_inputs[i]->size() > 0){
			status = ailiaSetInputBlobData(ailia_encoder, &(*encoder_inputs[i])[0], encoder_inputs[i]->size() * sizeof(float), encoder_input_blob_idx);
			if (status != AILIA_STATUS_SUCCESS) {
				setErrorDetail("ailiaSetInputBlobData",ailiaGetErrorDetail(ailia_encoder));
				return status;
			}
		}
    }

    status = ailiaUpdate(ailia_encoder);
	if (status != AILIA_STATUS_SUCCESS) {
		setErrorDetail("ailiaUpdate",ailiaGetErrorDetail(ailia_encoder));
		return status;
	}

    for (int i = 0; i < ENCODER_NUM_OUTPUTS; i++){
		unsigned int output_blob_idx = 0;
		status = ailiaGetBlobIndexByOutputIndex(ailia_encoder, &output_blob_idx, i);
		if (status != AILIA_STATUS_SUCCESS) {
			setErrorDetail("ailiaGetBlobIndexByInputIndex",ailiaGetErrorDetail(ailia_encoder));
			return status;
		}

        AILIAShape output_blob_shape;
		status=ailiaGetBlobShape(ailia_encoder, &output_blob_shape, output_blob_idx, AILIA_SHAPE_VERSION);
		if(status!=AILIA_STATUS_SUCCESS){
			setErrorDetail("ailiaGetBlobShape", ailiaGetErrorDetail(ailia_encoder));
			return status;
		}

        if (debug){
			PRINT_OUT("encoder output_blob_shape %d %d %d %d dims %d\n",output_blob_shape.x,output_blob_shape.y,output_blob_shape.z,output_blob_shape.w,output_blob_shape.dim);
		}

        (*encoder_outputs[i]).resize(output_blob_shape.x*output_blob_shape.y*output_blob_shape.z*output_blob_shape.w);

        status =ailiaGetBlobData(ailia_encoder, &(*encoder_outputs[i])[0], encoder_outputs[i]->size() * sizeof(float), output_blob_idx);
		if (status != AILIA_STATUS_SUCCESS) {
			setErrorDetail("ailiaGetBlobData",ailiaGetErrorDetail(ailia_encoder));
			return status;
		}
    }

    return AILIA_STATUS_SUCCESS;
}


int ailia_decode(AILIANetwork *ailia_decoder, 
    std::vector<float> *encoder_inputs[ENCODER_NUM_INPUTS], std::vector<float> *encoder_outputs[ENCODER_NUM_OUTPUTS],
    std::vector<float> *decoder_inputs[DECODER_NUM_INPUTS], std::vector<float> *decoder_outputs[DECODER_NUM_OUTPUTS]) {
    int status;

    for (int i = 0; i < DECODER_NUM_INPUTS; i++){
        unsigned int decoder_input_blob_idx = 0;
        status = ailiaGetBlobIndexByInputIndex(ailia_decoder, &decoder_input_blob_idx, i);
        if (status != AILIA_STATUS_SUCCESS) {
            setErrorDetail("ailiaGetBlobIndexByInputIndex", ailiaGetErrorDetail(ailia_decoder));
            return status;
        }

        AILIAShape decoder_input_blob_shape;
        int batch_size = 1;
        if (i == 0){
            decoder_input_blob_shape.x=encoder_inputs[1]->size(); //encoderのattention_maskのshape
            decoder_input_blob_shape.y=batch_size;
            decoder_input_blob_shape.z=1;
            decoder_input_blob_shape.w=1;
            decoder_input_blob_shape.dim=2;
        }
        else if (i == 1){
            decoder_input_blob_shape.x=decoder_inputs[i]->size();
            decoder_input_blob_shape.y=batch_size;
            decoder_input_blob_shape.z=1;
            decoder_input_blob_shape.w=1;
            decoder_input_blob_shape.dim=2;
        }
        else if (i == 2){
            decoder_input_blob_shape.x=512;
            decoder_input_blob_shape.y=encoder_outputs[0]->size()/512; //encoderのlast_hidden_stateのshape
            decoder_input_blob_shape.z=batch_size;
            decoder_input_blob_shape.w=1;
            decoder_input_blob_shape.dim=3;
        }
        else {
            decoder_input_blob_shape.x=64;
            decoder_input_blob_shape.y=decoder_inputs[i]->size()/64/8;
            decoder_input_blob_shape.z=8;
            decoder_input_blob_shape.w=batch_size;
            decoder_input_blob_shape.dim=4;
        }

        if (debug){
            PRINT_OUT("decoder input blob shape %d %d %d %d dims %d\n",decoder_input_blob_shape.x,decoder_input_blob_shape.y,decoder_input_blob_shape.z,decoder_input_blob_shape.w,decoder_input_blob_shape.dim);
        }

        status = ailiaSetInputBlobShape(ailia_decoder, &decoder_input_blob_shape, decoder_input_blob_idx, AILIA_SHAPE_VERSION);
        if(status!=AILIA_STATUS_SUCCESS){
            setErrorDetail("ailiaSetInputBlobShape",ailiaGetErrorDetail(ailia_decoder));
            return status;
        }

        if (i == 0){
            if (encoder_inputs[1]->size() > 0){
            status = ailiaSetInputBlobData(ailia_decoder, &(*encoder_inputs[1])[0], encoder_inputs[1]->size() * sizeof(float), decoder_input_blob_idx);
            }
        }
        else if (i == 1){
            if (decoder_inputs[1]->size() > 0){
                status = ailiaSetInputBlobData(ailia_decoder, &(*decoder_inputs[i])[0], decoder_inputs[i]->size() * sizeof(float), decoder_input_blob_idx);
            }
        }
        else if (i == 2){
            if (encoder_outputs[0]->size() > 0){
            status = ailiaSetInputBlobData(ailia_decoder, &(*encoder_outputs[0])[0], encoder_outputs[0]->size() * sizeof(float), decoder_input_blob_idx);
            }
        }
        else {
            if (decoder_inputs[i]->size() > 0){
            status = ailiaSetInputBlobData(ailia_decoder, &(*decoder_inputs[i])[0], decoder_inputs[i]->size() * sizeof(float), decoder_input_blob_idx);
            }
        }
        if (status != AILIA_STATUS_SUCCESS) {
            setErrorDetail("ailiaSetInputBlobData",ailiaGetErrorDetail(ailia_decoder));
            return status;
        }
    }

    status = ailiaUpdate(ailia_decoder);
    if (status != AILIA_STATUS_SUCCESS) {
        setErrorDetail("ailiaUpdate",ailiaGetErrorDetail(ailia_decoder));
        return status;
    }

    for (int i = 0; i < DECODER_NUM_OUTPUTS; i++){
        unsigned int decoder_output_blob_idx = 0;
        status = ailiaGetBlobIndexByOutputIndex(ailia_decoder, &decoder_output_blob_idx, i);
        if (status != AILIA_STATUS_SUCCESS) {
            setErrorDetail("ailiaGetBlobIndexByInputIndex",ailiaGetErrorDetail(ailia_decoder));
            return status;
        }

        AILIAShape decoder_output_blob_shape;
        status=ailiaGetBlobShape(ailia_decoder, &decoder_output_blob_shape, decoder_output_blob_idx, AILIA_SHAPE_VERSION);
        if(status!=AILIA_STATUS_SUCCESS){
            setErrorDetail("ailiaGetBlobShape", ailiaGetErrorDetail(ailia_decoder));
            return status;
        }

        if (debug){
            PRINT_OUT("output_blob_shape %d %d %d %d dims %d\n",decoder_output_blob_shape.x,decoder_output_blob_shape.y,decoder_output_blob_shape.z,decoder_output_blob_shape.w,decoder_output_blob_shape.dim);
        }

        (*decoder_outputs[i]).resize(decoder_output_blob_shape.x*decoder_output_blob_shape.y*decoder_output_blob_shape.z*decoder_output_blob_shape.w);

        status =ailiaGetBlobData(ailia_decoder, &(*decoder_outputs[i])[0], decoder_outputs[i]->size() * sizeof(float), decoder_output_blob_idx);
        if (status != AILIA_STATUS_SUCCESS) {
            setErrorDetail("ailiaGetBlobData",ailiaGetErrorDetail(ailia_decoder));
            return status;
        }

    }

    return AILIA_STATUS_SUCCESS;
}


static int recognize_from_text(AILIANetwork* encoder_net, AILIANetwork* decoder_net, struct AILIATokenizer *tokenizer_source, struct AILIATokenizer *tokenizer_target)
{
    int status = AILIA_STATUS_SUCCESS;
	int pad_token_id = 32000;

    PRINT_OUT("Input : %s\n", input_text.c_str());
    std::vector<int> tokens = encode(input_text, tokenizer_source);
	if (tokens.size() > MAX_LENGTH){
		tokens[MAX_LENGTH - 1] = tokens[tokens.size() - 1];
		tokens.resize(MAX_LENGTH);
	}

    //エンコーダーモデルの入力を定義
    std::vector<float> encoder_input_ids(tokens.size());
	std::vector<float> attention_mask(tokens.size());
	PRINT_OUT("Input Tokens :\n");
	for (size_t i = 0; i < tokens.size(); i++){
		encoder_input_ids[i] = (float)tokens[i];
		attention_mask[i] = 1;
		PRINT_OUT("%d ", (int)encoder_input_ids[i]);
	}
	PRINT_OUT("\n");

    //デコーダーモデルの入力を定義
    int num_beams = 1;
    std::vector<float> encoder_attention_mask;
	std::vector<float> decoder_input_ids(num_beams);
	std::vector<float> encorder_hidden_state;
    std::vector<float> past_key_values[DECODER_NUM_PAST_KEY];

    for (int i = 0; i < num_beams; i++){
		decoder_input_ids[i] = pad_token_id;
	}
	for (int i = 0; i < DECODER_NUM_PAST_KEY; i++){
		past_key_values[i].resize(num_beams * 8 * 0 * 64);
	}

    //エンコーダーモデルの入出力設定
    //入力
	std::vector<float> *encoder_inputs[ENCODER_NUM_INPUTS];
	encoder_inputs[0] = &encoder_input_ids;
	encoder_inputs[1] = &attention_mask;
    //出力
    std::vector<float> last_hidden_state;
	std::vector<float> *encoder_outputs[ENCODER_NUM_OUTPUTS];
	encoder_outputs[0] = &last_hidden_state;

    //デコーダーモデルの入出力設定
    //入力
    std::vector<float> *decoder_inputs[DECODER_NUM_INPUTS];
    decoder_inputs[0] = &encoder_attention_mask;
	decoder_inputs[1] = &decoder_input_ids;
	decoder_inputs[2] = &encorder_hidden_state;
	for (int i = 0; i < DECODER_NUM_PAST_KEY; i++){
		decoder_inputs[3 + i] = &past_key_values[i];
	}
    //出力
    std::vector<float> logits;
	std::vector<float> *decoder_outputs[DECODER_NUM_OUTPUTS];
	decoder_outputs[0] = &logits;
	for (int i = 0; i < DECODER_NUM_PAST_KEY; i++){
		decoder_outputs[1 + i] = &past_key_values[i];
	}


    status = ailia_encode(encoder_net, encoder_inputs, encoder_outputs);
    tokens.clear();
	while(tokens.size() < MAX_LENGTH){
        if (debug){
			std::string text = decode(tokens, tokenizer_target);
			PRINT_OUT("Loop %d %s\n", (int)tokens.size(), text.c_str());
		}

        status = ailia_decode(decoder_net, encoder_inputs, encoder_outputs, decoder_inputs, decoder_outputs);
        if (status != AILIA_STATUS_SUCCESS){
			return status;
		}

        logits[pad_token_id] = -INFINITY;

        int eos_token_id = 0;

        float prob = -INFINITY;
		int arg_max = 0;
		for (size_t i = 0; i < logits.size(); i++){
			//PRINT_OUT("%f ", logits[i]);
			if (prob < logits[i]){
				prob = logits[i];
				arg_max = i;
			}
		}

        if (debug){
			PRINT_OUT("Token %d (%f)\n", arg_max, prob);
		}

		tokens.push_back(arg_max);

        for (int i = 0; i < num_beams; i++){
			decoder_input_ids[i] = arg_max;
		}

		if (arg_max == eos_token_id){
			break;
		}
    }

    std::string text = decode(tokens, tokenizer_target);
	PRINT_OUT("Output : %s\n",text.c_str());

	PRINT_OUT("Output Tokens :\n");
	for (size_t i = 0; i < tokens.size(); i++){
		PRINT_OUT("%d ", tokens[i]);
	}
	PRINT_OUT("\n");

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

    // initialize encoder net
    AILIANetwork *ailia_encoder;
    {
        status = ailiaCreate(&ailia_encoder, env_id, AILIA_MULTITHREAD_AUTO);
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaCreate failed %d\n", status);
            if (status == AILIA_STATUS_LICENSE_NOT_FOUND || status==AILIA_STATUS_LICENSE_EXPIRED){
                PRINT_OUT("License file not found or expired : please place license file (AILIA.lic)\n");
            }
            return -1;
        }

        AILIAEnvironment *env_ptr = nullptr;
        status = ailiaGetSelectedEnvironment(ailia_encoder, &env_ptr, AILIA_ENVIRONMENT_VERSION);
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaGetSelectedEnvironment failed %d\n", status);
            ailiaDestroy(ailia_encoder);
            return -1;
        }

        PRINT_OUT("selected env name : %s\n", env_ptr->name);

        status = ailiaOpenStreamFile(ailia_encoder, encoder_model.c_str());
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaOpenStreamFile failed %d\n", status);
            PRINT_ERR("ailiaGetErrorDetail %s\n", ailiaGetErrorDetail(ailia_encoder));
            ailiaDestroy(ailia_encoder);
            return -1;
        }

        status = ailiaOpenWeightFile(ailia_encoder, encoder_weight.c_str());
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaOpenWeightFile failed %d\n", status);
            ailiaDestroy(ailia_encoder);
            return -1;
        }
    }


    // initialize decoder net
    AILIANetwork *ailia_decoder;
    {
        status = ailiaCreate(&ailia_decoder, env_id, AILIA_MULTITHREAD_AUTO);
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaCreate failed %d\n", status);
            if (status == AILIA_STATUS_LICENSE_NOT_FOUND || status==AILIA_STATUS_LICENSE_EXPIRED){
                PRINT_OUT("License file not found or expired : please place license file (AILIA.lic)\n");
            }
            return -1;
        }

        AILIAEnvironment *env_ptr = nullptr;
        status = ailiaGetSelectedEnvironment(ailia_decoder, &env_ptr, AILIA_ENVIRONMENT_VERSION);
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaGetSelectedEnvironment failed %d\n", status);
            ailiaDestroy(ailia_decoder);
            return -1;
        }

        PRINT_OUT("selected env name : %s\n", env_ptr->name);

        status = ailiaOpenStreamFile(ailia_decoder, decoder_model.c_str());
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaOpenStreamFile failed %d\n", status);
            PRINT_ERR("ailiaGetErrorDetail %s\n", ailiaGetErrorDetail(ailia_decoder));
            ailiaDestroy(ailia_decoder);
            return -1;
        }

        status = ailiaOpenWeightFile(ailia_decoder, decoder_weight.c_str());
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaOpenWeightFile failed %d\n", status);
            ailiaDestroy(ailia_decoder);
            return -1;
        }
    }

    // initialize tokenizer
    AILIATokenizer *tokenizer_source, *tokenizer_target;
	status = ailiaTokenizerCreate(&tokenizer_source, AILIA_TOKENIZER_TYPE_MARIAN, AILIA_TOKENIZER_FLAG_NONE);
	if (status != 0){
		PRINT_ERR("ailiaTokenizerCreate error %d\n", status);
		return -1;
	}
	status = ailiaTokenizerOpenModelFile(tokenizer_source, "source.spm");
    // status = ailiaTokenizerOpenModelFile(tokenizer_source, "target.spm"); //日本語と英語を入れ替える
    // pythonからsource.spmとtarget.spmをコピーしてきたらうまくいった... 要修正
	if (status != 0){
		PRINT_ERR("ailiaTokenizerOpenModelFile error %d\n", status);
		return -1;
	}
	status = ailiaTokenizerCreate(&tokenizer_target, AILIA_TOKENIZER_TYPE_MARIAN, AILIA_TOKENIZER_FLAG_NONE);
	if (status != 0){
		PRINT_ERR("ailiaTokenizerCreate error %d\n", status);
		return -1;
	}
	status = ailiaTokenizerOpenModelFile(tokenizer_target, "target.spm");
    // status = ailiaTokenizerOpenModelFile(tokenizer_target, "source.spm"); //日本語と英語を入れ替える //これをやっても変わらない...
    // pythonからsource.spmとtarget.spmをコピーしてきたらうまくいった... 要修正
	if (status != 0){
		PRINT_ERR("ailiaTokenizerOpenModelFile error %d\n", status);
		return -1;
	}

    status = recognize_from_text(ailia_encoder, ailia_decoder, tokenizer_source, tokenizer_target);

    ailiaTokenizerDestroy(tokenizer_source);
	ailiaTokenizerDestroy(tokenizer_target);

	ailiaDestroy(ailia_encoder);
    ailiaDestroy(ailia_decoder);

	return status;
}
