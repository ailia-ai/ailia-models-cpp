/*******************************************************************
*
*    DESCRIPTION:
*      AILIA clip sample
*    AUTHOR:
*
*    DATE:2023/06/16
*
*******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <string>
#include <algorithm>

#include <opencv2/opencv.hpp>

#if defined(_WIN32) || defined(_WIN64)
#define NOMINMAX
#include <windows.h>
#define sleep(n) Sleep(n*1000)
#else
#include <unistd.h>
#endif

#undef UNICODE

#include "ailia.h"
#include "ailia_tokenizer.h"

#include "utils.h"
#include "webcamera_utils.h"


// ======================
// Parameters
// ======================

#define WEIGHT_PATH_IMAGE "ViT-B32-encode_image.onnx"
#define MODEL_PATH_IMAGE  "ViT-B32-encode_image.onnx.prototxt"

#define WEIGHT_PATH_TEXT "ViT-B32-encode_text.onnx"
#define MODEL_PATH_TEXT  "ViT-B32-encode_text.onnx.prototxt"

#define IMAGE_PATH  "chelsea.png"

#define IMAGE_WIDTH  224 // for video mode
#define IMAGE_HEIGHT 224 // for video mode

#define CONTEXT_LENGTH 77
#define FEATURE_LENGTH 512

#if defined(_WIN32) || defined(_WIN64)
#define PRINT_OUT(...) fprintf_s(stdout, __VA_ARGS__)
#define PRINT_ERR(...) fprintf_s(stderr, __VA_ARGS__)
#else
#define PRINT_OUT(...) fprintf(stdout, __VA_ARGS__)
#define PRINT_ERR(...) fprintf(stderr, __VA_ARGS__)
#endif

#define BENCHMARK_ITERS 5

static std::string weight_image(WEIGHT_PATH_IMAGE);
static std::string model_image(MODEL_PATH_IMAGE);

static std::string weight_text(WEIGHT_PATH_TEXT);
static std::string model_text(MODEL_PATH_TEXT);

static std::string image_path(IMAGE_PATH);
static std::vector<std::string> texts;

static bool benchmark  = false;
static int args_env_id = -1;
static bool debug = false;


// ======================
// Arguemnt Parser
// ======================

static void print_usage()
{
    PRINT_OUT("usage: clip [-h] [-i IMAGE] [-b] [-e ENV_ID]\n");
    return;
}


static void print_help()
{
    PRINT_OUT("\n");
    PRINT_OUT("clip classification model\n");
    PRINT_OUT("\n");
    PRINT_OUT("optional arguments:\n");
    PRINT_OUT("  -h, --help            show this help message and exit\n");
    PRINT_OUT("  -i IMAGE, --input IMAGE\n");
    PRINT_OUT("                        The input image path.\n");
    PRINT_OUT("  -t TEXT, --text TEXT\n");
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
    PRINT_ERR("clip: error: unrecognized arguments: %s\n", arg.c_str());
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
			else if (arg == "-t" || arg == "--text") {
				status = 5;
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
                image_path = arg;
                break;
			case 4:
				args_env_id = atoi(arg.c_str());
				break;
			case 5:
				texts.push_back(arg);
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
// Utils
// ======================

static void preprocess_image(const cv::Mat& simg, cv::Mat& dimg)
{
    if (simg.channels() == 3) {
        cv::cvtColor(simg, dimg, cv::COLOR_BGR2RGBA);
    }
    else if (simg.channels() == 1) {
        cv::cvtColor(simg, dimg, cv::COLOR_GRAY2RGBA);
    }
    else {
        simg.copyTo(dimg);
    }

    return;
}

void softmax(float *data, int n){
	float sum=0;
	for(int i=0;i<n;i++){
		sum+=exp(data[i]);
	}
	for(int i=0;i<n;i++){
		data[i]=exp(data[i])/sum;
	}
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

// ======================
// Image embeddings
// ======================

std::vector<float> resize_and_center_crop(cv::Mat img){
    // rgb order, (/255 - mean )/std
    // resize to 224 and center crop
    std::vector<float> input_img(IMAGE_WIDTH * IMAGE_HEIGHT * 3);
    float mean[3] = {0.48145466, 0.4578275, 0.40821073};
    float stdf[3] = {0.26862954, 0.26130258, 0.27577711};
    float ratio_w = 1.0f * img.cols / IMAGE_WIDTH;
    float ratio_h = 1.0f * img.rows / IMAGE_HEIGHT;
    float ratio = std::min(ratio_w, ratio_h);
    if (debug){
        PRINT_OUT("input %dx%d output %dx%d ratio %fx%f\n", img.cols, img.rows, (int)(IMAGE_WIDTH * ratio), (int)(IMAGE_HEIGHT * ratio), ratio_w, ratio_h);
    }
    std::vector<unsigned char> preview(IMAGE_HEIGHT * IMAGE_WIDTH * 4);
    for (int y = 0; y < IMAGE_HEIGHT; y++){
        for (int x = 0; x < IMAGE_WIDTH; x++){
            int y2 = y * ratio + (img.rows - IMAGE_HEIGHT * ratio)/2;
            int x2 = x * ratio + (img.cols - IMAGE_WIDTH * ratio)/2;
            if (x2 < img.cols && y2 < img.rows){
                input_img[0 * IMAGE_WIDTH * IMAGE_HEIGHT + y * IMAGE_WIDTH + x] = (img.data[(img.cols * y2 + x2)*4 + 0] / 255.0f - mean[0])/stdf[0];
                input_img[1 * IMAGE_WIDTH * IMAGE_HEIGHT + y * IMAGE_WIDTH + x] = (img.data[(img.cols * y2 + x2)*4 + 1] / 255.0f - mean[1])/stdf[1];
                input_img[2 * IMAGE_WIDTH * IMAGE_HEIGHT + y * IMAGE_WIDTH + x] = (img.data[(img.cols * y2 + x2)*4 + 2] / 255.0f - mean[2])/stdf[2];
            }else{
                input_img[0 * IMAGE_WIDTH * IMAGE_HEIGHT + y * IMAGE_WIDTH + x] = 0.0f;
                input_img[1 * IMAGE_WIDTH * IMAGE_HEIGHT + y * IMAGE_WIDTH + x] = 0.0f;
                input_img[2 * IMAGE_WIDTH * IMAGE_HEIGHT + y * IMAGE_WIDTH + x] = 0.0f;
            }
            if (debug){
                for (int i = 0; i < 3; i++){
                    int v = (input_img[i * IMAGE_WIDTH * IMAGE_HEIGHT + y * IMAGE_WIDTH + x] * stdf[i] + mean[i])*255;
                    preview[(y*IMAGE_WIDTH + x)*4 + i] = std::max(0, std::min(255, v));
                }
            }
        }
    }

    if (debug){
        cv::Mat dest(IMAGE_HEIGHT, IMAGE_WIDTH, img.type());
        dest.data = preview.data();
        cv::imwrite("temp1.jpg", dest);

        cv::Mat dest2(img.rows, img.cols, img.type());
        dest2.data = img.data;
        cv::imwrite("temp2.jpg", dest2);
    }

    return input_img;
}

static std::vector<float> image_embedding(AILIANetwork *image_enc, std::string path)
{
    std::vector<float> features(FEATURE_LENGTH);

    // prepare input data
    cv::Mat simg = cv::imread(path.c_str(), cv::IMREAD_UNCHANGED);
    if (simg.empty()) {
        PRINT_ERR("\'%s\' image not found\n", image_path.c_str());
        return features;
    }
    
    // simg is bgr, img is rgba
    cv::Mat img;
    preprocess_image(simg, img);
    std::vector<float> input_img = resize_and_center_crop(img);

    // inference
    int status;
    unsigned int input_blob_idx = 0;
    status = ailiaGetBlobIndexByInputIndex(image_enc, &input_blob_idx, 0);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ImageEmbedding ailiaGetBlobIndexByInputIndex %d", status);
        return features;
    }

    AILIAShape sequence_shape;
    sequence_shape.x=IMAGE_WIDTH;
    sequence_shape.y=IMAGE_HEIGHT;
    sequence_shape.z=3;
    sequence_shape.w=1;
    sequence_shape.dim=4;
    
    status = ailiaSetInputBlobShape(image_enc,&sequence_shape,input_blob_idx,AILIA_SHAPE_VERSION);
    if(status!=AILIA_STATUS_SUCCESS){
        PRINT_ERR("ImageEmbedding ailiaSetInputBlobShape failed %d\n", status);
        return features;
    }

    status = ailiaPredict(image_enc, &features[0], features.size() * sizeof(float), &input_img[0], input_img.size() * sizeof(float));
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ImageEmbedding ailiaPredict failed %d\n", status);
        return features;
    }

    return features;
}

// ======================
// Text embeddings
// ======================

std::vector<int> tokenize(AILIATokenizer * tokenizer, std::string text){
    if (debug){
        printf("Input Text : %s\n", text.c_str());
    }
	ailiaTokenizerEncode(tokenizer, text.c_str());
	unsigned int count;
	ailiaTokenizerGetTokenCount(tokenizer, &count);
	std::vector<int> tokens(count);
	ailiaTokenizerGetTokens(tokenizer, &tokens[0], count);
    std::vector<int> pad_tokens(CONTEXT_LENGTH);
    for (int i = 0; i < CONTEXT_LENGTH; i++){
        if ((size_t)i < tokens.size()){
            pad_tokens[i] = tokens[i];
            if (i == CONTEXT_LENGTH - 1){
                pad_tokens[i] = tokens[tokens.size() - 1]; // SOT
            }
        }else{
            pad_tokens[i] = 0;
        }
    }
    if (debug){
        printf("Tokens : ");
        for (size_t i = 0; i < pad_tokens.size(); i++){
            printf("%d ", pad_tokens[i]);
        }
        printf("\n");
    }
    return pad_tokens;
}

std::vector<float> text_embedding(AILIANetwork *ailia_text, std::vector<int> &tokens){
    std::vector<float> features(FEATURE_LENGTH);

    unsigned int input_blob_idx = 0;
    int status = ailiaGetBlobIndexByInputIndex(ailia_text, &input_blob_idx, 0);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("TextEmbedding ailiaGetBlobIndexByInputIndex %d", status);
        return features;
    }

    AILIAShape sequence_shape;
    sequence_shape.x=CONTEXT_LENGTH;
    sequence_shape.y=1;
    sequence_shape.z=1;
    sequence_shape.w=1;
    sequence_shape.dim=2;

    status = ailiaSetInputBlobShape(ailia_text,&sequence_shape,input_blob_idx,AILIA_SHAPE_VERSION);
    if(status!=AILIA_STATUS_SUCCESS){
        PRINT_ERR("TextEmbedding ailiaSetInputBlobShape failed %d\n", status);
        return features;
    }

    if (tokens.size() != CONTEXT_LENGTH){
        PRINT_ERR("Invalid token size %lu\n", tokens.size());
        return features;
    }

    std::vector<float> tokens_float(CONTEXT_LENGTH);
    for (int i = 0; i < CONTEXT_LENGTH; i++){
        tokens_float[i] = (float)tokens[i];
    }
    status = ailiaPredict(ailia_text, &features[0], features.size() * sizeof(float), &tokens_float[0], tokens_float.size() * sizeof(float));
    if (status != AILIA_STATUS_SUCCESS){
        PRINT_ERR("TextEmbedding ailiaGetErrorDetail %s\n", ailiaGetErrorDetail(ailia_text));
        return features;
    }
    return features;
}

// ======================
// Main functions
// ======================

int get_env_id(void)
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

int initialize_image_encoder(AILIANetwork **ailia_image, int env_id){
    int status = ailiaCreate(ailia_image, env_id, AILIA_MULTITHREAD_AUTO);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaCreate failed %d\n", status);
        return -1;
    }
    status = ailiaOpenStreamFile(*ailia_image, model_image.c_str());
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaOpenStreamFile failed %d\n", status);
        PRINT_ERR("ailiaGetErrorDetail %s\n", ailiaGetErrorDetail(*ailia_image));
        ailiaDestroy(*ailia_image);
        return -1;
    }
    status = ailiaOpenWeightFile(*ailia_image, weight_image.c_str());
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaOpenWeightFile failed %d\n", status);
        ailiaDestroy(*ailia_image);
        return -1;
    }
    return AILIA_STATUS_SUCCESS;
}

int initialize_text_encoder(AILIANetwork **ailia_text, int env_id){
    int status = ailiaCreate(ailia_text, env_id, AILIA_MULTITHREAD_AUTO);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaCreate failed %d\n", status);
        return -1;
    }
    status = ailiaOpenStreamFile(*ailia_text, model_text.c_str());
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaOpenStreamFile failed %d\n", status);
        PRINT_ERR("ailiaGetErrorDetail %s\n", ailiaGetErrorDetail(*ailia_text));
        ailiaDestroy(*ailia_text);
        return -1;
    }
    status = ailiaOpenWeightFile(*ailia_text, weight_text.c_str());
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaOpenWeightFile failed %d\n", status);
        ailiaDestroy(*ailia_text);
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
    AILIANetwork *ailia_image;
    status = initialize_image_encoder(&ailia_image, env_id);
    if (status != AILIA_STATUS_SUCCESS) {
        return -1;
    }

    AILIANetwork *ailia_text;
    status = initialize_text_encoder(&ailia_text, env_id);
    if (status != AILIA_STATUS_SUCCESS) {
        return -1;
    }

    // Tokenize
    std::vector<std::vector<int>> tokens;
    if (texts.size() == 0){
        texts.push_back(std::string("a dog"));
        texts.push_back(std::string("a cat"));
        texts.push_back(std::string("a human"));
    }
	AILIATokenizer *tokenizer;
	status = ailiaTokenizerCreate(&tokenizer, AILIA_TOKENIZER_TYPE_CLIP, AILIA_TOKENIZER_FLAG_NONE);
	if (status != 0){
		PRINT_ERR("ailiaTokenizerCreate error %d\n", status);
		return -1;
	}
    PRINT_OUT("Tokenize...\n");
    for (size_t i = 0; i < texts.size(); i++){
        std::vector<int>  token = tokenize(tokenizer, texts[i]);
        tokens.push_back(token);
    }
	ailiaTokenizerDestroy(tokenizer);

    // text embedding
    PRINT_OUT("Text embedding...\n");
    std::vector< std::vector<float> > text_features;
    for (size_t i = 0; i < texts.size(); i++){
        std::vector<float> features = text_embedding(ailia_text, tokens[i]);
        text_features.push_back(features);
    }

    // image embedding
    PRINT_OUT("Image embedding...\n");
    std::vector<float> image_features = image_embedding(ailia_image, image_path);

    // distance
    PRINT_OUT("Similarity...\n");
    std::vector<float> confs;
    std::vector<float> sims;
    for (size_t i = 0; i < texts.size(); i++){
        float sim = cos_similarity(image_features, text_features[i]);
        confs.push_back(sim * 100);
        sims.push_back(sim);
    }

    softmax(&confs[0], confs.size());

    for (size_t i = 0; i < texts.size(); i++){
        printf("Label %s Confidence %f Similarity %f\n", texts[i].c_str(), confs[i], sims[i]);
    }

    // release instance
    ailiaDestroy(ailia_image);
    ailiaDestroy(ailia_text);

    PRINT_OUT("Program finished successfully.\n");

    return status;
}
