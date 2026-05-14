#include "ailia_llm.h"

#include <stdio.h>
#include <string.h>
#include <vector>
#include <string>

#ifdef _WIN32
#include <windows.h>
#endif

void DisplayBackend(){
    unsigned int count;
    ailiaLLMGetBackendCount(&count);
    for (unsigned int i = 0; i < count; i++){
        const char * name;
        ailiaLLMGetBackendName(&name, i);
        printf("ailia LLM Backend : %d %s\n", i, name);
    }
}

void DisplayContextSize(struct AILIALLM* llm){
    unsigned int context_size;
    ailiaLLMGetContextSize(llm, &context_size);
    printf("ailia LLM Context size : %d\n", context_size);
}

void DisplayMultimodalCapabilities(struct AILIALLM* llm){
    unsigned int vision_support, audio_support;
    int status = ailiaLLMGetMultimodalCapabilities(llm, &vision_support, &audio_support);
    if (status == AILIA_LLM_STATUS_SUCCESS){
        printf("ailia LLM Multimodal Capabilities:\n");
        printf("  Vision Support: %s\n", vision_support ? "Yes" : "No");
        printf("  Audio Support: %s\n", audio_support ? "Yes" : "No");
    } else {
        printf("Failed to get multimodal capabilities: %d\n", status);
    }
}

int Generate(struct AILIALLM* llm, std::string &text){
    unsigned int done = 0;
    text = "";
    while(true) {
        int status = ailiaLLMGenerate(llm, &done);
        if (done == 1){
            break;
        }
        if (status != 0) {
            if (status == AILIA_LLM_STATUS_CONTEXT_FULL){
                printf("ailia LLM Context Size Full. Please reduce prompt messages.");
            } else {
                printf("ailia LLM Error Detected %d\n", status);
            }
            return -1;
        }

        unsigned int size = 0;
        ailiaLLMGetDeltaTextSize(llm, &size);
        std::vector<char> delta(size);
        ailiaLLMGetDeltaText(llm, &delta[0], delta.size());
        text = text + std::string(&delta[0]);
    }
    return 0;
}

int main(int argc, char *argv[]){
    const char* model_path = "gemma-4-E2B-it-Q4_K_M.gguf";
    const char* mmproj_path = "gemma-4-E2B-it-mmproj-F16.gguf";
    const char* image_path = "sample.jpg";

    if (argc >= 2) model_path = argv[1];
    if (argc >= 3) mmproj_path = argv[2];
    if (argc >= 4) image_path = argv[3];

    if (argc == 1){
        printf("Usage : gemma-multimodal <model_path> <mmproj_path> [image_path]\n");
        printf("Using defaults: %s %s %s\n", model_path, mmproj_path, image_path);
    }

    DisplayBackend();

    struct AILIALLM* llm;
    int n_ctx = 2048;

    int status = ailiaLLMCreate(&llm);
    if (status != AILIA_LLM_STATUS_SUCCESS){
        printf("Failed to create LLM instance: %d\n", status);
        return -1;
    }

    #ifdef _WIN32
        wchar_t model_path_w[1024];
        MultiByteToWideChar(CP_UTF8, 0, model_path, -1, model_path_w, sizeof(model_path_w)/sizeof(wchar_t));
        status = ailiaLLMOpenModelFileW(llm, model_path_w, n_ctx);
    #else
        status = ailiaLLMOpenModelFileA(llm, model_path, n_ctx);
    #endif

    if (status != AILIA_LLM_STATUS_SUCCESS){
        printf("Failed to load text model: %d\n", status);
        ailiaLLMDestroy(llm);
        return -1;
    }

    #ifdef _WIN32
        wchar_t mmproj_path_w[1024];
        MultiByteToWideChar(CP_UTF8, 0, mmproj_path, -1, mmproj_path_w, sizeof(mmproj_path_w)/sizeof(wchar_t));
        status = ailiaLLMOpenMultimodalProjectorFileW(llm, mmproj_path_w);
    #else
        status = ailiaLLMOpenMultimodalProjectorFileA(llm, mmproj_path);
    #endif

    if (status != AILIA_LLM_STATUS_SUCCESS){
        printf("Failed to load multimodal projector: %d\n", status);
        ailiaLLMDestroy(llm);
        return -1;
    }

    ailiaLLMSetSamplingParams(llm, 40, 0.9, 0.4, 1234);

    DisplayContextSize(llm);
    DisplayMultimodalCapabilities(llm);

    std::vector<AILIALLMMultimodalChatMessage> messages;
    AILIALLMMultimodalChatMessage message;

    message.role = "system";
    message.content = u8"You are a helpful assistant that can analyze images. Respond in Japanese.";
    message.media_data = NULL;
    message.media_count = 0;
    messages.push_back(message);

    AILIALLMMediaData media_data;
    media_data.media_type = "image";
    media_data.file_path = image_path;
    media_data.data = NULL;
    media_data.data_size = 0;
    media_data.width = 0;
    media_data.height = 0;

    message.role = "user";
    message.content = u8"この画像について詳しく説明してください: <__image__>";
    message.media_data = &media_data;
    message.media_count = 1;
    messages.push_back(message);

    printf("ailia LLM Multimodal Query: %s\n", message.content);
    printf("ailia LLM Image Path: %s\n", image_path);

    status = ailiaLLMSetMultimodalPrompt(llm, &messages[0], messages.size());
    if (status != AILIA_LLM_STATUS_SUCCESS){
        printf("Failed to set multimodal prompt: %d\n", status);
        ailiaLLMDestroy(llm);
        return -1;
    }

    std::string answer;
    int gen_status = Generate(llm, answer);
    if (gen_status != 0){
        printf("Failed to generate response\n");
        ailiaLLMDestroy(llm);
        return -1;
    }

    printf("ailia LLM Multimodal Answer: %s\n", answer.c_str());

    ailiaLLMDestroy(llm);

    return 0;
}
