#include "ailia_llm.h"

#include <stdio.h>
#include <string.h>
#include <vector>
#include <string>
#include <iostream>

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

void printUsage(const char* prog_name) {
    std::cerr << "Usage: " << prog_name << " -m MODEL_PATH" << std::endl;
    std::cerr << "  -m, --model MODEL_PATH    path to model file (required)" << std::endl;
}

int main(int argc, char *argv[]){
    std::string model_path;
    bool model_specified = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) {
            if (i + 1 < argc) {
                model_path = argv[i + 1];
                model_specified = true;
                i++;
            } else {
                std::cerr << "Error: -m/--model requires a model path argument" << std::endl;
                printUsage(argv[0]);
                return -1;
            }
        } else {
            std::cerr << "Error: Unknown argument: " << argv[i] << std::endl;
            printUsage(argv[0]);
            return -1;
        }
    }

    if (!model_specified) {
        model_path = "gemma-4-E2B-it-Q4_K_M.gguf";
    }

    DisplayBackend();

    struct AILIALLM* llm;
    int n_ctx = 512;
    ailiaLLMCreate(&llm);

    printf("Loading model: %s\n", model_path.c_str());
    #ifdef _WIN32
        int size_needed = MultiByteToWideChar(CP_UTF8, 0, model_path.c_str(), -1, NULL, 0);
        std::wstring widestr(size_needed, 0);
        MultiByteToWideChar(CP_UTF8, 0, model_path.c_str(), -1, &widestr[0], size_needed);
        int status = ailiaLLMOpenModelFileW(llm, widestr.c_str(), n_ctx);
    #else
        int status = ailiaLLMOpenModelFileA(llm, model_path.c_str(), n_ctx);
    #endif

    if (status != 0) {
        printf("Failed to load model file: %s (status: %d)\n", model_path.c_str(), status);
        ailiaLLMDestroy(llm);
        return -1;
    }

    ailiaLLMSetSamplingParams(llm, 40, 0.9, 0.4, 1234);

    DisplayContextSize(llm);

    std::vector<AILIALLMChatMessage> messages;
    AILIALLMChatMessage message;

    message.role = "system";
    message.content = u8"語尾に「くま」をつけて回答してください。";
    messages.push_back(message);

    message.role = "user";
    message.content = u8"こんにちは。";
    messages.push_back(message);

    printf("ailia LLM Query : %s\n", message.content);

    ailiaLLMSetPrompt(llm, &messages[0], messages.size());
    std::string answer;
    Generate(llm, answer);

    printf("ailia LLM Answer : %s\n", answer.c_str());

    message.role = "assistant";
    message.content = answer.c_str();
    messages.push_back(message);

    message.role = "user";
    message.content = u8"さっきの回答を英語に翻訳してください。";
    messages.push_back(message);

    printf("ailia LLM Query : %s\n", message.content);

    ailiaLLMSetPrompt(llm, &messages[0], messages.size());
    std::string answer2;
    Generate(llm, answer2);

    printf("ailia LLM Answer : %s\n", answer2.c_str());

    ailiaLLMDestroy(llm);

    return 0;
}
