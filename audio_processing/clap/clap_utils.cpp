/*******************************************************************
*
*    DESCRIPTION:
*      AILIA clap sample
*    AUTHOR:
*
*    DATE:2024/01/25
*
*******************************************************************/
#include <string.h>
#include <math.h>
#include <stdlib.h>

#include "clap_utils.h"
#include "ailia.h"
#include "ailia_audio.h"
extern bool debug;

static std::vector<float> get_mel_ailia(std::vector<float>& audio_data, const AUDIO_CONFIG& audio_cfg,
    int* dst_frame_n=NULL, int* dst_mel_n=NULL)
{
    const int center = AILIA_AUDIO_STFT_CENTER_ENABLE;
    const int mel_n = 64;
    int status;
    int frame_n;
    std::vector<float> mel;
    
    status = ailiaAudioGetFrameLen(&frame_n, (int)audio_data.size(), audio_cfg.window_size, audio_cfg.hop_size, center);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaAudioGetFrameLen failed %d\n", status);
        return mel;
    }
    if(debug){
        PRINT_OUT("frame_n = %d\n", frame_n);
    }
    
    mel = std::vector<float>(mel_n * frame_n);  // [mel_n][frame_n]
    status = ailiaAudioGetMelSpectrogram(
        &mel[0],
        &audio_data[0],
        audio_data.size(),
        audio_cfg.sample_rate,
        audio_cfg.window_size,
        audio_cfg.hop_size,
        audio_cfg.window_size,
        AILIA_AUDIO_WIN_TYPE_HANN,
        frame_n,
        center,
        2.0,    // power
        AILIA_AUDIO_FFT_NORMALIZE_NONE,
        audio_cfg.fmin,
        audio_cfg.fmax,
        mel_n,
        AILIA_AUDIO_MEL_NORMALIZE_NONE,
        AILIA_AUDIO_MEL_SCALE_FORMULA_HTK
    );
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaAudioGetMelSpectrogram failed %d\n", status);
        return mel;
    }

    // amplitude_to_db
    const float ref = 1.0;
    const float amin = 1e-10;
    for(auto v=mel.begin(); v!=mel.end(); ++v){
        float s = (*v) * (*v);
        if(s >= 0 && s < amin) s = amin;
        if(s < 0 && s > -amin) s = -amin;
        *v = 10 * log10f(s / ref);
    }
    
    // transpose(1, 0):  [mel_n][frame_n] to [frame_n][mel_n]
    std::vector<float> mel_t(mel_n * frame_n);
    float* dst = &mel_t[0];
    for(int j=0; j<frame_n; j++){
        for(int i=0; i<mel_n; i++){
            *dst++ = mel[i * frame_n + j];
        }
    }
    if(dst_frame_n) *dst_frame_n = frame_n;
    if(dst_mel_n) *dst_mel_n = mel_n;
    return mel_t;
}

static void resize_bilinear(float* dst, int dh, int dw, float* src, int sh, int sw)
{
    float hr = (float)sh / (float)dh;
    float wr = (float)sw / (float)dw;
    float fy = 0;
    for(int y=0; y<dh; y++){
        int y0 = std::min<int>((int)floorf(fy), sh - 1);
        int y1 = std::min<int>((int)ceilf(fy), sh - 1);
        float yw = 1.f;
        if(y0 != y1){
            yw = fy - y0;
        }
        float fx = 0;
        for(int x=0; x<dw; x++){
            int x0 = std::min<int>((int)floorf(fx), sw - 1);
            int x1 = std::min<int>((int)ceilf(fx), sw - 1);
            float xw = 1.f;
            if(x0 != x1){
                xw = fx - x0;
            }
            float v00 = src[sw * y0 + x0];
            float v01 = src[sw * y0 + x1];
            float v10 = src[sw * y1 + x0];
            float v11 = src[sw * y1 + x1];
            float res = 0;
            res += v00 * (1.f - yw) * (1.f - xw);
            res += v01 * (1.f - yw) * (xw);
            res += v10 * (yw) * (1.f - xw);
            res += v11 * (yw) * (xw);
            *dst++ = res;
            
            fx += wr;
        }
        fy += hr;
    }
}

std::vector<float> get_audio_features(std::vector<float>& audio_data, unsigned int max_len, 
    std::string data_truncating, std::string data_filling, const AUDIO_CONFIG& audio_cfg, bool* plonger)
{
    std::vector<float> mel_fusion;
    if(plonger) *plonger = false;
    
    if(audio_data.size() > max_len){
        if(data_truncating == "fusion"){
            // fusion
            int frame_n = 0, mel_n = 0;
            std::vector<float> mel = get_mel_ailia(audio_data, audio_cfg, &frame_n, &mel_n);
            if(mel.size() < 1) return mel_fusion;

            int chunk_frames = max_len / audio_cfg.hop_size + 1;  // the +1 related to how the spectrogram is computed
            int total_frames = frame_n;
            if(debug){
                PRINT_OUT("shrink audio %ld to be %d, frame_n=%d\n", audio_data.size(), max_len, frame_n);
            }
            if(chunk_frames == total_frames){
                // there is a corner case where the audio length is
                // larger than max_len but smaller than max_len+hop_size.
                // In this case, we just use the whole audio.
                mel_fusion = std::vector<float>(mel.size() * 4);
                for(int i=0; i<4; i++){
                    memcpy(&mel_fusion[i * mel.size()], &mel[0], mel.size() * sizeof(float));
                }
            }
            else{
                // split to three parts
                const int frame_size = chunk_frames * mel_n;
                mel_fusion = std::vector<float>(frame_size * 4);
                int num = total_frames - chunk_frames + 1;
                int div = std::max(1, (num / 3));
                for(int last=0, i=0; i<3; i++){
                    int first = last;
                    last = std::min(num, first + div);
                    int choice = 0;
                    if(first < last){
                        // randomly choose index for each part
                        choice = first + (rand() % (last - first));
                    }
                    if(debug){
                        PRINT_OUT("  random choose %d / %d\n", choice, num);
                    }
                    memcpy(&mel_fusion[i * frame_size], &mel[choice * mel_n], frame_size * sizeof(float));
                }
                // shrink the mel
                resize_bilinear(&mel_fusion[3 * frame_size], chunk_frames, mel_n, &mel[0], frame_n, mel_n);

                if(plonger) *plonger = true;
            }
        }
        else{
            PRINT_ERR("Not support data_truncating: %s\n", data_truncating.c_str());
            return mel_fusion;
        }
    }
    else{   // padding
        if(audio_data.size() < max_len){
            if(debug){
                PRINT_OUT("padding for audio %ld to be %d\n", audio_data.size(), max_len);
            }
            std::vector<float> new_audio_data(max_len, 0);
            if(data_filling == "repeatpad"  || data_filling == "repeat"){
                int n_repeat = max_len / audio_data.size();
                for(int i=0; i<n_repeat; i++){
                    memcpy(&new_audio_data[i * audio_data.size()], &audio_data[0], audio_data.size() * sizeof(float));
                }
                if(data_filling == "repeat"){
                    int rem = max_len - audio_data.size() * n_repeat;
                    memcpy(&new_audio_data[n_repeat * audio_data.size()], &audio_data[0], rem * sizeof(float));
                }
            }
            else if(data_filling == "pad"){
                memcpy(&new_audio_data[0], &audio_data[0], audio_data.size() * sizeof(float));
            }
            else{
                PRINT_ERR("Not support data_filling: %s\n", data_filling.c_str());
                return mel_fusion;
            }
            audio_data = new_audio_data;
        }
        if(data_truncating == "fusion"){
            std::vector<float> mel = get_mel_ailia(audio_data, audio_cfg);
            if(mel.size() < 1) return mel_fusion;
            mel_fusion = std::vector<float>(mel.size() * 4);
            for(int i=0; i<4; i++){
                memcpy(&mel_fusion[i * mel.size()], &mel[0], mel.size() * sizeof(float));
            }
        }
        else{
            PRINT_ERR("Not support data_truncating: %s\n", data_truncating.c_str());
            return mel_fusion;
        }
    }
    
    return mel_fusion;
}
