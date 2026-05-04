/*******************************************************************
*
*    DESCRIPTION:
*      ailia Voice Util
*    AUTHOR:
*      ax Inc.
*    DATE:2026/05/04
*
*******************************************************************/

#pragma once

#include "ailia_voice.h"
#include "ailia_audio.h"
#include "ailia.h"

inline AILIAVoiceApiCallback ailiaVoiceUtilGetCallback(void){
	AILIAVoiceApiCallback callback;
	callback.ailiaAudioResample = ailiaAudioResample;
	callback.ailiaAudioGetResampleLen = ailiaAudioGetResampleLen;
	callback.ailiaCreate = ailiaCreate;
	callback.ailiaOpenWeightFileA = ailiaOpenWeightFileA;
	callback.ailiaOpenWeightFileW = ailiaOpenWeightFileW;
	callback.ailiaOpenWeightMem = ailiaOpenWeightMem;
	callback.ailiaSetMemoryMode = ailiaSetMemoryMode;
	callback.ailiaDestroy = ailiaDestroy;
	callback.ailiaUpdate = ailiaUpdate;
	callback.ailiaGetBlobIndexByInputIndex = ailiaGetBlobIndexByInputIndex;
	callback.ailiaGetBlobIndexByOutputIndex = ailiaGetBlobIndexByOutputIndex;
	callback.ailiaGetBlobData = ailiaGetBlobData;
	callback.ailiaSetInputBlobData = ailiaSetInputBlobData;
	callback.ailiaSetInputBlobShape = ailiaSetInputBlobShape;
	callback.ailiaGetBlobShape = ailiaGetBlobShape;
	callback.ailiaGetInputBlobCount = ailiaGetInputBlobCount;
	callback.ailiaGetOutputBlobCount = ailiaGetOutputBlobCount;
	callback.ailiaGetErrorDetail = ailiaGetErrorDetail;
	callback.ailiaCopyBlobData = ailiaCopyBlobData;
	callback.ailiaAudioGetFrameLen = ailiaAudioGetFrameLen;
	callback.ailiaAudioGetSpectrogram = ailiaAudioGetSpectrogram;
	callback.ailiaAudioGetMelSpectrogram = ailiaAudioGetMelSpectrogram;
	return callback;
}
