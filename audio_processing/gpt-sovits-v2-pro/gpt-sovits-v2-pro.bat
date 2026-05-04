@echo off
setlocal enabledelayedexpansion
cd %~dp0

set MODEL=gpt-sovits-v2-pro

set GPT_SOVITS_V2_PRO_FILES=t2s_encoder.onnx t2s_fsdec.onnx t2s_sdec.opt.onnx cnhubert.onnx vits.onnx sv.onnx chinese-roberta.onnx vocab.txt
set OPEN_JTALK_FILES=COPYING char.bin left-id.def matrix.bin pos-id.def rewrite.def right-id.def sys.dic unk.dic
set G2P_EN_FILES=averaged_perceptron_tagger_classes.txt averaged_perceptron_tagger_tagdict.txt averaged_perceptron_tagger_weights.txt cmudict homographs.en g2p_decoder.onnx g2p_encoder.onnx
set G2P_CN_FILES=pinyin.txt opencpop-strict.txt jieba.dict.utf8 hmm_model.utf8 user.dict.utf8 idf.utf8 stop_words.utf8
set G2PW_MODEL_FILES=g2pW.onnx POLYPHONIC_CHARS.txt bopomofo_to_pinyin_wo_tune_dict.json
set G2PW_DATA_FILES=polyphonic.rep polyphonic-fix.rep

rem download
if not "%1" == "-h" if not "%1" == "--help" (
    if not exist gpt-sovits-v2-pro (
        mkdir gpt-sovits-v2-pro
    )
    for %%f in (%GPT_SOVITS_V2_PRO_FILES%) do (
        if not exist gpt-sovits-v2-pro\%%f (
            echo Downloading model file... ^(save path: gpt-sovits-v2-pro\%%f^)
            curl https://storage.googleapis.com/ailia-models/%MODEL%/%%f -o gpt-sovits-v2-pro\%%f
        )
    )

    if not exist open_jtalk_dic_utf_8-1.11 (
        mkdir open_jtalk_dic_utf_8-1.11
    )
    for %%f in (%OPEN_JTALK_FILES%) do (
        if not exist open_jtalk_dic_utf_8-1.11\%%f (
            echo Downloading Open JTalk dictionary... ^(save path: open_jtalk_dic_utf_8-1.11\%%f^)
            curl https://storage.googleapis.com/ailia-models/open_jtalk/open_jtalk_dic_utf_8-1.11/%%f -o open_jtalk_dic_utf_8-1.11\%%f
        )
    )

    if not exist g2p_en (
        mkdir g2p_en
    )
    for %%f in (%G2P_EN_FILES%) do (
        if not exist g2p_en\%%f (
            echo Downloading g2p_en data... ^(save path: g2p_en\%%f^)
            curl https://storage.googleapis.com/ailia-models/g2p_en/%%f -o g2p_en\%%f
        )
    )

    if not exist g2p_cn (
        mkdir g2p_cn
    )
    for %%f in (%G2P_CN_FILES%) do (
        if not exist g2p_cn\%%f (
            echo Downloading g2p_cn data... ^(save path: g2p_cn\%%f^)
            curl https://storage.googleapis.com/ailia-models/g2p_cn/%%f -o g2p_cn\%%f
        )
    )

    if not exist g2pw (
        mkdir g2pw
    )
    for %%f in (%G2PW_MODEL_FILES%) do (
        if not exist g2pw\%%f (
            echo Downloading g2pw data... ^(save path: g2pw\%%f^)
            curl https://storage.googleapis.com/ailia-models/g2pw/1.1/%%f -o g2pw\%%f
        )
    )
    for %%f in (%G2PW_DATA_FILES%) do (
        if not exist g2pw\%%f (
            echo Downloading g2pw data... ^(save path: g2pw\%%f^)
            curl https://raw.githubusercontent.com/axinc-ai/ailia-models/master/audio_processing/gpt-sovits-v2/text/g2pw/%%f -o g2pw\%%f
        )
    )

    echo All required files are prepared^^!
)

rem execute
.\%MODEL%.exe %*
