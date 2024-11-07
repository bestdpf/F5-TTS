export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=../..
export OMP_NUM_THREADS=10
export DS_ACCELERATOR="cuda"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
export WANDB_DISABLED="true"
export SPEECHT5_DIR="/home/heyi/speech/speecht5_tts"
export VCODER_DIR="/home/heyi/speech/speecht5_hifigan"
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export PHN_TOKENIZER_DIR="/home/heyi/speech/split_phn_tokenizer"
#export PHONEMIZER_ESPEAK_LIBRARY="/home/fangbing/minillm/lib/libespeak-ng.so.1"

# export HF_DATASETS_IN_MEMORY_MAX_SIZE=100000000000
# export MODEL_DIR="../test/multi_all_base_extra_loss_model_out/checkpoint-$1"
export MODEL_DIR="/home/heyi/speech/f5_model/model_last.pt"

export AUDIO_PATH="../test/my_voice.wav"
export PROMPT_TEXT="注意看那个男人"




echo "run test on model dir $MODEL_DIR"
python infer.py --model_dir $MODEL_DIR --device cuda --vcoder_dir $VCODER_DIR --text "Oi, eu te amo, querido!" --audio_path $AUDIO_PATH  --prompt_text $PROMPT_TEXT --out_path ../test/test_pt_$1.wav
python infer.py --model_dir $MODEL_DIR --device cuda --vcoder_dir $VCODER_DIR --text "Dov'è il mio cannone italiano?" --audio_path $AUDIO_PATH  --prompt_text $PROMPT_TEXT --out_path ../test/test_it_$1.wav
python infer.py --model_dir $MODEL_DIR --device cuda --vcoder_dir $VCODER_DIR --text "Cześć, kochanie, kocham cię bardzo, bardziej niż jestem w stanie to wyrazić" --audio_path $AUDIO_PATH  --prompt_text $PROMPT_TEXT --out_path ../test/test_pl_$1.wav
python infer.py --model_dir $MODEL_DIR --device cuda --vcoder_dir $VCODER_DIR --text "Hoi schat, ik hou heel veel van je, meer dan ik kan zeggen" --audio_path $AUDIO_PATH  --prompt_text $PROMPT_TEXT --out_path ../test/test_nl_$1.wav
python infer.py --model_dir $MODEL_DIR --device cuda --vcoder_dir $VCODER_DIR --text "Hi, babe! I love you!" --audio_path $AUDIO_PATH  --prompt_text $PROMPT_TEXT --out_path ../test/test_en_$1.wav
python infer.py --model_dir $MODEL_DIR --device cuda --vcoder_dir $VCODER_DIR --text "自动消息：感谢您的反馈!" --audio_path "$AUDIO_PATH" --prompt_text $PROMPT_TEXT --out_path ../test/test_cn_$1.wav
python infer.py --model_dir $MODEL_DIR --device cuda --vcoder_dir $VCODER_DIR --text "안녕, 사랑해 자기야" --audio_path "$AUDIO_PATH" --prompt_text $PROMPT_TEXT --out_path ../test/test_ko_$1.wav
python infer.py --model_dir $MODEL_DIR --device cuda --vcoder_dir $VCODER_DIR --text "bonjour, je t'aime bébé" --audio_path "$AUDIO_PATH" --prompt_text $PROMPT_TEXT --out_path ../test/test_fr_$1.wav
python infer.py --model_dir $MODEL_DIR --device cuda --vcoder_dir $VCODER_DIR --text "Hallo, ich liebe dich, Baby" --audio_path "$AUDIO_PATH" --prompt_text $PROMPT_TEXT --out_path ../test/test_de_$1.wav
python infer.py --model_dir $MODEL_DIR --device cuda --vcoder_dir $VCODER_DIR --text "こんにちは、愛してるよ、ベイビー" --audio_path "$AUDIO_PATH" --prompt_text $PROMPT_TEXT --out_path ../test/test_ja_$1.wav
python infer.py --model_dir $MODEL_DIR --device cuda --vcoder_dir $VCODER_DIR --text "你好，我愛你寶貝" --audio_path "$AUDIO_PATH" --prompt_text $PROMPT_TEXT --out_path ../test/test_tw_$1.wav
python infer.py --model_dir $MODEL_DIR --device cuda --vcoder_dir $VCODER_DIR --text "مرحبا، أحبك عزيزتي" --audio_path "$AUDIO_PATH" --prompt_text $PROMPT_TEXT --out_path ../test/test_ar_$1.wav
python infer.py --model_dir $MODEL_DIR --device cuda --vcoder_dir $VCODER_DIR --text "hola te amo nena" --audio_path "$AUDIO_PATH" --prompt_text $PROMPT_TEXT --out_path ../test/test_es_$1.wav




