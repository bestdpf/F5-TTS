import argparse
import os.path
import threading

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import SpeechT5HifiGan

from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
)
from f5_tts.model import DiT, UNetT, CFM
from f5_tts.model.token_utils import get_phn_tokenizer
from f5_tts.infer.utils_infer import load_checkpoint
from transformers.models.speecht5 import SpeechT5FeatureExtractor, SpeechT5Processor


# target_sample_rate = 24000
target_sample_rate = 16000
# n_mel_channels = 100
n_mel_channels = 80
# hop_length = 256
hop_length = 16
# win_length = 1024
win_length = 64
n_fft = 1024
mel_spec_type = "t5"  # 'vocos' or 'bigvgan'


ode_method = "euler"
nfe_step = 32  # 16, 32
cfg_strength = 2.0
sway_sampling_coef = -1.0
speed = 1.0


PROCESSOR = None
def get_processor(device='cpu'):
    global PROCESSOR
    if PROCESSOR is not None:
        return PROCESSOR
    with threading.Lock():
        if PROCESSOR is None:
            SPEECHT5_DIR = os.environ.get('SPEECHT5_DIR', '/Volumes/ExFAT/speecht5/speech/speecht5_tts')
            PROCESSOR = SpeechT5Processor.from_pretrained(SPEECHT5_DIR, device_map=device,
                                                          normalize=False)
        return PROCESSOR

def get_t5_mel(wavform, device):
    processor = get_processor(device)

    input_dict = processor(text='test',
                           audio_target=wavform,
                           sampling_rate=16000,
                           return_attention_mask=False,
                           )
    return input_dict['labels']

def run_eval(model_dir, vcoder_dir, out_path, text, prompt_audio_path, prompt_text, device):
    model_cls = DiT
    model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)

    phn_tokenizer = get_phn_tokenizer(device)

    vocab_size = len(phn_tokenizer.get_vocab())
    vocab_char_map = None

    mel_spec_kwargs = dict(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mel_channels=n_mel_channels,
        target_sample_rate=target_sample_rate,
        mel_spec_type=mel_spec_type,
    )

    model = CFM(
        transformer=model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    )

    model = load_checkpoint(model, model_dir, device, dtype=torch.float16, use_ema=True)

    vocoder = SpeechT5HifiGan.from_pretrained(vcoder_dir, device_map=device)


    cur_audio_data, src = librosa.load(prompt_audio_path, sr=16000)
    audio_array = np.asarray(cur_audio_data)
    prompt_mel = get_t5_mel(audio_array, device)

    print(f'prompt mel shape is {prompt_mel.shape}')
    ref_audio_len = prompt_mel.shape[1]
    # Calculate duration
    ref_text_len = len(prompt_text.encode("utf-8"))
    gen_text_len = len(text.encode("utf-8"))
    duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / speed)

    with torch.inference_mode():
        generated, _ = model.sample(
            cond=prompt_mel,
            text=[prompt_text + " " + text],
            duration=duration,
            steps=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
        )
    generated = generated.detach().to(torch.float32)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    with torch.inference_mode():
        gen_audio = vocoder(
            generated[:, ref_audio_len:, :].to(device)
        )
    torchaudio.save(
        out_path, gen_audio.detach().cpu(), target_sample_rate
    )






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="../test/model_out-full",
                        help="pretrained model dir")
    # /home/heyi/speech/speecht5_hifigan
    parser.add_argument("--vcoder_dir", type=str, default="/home/heyi/speech/speecht5_hifigan",
    # parser.add_argument("--vcoder_dir", type=str, default="/Volumes/ExFAT/speecht5/speech/speecht5_hifigan",
                        help="vcoder model dir")
    parser.add_argument("--text", type=str, default="我是大帅哥",
                        help="text")
    parser.add_argument("--prompt_text", type=str, default="说点啥",
                        help="prompt_text")
    parser.add_argument("--audio_path", type=str, default="../test/cctv-man.wav",
                        help="audio path")
    parser.add_argument("--out_path", type=str, default="../test/test-cctv-shuai.wav",
                        help="output audio path")
    parser.add_argument("--device", type=str, default="cpu",
                        help="device: cpu, gpu")
    args = parser.parse_args()
    run_eval(args.model_dir, args.vcoder_dir, args.out_path, args.text, args.audio_path, args.prompt_text, args.device)
