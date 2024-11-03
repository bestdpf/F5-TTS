# training script.

from importlib.resources import files

import torch

from f5_tts.model import CFM, DiT, Trainer, UNetT
from f5_tts.model.dataset import load_dataset
from f5_tts.model.token_utils import get_phn_tokenizer

# -------------------------- Dataset Settings --------------------------- #

# target_sample_rate = 24000
target_sample_rate = 16000
# n_mel_channels = 100
n_mel_channels = 80
# hop_length = 256
hop_length = 16
# win_length = 1024
win_length = 64
n_fft = 1024
mel_spec_type = "bigvgan"  # 'vocos' or 'bigvgan'

tokenizer = "custom"  # 'pinyin', 'char', or 'custom'
tokenizer_path = '/home/projects/u6554606/llm/split_phn_tokenizer'  # if tokenizer = 'custom', define the path to the tokenizer you want to use (should be vocab.txt)
dataset_name = "/home/projects/u6554606/llm/multi_lang_mel_test"

# -------------------------- Training Settings -------------------------- #

exp_name = "F5TTS_Base"  # F5TTS_Base | E2TTS_Base

learning_rate = 7.5e-5

# batch_size_per_gpu = 38400  # 8 GPUs, 8 * 38400 = 307200
batch_size_per_gpu = 8  # 8 GPUs, 8 * 38400 = 307200
# batch_size_type = "frame"  # "frame" or "sample"
batch_size_type = "sample"  # "frame" or "sample"
max_samples = 64  # max sequences per batch if use frame-wise batch_size. we set 32 for small models, 64 for base models
grad_accumulation_steps = 1  # note: updates = steps / grad_accumulation_steps
max_grad_norm = 1.0

epochs = 22  # use linear decay, thus epochs control the slope
num_warmup_updates = 10000  # warmup steps
save_per_updates = 20000  # save checkpoint per steps
last_per_steps = 2000  # save last checkpoint per steps

# model params
if exp_name == "F5TTS_Base":
    wandb_resume_id = None
    model_cls = DiT
    model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
elif exp_name == "E2TTS_Base":
    wandb_resume_id = None
    model_cls = UNetT
    model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)


# ----------------------------------------------------------------------- #


def main():

    phn_tokenizer = get_phn_tokenizer('cuda' if torch.cuda.is_available() else 'cpu')

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

    ckpt_path = str(files("f5_tts").joinpath(f"../../ckpts/{exp_name}"))
    print(f'ckpt path is {ckpt_path}')

    trainer = Trainer(
        model,
        epochs,
        learning_rate,
        num_warmup_updates=num_warmup_updates,
        save_per_updates=save_per_updates,
        checkpoint_path=ckpt_path,
        batch_size=batch_size_per_gpu,
        batch_size_type=batch_size_type,
        max_samples=max_samples,
        grad_accumulation_steps=grad_accumulation_steps,
        max_grad_norm=max_grad_norm,
        wandb_project="CFM-TTS",
        wandb_run_name=exp_name,
        wandb_resume_id=wandb_resume_id,
        last_per_steps=last_per_steps,
        log_samples=True,
        mel_spec_type=mel_spec_type,
    )

    train_dataset = load_dataset(dataset_name, tokenizer, 'HFDataset', mel_spec_kwargs=mel_spec_kwargs)
    trainer.train(
        train_dataset,
        resumable_with_seed=666,  # seed for shuffling dataset
    )


if __name__ == "__main__":
    main()
