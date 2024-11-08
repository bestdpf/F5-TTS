import glob
import json
import os
import random
from importlib.resources import files

import datasets
import torch
import torch.nn.functional as F
import torchaudio
from datasets import Dataset as Dataset_
from datasets import load_from_disk
from torch import nn
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import default


class HFDataset(Dataset):
    def __init__(
        self,
        hf_dataset: Dataset,
        target_sample_rate=24_000,
        n_mel_channels=100,
        hop_length=256,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
    ):
        self.data = hf_dataset
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length

        self.mel_spectrogram = MelSpec(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        )

    def get_frame_len(self, index):
        row = self.data[index]
        audio = row["audio"]["array"]
        sample_rate = row["audio"]["sampling_rate"]
        return audio.shape[-1] / sample_rate * self.target_sample_rate / self.hop_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        audio = row["audio"]["array"]

        # logger.info(f"Audio shape: {audio.shape}")

        sample_rate = row["audio"]["sampling_rate"]
        duration = audio.shape[-1] / sample_rate

        if duration > 30 or duration < 0.3:
            return self.__getitem__((index + 1) % len(self.data))

        audio_tensor = torch.from_numpy(audio).float()

        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            audio_tensor = resampler(audio_tensor)

        audio_tensor = audio_tensor.unsqueeze(0)  # 't -> 1 t')

        mel_spec = self.mel_spectrogram(audio_tensor)

        mel_spec = mel_spec.squeeze(0)  # '1 d t -> d t'

        text = row["text"]

        return dict(
            mel_spec=mel_spec,
            text=text,
        )


class CustomDataset(Dataset):
    def __init__(
        self,
        custom_dataset: Dataset,
        durations=None,
        target_sample_rate=24_000,
        hop_length=256,
        n_mel_channels=100,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
        preprocessed_mel=False,
        mel_spec_module: nn.Module | None = None,
    ):
        self.data = custom_dataset
        self.durations = durations
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.mel_spec_type = mel_spec_type
        self.preprocessed_mel = preprocessed_mel

        if not preprocessed_mel:
            self.mel_spectrogram = default(
                mel_spec_module,
                MelSpec(
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    n_mel_channels=n_mel_channels,
                    target_sample_rate=target_sample_rate,
                    mel_spec_type=mel_spec_type,
                ),
            )

    def get_frame_len(self, index):
        if (
            self.durations is not None
        ):  # Please make sure the separately provided durations are correct, otherwise 99.99% OOM
            return self.durations[index] * self.target_sample_rate / self.hop_length
        return self.data[index]["duration"] * self.target_sample_rate / self.hop_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        audio_path = row["audio_path"]
        text = row["text"]
        duration = row["duration"]

        if self.preprocessed_mel:
            mel_spec = torch.tensor(row["mel_spec"])

        else:
            audio, source_sample_rate = torchaudio.load(audio_path)
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)

            if duration > 30 or duration < 0.3:
                return self.__getitem__((index + 1) % len(self.data))

            if source_sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(source_sample_rate, self.target_sample_rate)
                audio = resampler(audio)

            mel_spec = self.mel_spectrogram(audio)
            mel_spec = mel_spec.squeeze(0)  # '1 d t -> d t')

        return dict(
            mel_spec=mel_spec,
            text=text,
        )


# Dynamic Batch Sampler


class DynamicBatchSampler(Sampler[list[int]]):
    """Extension of Sampler that will do the following:
    1.  Change the batch size (essentially number of sequences)
        in a batch to ensure that the total number of frames are less
        than a certain threshold.
    2.  Make sure the padding efficiency in the batch is high.
    """

    def __init__(
        self, sampler: Sampler[int], frames_threshold: int, max_samples=0, random_seed=None, drop_last: bool = False
    ):
        self.sampler = sampler
        self.frames_threshold = frames_threshold
        self.max_samples = max_samples

        indices, batches = [], []
        data_source = self.sampler.data_source

        print('shape of mel')
        print(data_source[0]['mel_spec'].shape)


        if os.path.exists('indices.json'):
            indices = json.load(open('indices.json', 'r'))
        else:
            for idx in tqdm(
                    self.sampler,
                    desc="Sorting with sampler... if slow, check whether dataset is provided with duration"
            ):
                indices.append((idx, data_source[idx]['mel_spec'].shape[0]))
            indices.sort(key=lambda elem: elem[1])
            json.dump(indices, open('indices.json', 'w'))

        batch = []
        batch_frames = 0
        for idx, frame_len in tqdm(
            indices, desc=f"Creating dynamic batches with {frames_threshold} audio frames per gpu"
        ):
            if batch_frames + frame_len <= self.frames_threshold and (max_samples == 0 or len(batch) < max_samples):
                batch.append(idx)
                batch_frames += frame_len
            else:
                if len(batch) > 0:
                    batches.append(batch)
                if frame_len <= self.frames_threshold:
                    batch = [idx]
                    batch_frames = frame_len
                else:
                    batch = []
                    batch_frames = 0

        if not drop_last and len(batch) > 0:
            batches.append(batch)

        del indices

        # if want to have different batches between epochs, may just set a seed and log it in ckpt
        # cuz during multi-gpu training, although the batch on per gpu not change between epochs, the formed general minibatch is different
        # e.g. for epoch n, use (random_seed + n)
        random.seed(random_seed)
        random.shuffle(batches)

        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


# Load dataset


def load_dataset(
    dataset_name: str,
    tokenizer: str = "pinyin",
    dataset_type: str = "CustomDataset",
    audio_type: str = "raw",
    mel_spec_module: nn.Module | None = None,
    mel_spec_kwargs: dict = dict(),
) -> CustomDataset | HFDataset:
    """
    dataset_type    - "CustomDataset" if you want to use tokenizer name and default data path to load for train_dataset
                    - "CustomDatasetPath" if you just want to pass the full path to a preprocessed dataset without relying on tokenizer
    """

    print("Loading dataset ...")

    if dataset_type == "CustomDataset":
        rel_data_path = str(files("f5_tts").joinpath(f"../../data/{dataset_name}_{tokenizer}"))
        if audio_type == "raw":
            try:
                train_dataset = load_from_disk(f"{rel_data_path}/raw")
            except:  # noqa: E722
                train_dataset = Dataset_.from_file(f"{rel_data_path}/raw.arrow")
            preprocessed_mel = False
        elif audio_type == "mel":
            train_dataset = Dataset_.from_file(f"{rel_data_path}/mel.arrow")
            preprocessed_mel = True
        with open(f"{rel_data_path}/duration.json", "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        durations = data_dict["duration"]
        train_dataset = CustomDataset(
            train_dataset,
            durations=durations,
            preprocessed_mel=preprocessed_mel,
            mel_spec_module=mel_spec_module,
            **mel_spec_kwargs,
        )

    elif dataset_type == "CustomDatasetPath":
        try:
            train_dataset = load_from_disk(f"{dataset_name}/raw")
        except:  # noqa: E722
            train_dataset = Dataset_.from_file(f"{dataset_name}/raw.arrow")

        with open(f"{dataset_name}/duration.json", "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        durations = data_dict["duration"]
        train_dataset = CustomDataset(
            train_dataset, durations=durations, preprocessed_mel=preprocessed_mel, **mel_spec_kwargs
        )

    elif dataset_type == "HFDataset":
        print(
            "Should manually modify the path of huggingface dataset to your need.\n"
            + "May also the corresponding script cuz different dataset may have different format."
        )
        data_files = glob.glob(os.path.join(dataset_name, '*.arrow')) + glob.glob(
            os.path.join(dataset_name, '*.parquet'))
        if data_files[0].endswith('.arrow'):
            ds_format = 'arrow'
        else:
            ds_format = 'parquet'
        if ds_format == 'arrow':
            train_dataset = datasets.load_from_disk(dataset_name).train_test_split(0.01, shuffle=False)
        else:
            ds_cache_dir = '/home/projects/u6554606/llm/ds_cache_dir'
            train_dataset = datasets.load_dataset(path=ds_format if ds_format == 'parquet' else dataset_name,
                                             data_files=data_files if ds_format == 'parquet' else None,
                                             split='train',
                                             streaming=False,
                                             cache_dir=ds_cache_dir,
                                             num_proc=40).with_format('torch')
        train_dataset = train_dataset.rename_column('texts', 'text')
        train_dataset = train_dataset.rename_column('labels', 'mel_spec')
        # train_dataset['mel_spec'] = train_dataset['mel_spec'].transpose(-1, -2)
        # del train_dataset['input_ids'], train_dataset['speaker_embeddings']
        train_dataset.remove_columns(['input_ids', 'speaker_embeddings'])


    return train_dataset


# collation


def collate_fn(batch):
    mel_specs = [item["mel_spec"].squeeze(0).transpose(-1, -2) for item in batch]
    mel_lengths = torch.LongTensor([spec.shape[-1] for spec in mel_specs])
    max_mel_length = mel_lengths.amax()

    padded_mel_specs = []
    for spec in mel_specs:  # TODO. maybe records mask for attention here
        padding = (0, max_mel_length - spec.size(-1))
        padded_spec = F.pad(spec, padding, value=0)
        padded_mel_specs.append(padded_spec)

    mel_specs = torch.stack(padded_mel_specs)

    text = [item["text"] for item in batch]
    text_lengths = torch.LongTensor([len(item) for item in text])

    return dict(
        mel=mel_specs,
        mel_lengths=mel_lengths,
        text=text,
        text_lengths=text_lengths,
    )
