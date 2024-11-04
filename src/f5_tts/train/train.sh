export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=10
export DS_ACCELERATOR="cuda"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export HF_DATASETS_IN_MEMORY_MAX_SIZE=130000000000
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export PATH=$PATH:/home/remote/u6554606/fb-project/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/remote/u6554606/fb-project/lib:/home/projects/u6554606/llm/f5/lib/
export PYTHONPATH=/home/projects/u6554606/llm/F5-TTS/src/third_party/BigVGAN
export PHONEMIZER_ESPEAK_LIBRARY=/home/remote/u6554606/fb-project/lib/libespeak-ng.so.1
export TOKENIZERS_PARALLELISM=false
export ACCELERATE_USE_DEEPSPEED=true


python train.py > train.out 2>train.err&