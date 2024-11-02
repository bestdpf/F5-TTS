export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=10
export DS_ACCELERATOR="cuda"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export PATH=$PATH:/home/remote/u6554606/fb-project/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/remote/u6554606/fb-project/lib


python train.py > train.out 2>train.err&