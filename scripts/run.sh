export LD_LIBRARY_PATH=~/env/lib64:$LD_LIBRARY_PATH
export LANG="en_US.UTF-8"
export CUDA_VISIBLE_DEVICES=2

python3 main.py ./configs/config.json
# export CUDA_VISIBLE_DEVICES=1,2
#python3 -m torch.distributed.launch main.py ./configs/config.json
# python3 -m torch.distributed.launch --nproc_per_node 2 main.py  \
#     ./configs/config.json