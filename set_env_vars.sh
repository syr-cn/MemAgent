export GLOO_SOCKET_IFNAME=eth0,eth1
# check device via `ip link show`
wandb_token="8c63841d0875e4fde65a42fb47b52e6a18b8a1ed"
export WANDB_MODE="disabled"
export WANDB_API_KEY=$wandb_token
export WAND_PROJECT="memory-agent"
export HF_ENDPOINT="https://hf-mirror.com"
export PYTHONPATH="/mnt/finder/shiyr/code/Mem/MemAgent:$PYTHONPATH"

# On main node:
## ray start --head --dashboard-host=0.0.0.0
### or 
## CUDA_VISIBLE_DEVICES=0,1,2,3 ray start --head --dashboard-host=0.0.0.0 --num-gpus=4

# On associated nodes:
### connect to the GCS address of main server
## ray start --address='10.249.217.198:6379'