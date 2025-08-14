## Setup
### Installation

```bash
pip install httpx==0.23.1 aiohttp -U ray[serve,default] vllm

pip install nltk pyyaml beautifulsoup4 html2text wonderwords tenacity fire
pip install vllm==0.9 --index-url https://download.pytorch.org/whl/cu126
pip install hydra-core accelerate tensordict torchdata wandb "tensordict<=0.6.2"
```

version `GLIBC_2.32` not found 
https://github.com/modular/modular/issues/3684

```bash
deb http://th.archive.ubuntu.com/ubuntu jammy main
sudo apt update
apt install libc6
```

### Test
1. Prepare QA data
2. Preparing models
3. Preparing models
```bash
cd taskutils/memory_data
bash download_qa_dataset.sh

bash hfd.sh BytedTsinghua-SIA/hotpotqa --dataset --tool aria2c -x 10
export DATAROOT="/mnt/finder/shiyr/code/Mem/MemAgent/data/hotpotqa"

bash hfd.sh Qwen/Qwen2.5-7B-Instruct --model --tool aria2c -x 10
export MODELROOT="/mnt/finder/shiyr/code/Mem/MemAgent/models"
mv Qwen2.5-7B-Instruct $MODELROOT/Qwen2.5-7B-Instruct-128K
```

### Train
```bash
bash run_memory_7B.sh
```


## Components
- memory module: recurrent/impls/memory.py
- run_llm_loop: recurrent/generation_manager.py
- fit & validate: verl/trainer/ppo/ray_trainer.py
- main func: verl/trainer/main_ppo.py

## my script
```bash
#!/bin/bash
set -x

# at least 1 nodes, 4nodes=3~4days to converge
NNODES=1
NGPUS_PER_NODE=4
PROJ_ROOT="/mnt/finder/shiyr/code/Mem/MemAgent/results"
DATASET_ROOT="/mnt/finder/shiyr/code/Mem/MemAgent/data"


wandb_token="8c63841d0875e4fde65a42fb47b52e6a18b8a1ed"
export WANDB_MODE="disabled"
export WANDB_API_KEY=$wandb_token
export WAND_PROJECT="memory-agent"

# MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
export HF_ENDPOINT="https://hf-mirror.com"
MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct"
VAL_PATH="${DATASET_ROOT}/hotpotqa/hotpotqa_dev.parquet"
TRAIN_PATH="${DATASET_ROOT}/hotpotqa/hotpotqa_train_32k.parquet"
EXP=memory_agent/0.5B
EXP_LOG_NAME=memory_agent_0.5B
PROJ_DIR=${PROJ_ROOT}/${EXP}
export PYTHONPATH="/mnt/finder/shiyr/code/Mem/MemAgent:$PYTHONPATH"

# Please note that recurrent framewrok will use max_length defined in task config.
# These two values are just for vLLM to decide max_model_length.
MAXLEN=8192 
MAX_NEW_TOKEN=1024

LOG_PATH="/mnt/finder/shiyr/code/Mem/MemAgent/log/$EXP_LOG_NAME.log"
python3 -m verl.trainer.main_ppo \
    recurrent.enable=memory \
    recurrent.memory.config.chunk_size=5000 \
    algorithm.adv_estimator=grpo \
    algorithm.grpo_use_adv=False \
    trainer.save_freq=10 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    trainer.logger=['console','wandb'] \
    actor_rollout_ref.actor.optim.lr_warmup_steps=20 \
    actor_rollout_ref.actor.clip_ratio_high=0.20 \
    actor_rollout_ref.actor.entropy_coeff=0.000 \
    data.train_files=$TRAIN_PATH \
    data.val_files=$VAL_PATH \
    data.shuffle=False \
    data.filter_overlong_prompts=True \
    data.train_batch_size=32 \
    data.truncation='center' \
    +data.context_key='context' \
    data.max_prompt_length=$MAXLEN \
    data.max_response_length=$MAX_NEW_TOKEN \
    reward_model.reward_manager='thread' \
    actor_rollout_ref.model.path=$MODEL_PATH  \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32768 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384\
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=${EXP} \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=$NGPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.test_freq=5 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=$PROJ_DIR \
    trainer.total_epochs=30 \
    2>&1 | tee $LOG_PATH
```