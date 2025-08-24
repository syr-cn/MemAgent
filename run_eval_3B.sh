export PROJECT_ROOT="/home/hexngroup/shiyr/Mem/MemAgent"
export EXP_NAME="MemoryAgent-3B-reproduce"
export MODEL_ROOT="$PROJECT_ROOT/results/memory_agent/memory_agent_3B_n4/global_step_750/actor"
export MODEL_TP=4

export DATAROOT="$PROJECT_ROOT/data/hotpotqa"

export MODEL_NAME="$MODEL_ROOT/hf_ckpt"
if [ ! -d "$MODEL_NAME" ] || [ ! -f "$MODEL_NAME/vocab.json" ]; then
    bash $PROJECT_ROOT/tmp-hanhai-script/merge_ckpt.sh $MODEL_ROOT
    echo "Merged checkpoint to $MODEL_NAME successfully"
else
    echo "$MODEL_NAME already exist"
fi

python taskutils/memory_eval/run_custom.py 2>&1 | tee $PROJECT_ROOT/log/eval_$EXP_NAME.log