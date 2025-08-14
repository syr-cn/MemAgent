ROOT="/mnt/finder/shiyr/code/Mem/MemAgent"
export HF_ENDPOINT="https://hf-mirror.com"

export DATAROOT="$ROOT/data/hotpotqa"
export MODELROOT="$ROOT/models"

cd $ROOT/taskutils/memory_eval
SERVE_PORT=8000 DASH_PORT=8265 python run.py 2>&1 | tee $ROOT/log/test.log

# python /mnt/finder/shiyr/code/Mem/MemAgent/serve/llm070.py --model BytedTsinghua-SIA/RL-MemoryAgent-7B --tp 1
# curl -m 100000000 http://127.0.0.1:8000/v1/models