export HF_ENDPOINT="https://hf-mirror.com"

vllm serve BytedTsinghua-SIA/RL-MemoryAgent-7B --tensor_parallel_size 4 --gpu-memory-utilization 0.8 | tee log/vllm.log