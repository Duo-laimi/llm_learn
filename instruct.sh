vllm serve lanyun-tmp/model/cache/Qwen3-8B-unsloth-bnb-4bit \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --port 12500

evalscope perf \
    --url "http://b328aaf2fed84ff4b489cc95b9fd4d3a.qhdcloud.lanyun.net:12500/v1/chat/completions" \
    --parallel 5 \
    --model lanyun-tmp/model/cache/Qwen3-8B-unsloth-bnb-4bit \
    --api openai \
    --dataset openqa \
    --stream


evalscope perf \
    --url "http://localhost:12500/v1/chat/completions" \
    --parallel 5 \
    --model lanyun-tmp/model/cache/Qwen3-8B-unsloth-bnb-4bit \
    --api openai \
    --dataset openqa \
    --stream

