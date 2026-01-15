微调
全量微调
高效微调
LoRA/QLoRA
 - 对话风格
 - 知识灌注
 - 推理能力提升
 - Agent能力提升

OpenMathReasoning：带思考过程
FineTome-100k：普通问答

主流微调工具
unsloth：单机单卡微调，动态量化，适用于个人配置
Llama-Factory：单机多卡，使用工业级
ms-SWIFT：轻量级框架，支持单机多卡，目前适配有限
ColossalAI：工业级微调框架，支持多机多卡
强化学习
veRL
OpenRLHF
模型性能评估框架：EvalScope

怎样准备微调数据集？



Qwen0.5B

im_start
im_end
think
function calling 
混合

Qwen3混合推理

vllm serve lanyun-tmp/model/cache/Qwen3-8B-unsloth-bnb-4bit --enable-auto-tool-choice --tool-call-parser hermes --port 12500

evalscope perf \
    --url "http://b328aaf2fed84ff4b489cc95b9fd4d3a.qhdcloud.lanyun.net:12500/v1/chat/completions"
    --
