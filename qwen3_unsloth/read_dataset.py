from datasets import load_dataset
from transformers import DeepseekV3ForCausalLM
from transformers import Qwen3ForCausalLM, Qwen3Model

# Login using e.g. `huggingface-cli login` to access this dataset
# ds = load_dataset("nvidia/OpenMathReasoning")

#

# modelscope download --model unsloth/Qwen3-0.6B-unsloth-bnb-4bit --local_dir .