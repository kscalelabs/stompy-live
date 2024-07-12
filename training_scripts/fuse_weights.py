import GPUtil
import os

def set_least_used_gpu():
    gpus = GPUtil.getGPUs()
    if not gpus:
        raise RuntimeError("No GPUs found.")
    least_used_gpu = min(gpus, key=lambda x: x.memoryUtil)
    gpu_id = least_used_gpu.id
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return gpu_id

# Set the least used GPU at the beginning of your script
gpu_id = set_least_used_gpu()
print(f"Using GPU {gpu_id}")

# =========================================================================================================

import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForVision2Seq

# Define paths
model_weights_path = "/ephemeral/users/tgao/model_angles/openvla-7b+PushCubeDataset+b16+lr-2e-05cube_step_angles+lora-r32+dropout-0.0+PushCubeDataset+b16+lr-2e-05cube_step_angles_brown_table+lora-r32+dropout-0.0"
lora_weights_path = "/ephemeral/users/tgao/adapter-tmp/openvla-7b+PushCubeDataset+b16+lr-2e-05cube_step_angles+lora-r32+dropout-0.0+PushCubeDataset+b16+lr-2e-05cube_step_angles_brown_table+lora-r32+dropout-0.0"
fused_weights_path = "/ephemeral/users/tgao/fused_weights/fused_model_weights.pth"

# Ensure the fused weights path exists
os.makedirs(fused_weights_path, exist_ok=True)

# Load the base model
model = AutoModelForVision2Seq.from_pretrained(
    model_weights_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
)

# Load the LoRA configuration and model
peft_config = PeftConfig.from_pretrained(lora_weights_path)
peft_model = PeftModel.from_pretrained(model, lora_weights_path)

# Merge the LoRA weights into the base model
peft_model.merge_and_unload()

# Save the merged model weights
peft_model.save_pretrained(fused_weights_path)

print(f"Fused weights saved to {fused_weights_path}")
