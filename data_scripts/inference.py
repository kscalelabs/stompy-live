import os
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel
from PIL import Image
import numpy as np
import json
# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Inference Configuration
class InferenceConfig:
    model_path = '/ephemeral/users/tgao/model_angles/openvla-7b+PushCubeDataset+b16+lr-2e-05cube_step_angles+lora-r32+dropout-0.0+PushCubeDataset+b16+lr-2e-05cube_step_angles_brown_table+lora-r32+dropout-0.0'  # Path to the fine-tuned model
    adapter_dir = '/ephemeral/users/tgao/adapter-tmp/openvla-7b+PushCubeDataset+b16+lr-2e-05cube_step_angles+lora-r32+dropout-0.0+PushCubeDataset+b16+lr-2e-05cube_step_angles_brown_table+lora-r32+dropout-0.0'
    json_path = '/ephemeral/users/tgao/data/episode-0.json'  # Path to the input image for inference
    prompt = 'Describe the scene in the image.'  # Text prompt for inference
    use_lora = True  # Whether the model was fine-tuned using LoRA

# Load the Processor and Model
def load_model_and_processor(cfg: InferenceConfig):
    processor = AutoProcessor.from_pretrained(cfg.model_path, trust_remote_code=True)
    
    model = AutoModelForVision2Seq.from_pretrained(
        cfg.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    if cfg.use_lora:
        model = PeftModel.from_pretrained(model, cfg.adapter_dir)
        model = model.merge_and_unload()

    model.eval()
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    return model, processor

# Perform Inference
def perform_inference(cfg: InferenceConfig):
    model, processor = load_model_and_processor(cfg)
    
    with open(cfg.json_path, 'r') as file:
        cur_episode = json.load(file)
        
    image = np.asarray(cur_episode[0]['image']).astype(np.uint8)
    image = Image.fromarray(image)
    inputs = processor(images=image, return_tensors="pt").to('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Running inference...")
    
    # Perform inference
    with torch.no_grad():
        outputs = model.generate(
            pixel_values=inputs['pixel_values'],
            max_length=50,
            num_beams=5,
            early_stopping=True
        )
    
    # Decode the outputs
    decoded_outputs = processor.batch_decode(outputs, skip_special_tokens=True)
    return decoded_outputs

if __name__ == "__main__":
    cfg = InferenceConfig()
    results = perform_inference(cfg)
    print("Inference Results:", results)

# aws s3 sync s3://dpsh-models/vla/data/ .