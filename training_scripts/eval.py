"""
evaluate.py

Script for evaluating a fine-tuned OpenVLA model on validation data.

Notes:
    - Requires PEFT (`pip install peft==0.11.1`)
    - Make sure the fine-tuned model is available locally or via a path

Run with:
    - [Single Node Multi-GPU (= $K) ]: torchrun --standalone --nnodes 1 --nproc-per-node $K evaluate.py \
                                    --data_root_dir <PATH/TO/VALIDATION/DATASETS/DIRECTORY> \
                                    --dataset_name <DATASET_NAME> \
                                    --run_root_dir <PATH/TO/LOGS/DIR> \
                                    ...
"""

"""
finetune.py

Simple script for parameter-efficient fine-tuning of OpenVLA models loaded through the HuggingFace AutoClasses, using
HuggingFace PEFT library for low-rank adaptation (LoRA).

Notes & Benchmarks:
    - Requires PEFT (`pip install peft==0.11.1`)
    - LoRA fine-tuning (see parameters below -- no quantization, LoRA rank = 32, target_modules = all-linear):
        + One 48 GB GPU can fit a Batch Size of 12
        + One 80 GB GPU can fit a Batch Size of 24

Run with:
    - [Single Node Multi-GPU (= $K) ]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py
    - [Override Config Values]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py \
                                    --data_root_dir <PATH/TO/RLDS/DATASETS/DIRECTORY> \
                                    --dataset_name <DATASET_NAME> \
                                    --run_root_dir <PATH/TO/LOGS/DIR> \
                                    ...
"""

import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import draccus
import torch
import torch.distributed as dist
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from datasets import PushCubeDataset

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class EvaluateConfig:
    data_path = '/ephemeral/users/tgao/val_data/brown_table_val.h5' # Path to h5 file
    
    # fmt: off
    # vla_path: str = "openvla/openvla-7b"
    vla_path: str = "/ephemeral/users/tgao/model_angles/cube_step_angles_brown_table"
    # vla_path: str = "/ephemeral/users/tgao/model_angles/openvla-7b+PushCubeDataset+b16+lr-2e-05cube_step_angles_brown_table"
    
    # Directory Paths
    # data_root_dir: Path = Path("/ephemeral/users/tgao/data/open-x-embodiment")        # Path to Open-X dataset directory
    dataset_name: str = "PushCubeDataset"                                # Name of evaluation dataset (e.g., `droid_wipe`)
    run_root_dir: Path = Path("/ephemeral/users/tgao/model_angles/")                               # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("/ephemeral/users/tgao/adapter-tmp")                     # Temporary directory for LoRA weights before fusing

    # Evaluation Parameters
    batch_size: int = 16                                            # Evaluation batch size

    # LoRA Arguments
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = False                                  # Whether to 4-bit quantize VLA for evaluation

    # fmt: on


@draccus.wrap()
def evaluate(cfg: EvaluateConfig) -> None:
    print(f"Evaluating OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Evaluation assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()
    
    # Configure Unique Experiment ID & Log Directory
    exp_id = cfg.data_path.split('/')[-1].split('.')[0]
    
    # run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    # os.makedirs(run_dir, exist_ok=True)

    # Quantization Config =>> only if LoRA fine-tuning
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized evaluation only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    # [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)

    # Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training
    vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Load Evaluation Dataset
    vla_dataset = PushCubeDataset(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
        data_path = cfg.data_path
    )
    
    # If using pre-formatted RLDS Dataset
    # vla_dataset = RLDSDataset(
    #     cfg.data_root_dir,
    #     cfg.dataset_name,
    #     batch_transform,
    #     resize_resolution=tuple(vla.module.config.image_sizes),
    #     shuffle_buffer_size=cfg.shuffle_buffer_size,
    #     image_aug=cfg.image_aug,
    # )

    # Create Collator and DataLoader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )

    # Evaluation Loop
    vla.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        with tqdm.tqdm(total=len(dataloader), desc="Evaluating", leave=True) as progress:
            for batch_idx, batch in enumerate(dataloader):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = vla(
                        input_ids=batch["input_ids"].to(device_id),
                        attention_mask=batch["attention_mask"].to(device_id),
                        pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                        labels=batch["labels"],
                    )
                    loss = output.loss
                    total_loss += loss.item()

                    action_logits = output.logits[:, vla.module.vision_backbone.featurizer.patch_embed.num_patches : -1]
                    action_preds = action_logits.argmax(dim=2)
                    action_gt = batch["labels"][:, 1:].to(action_preds.device)
                    mask = action_gt > action_tokenizer.action_token_begin_idx

                    correct_preds = (action_preds == action_gt) & mask
                    total_correct += correct_preds.sum().item()
                    total_samples += mask.sum().item()

                # Compute average loss and accuracy so far
                avg_loss_so_far = total_loss / (batch_idx + 1)
                accuracy_so_far = total_correct / total_samples

                # Update the tqdm progress bar with current loss and accuracy
                progress.set_postfix(loss=avg_loss_so_far, accuracy=accuracy_so_far)
                progress.update()

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples

    print(f"Final Evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    # Log final evaluation metrics
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    evaluate()

# torchrun --standalone --nnodes 1 --nproc-per-node 8 eval.py