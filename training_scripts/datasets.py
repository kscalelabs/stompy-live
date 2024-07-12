from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Type

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import tree_map
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.rlds import make_interleaved_dataset, make_single_dataset
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights
from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType

import os
import json
import h5py
import random
from tqdm import tqdm

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

# def rescale()

# def get_steps(file_path):
#     res = [] # Sequence of steps: {image: 224x224x3, action: 7d-vector}

#     print("Reading json files...")
#     for filename in tqdm(os.listdir(file_path)):
#         json_file_path = os.path.join(file_path, filename)
        
#         if "episode" in filename:
#             # print(filename)
            
#             with open(json_file_path, 'r') as file:
#                 cur_episode = json.load(file)
            
#             # rescaled_image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
#             for episode_step in cur_episode:                
#                 res.append({
#                     "action": episode_step["action"],
#                     "image": episode_step["image"]})
                
#         break
    
#     print("Done", len(res), "number of samples")
    
#     return res

def load_steps(file_path):
    with h5py.File(file_path, 'r') as hdf:
        images = hdf['images'][:]
        # images = hdf['masked_images'][:]
        
        actions = hdf['actions'][:]
        
        return (images, actions)

def get_steps(file_path):
    print("Loading data...")
    images, actions = load_steps(file_path)
    print("Done loading")
    print(images.shape, actions.shape)
    return images, actions

class PushCubeDataset(Dataset):
    def __init__(
        self,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
        data_path: str
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn
        self.instruction = "push the cube to the target"
        
        print(f"What action should the robot take to {self.instruction}?")
        
        # self.images, self.actions = get_steps("/ephemeral/users/tgao/data/stompy-live/data/data")
        # self.LEN = len(self.steps)
        
        print("Path to Data", data_path)
        
        self.images, self.actions = get_steps(data_path)
        self.LEN = self.images.shape[0]
        numbers = list(range(self.LEN))
        random.shuffle(numbers)
        self.permutation = numbers
        
        # Note =>> We expect the dataset to store statistics for action de-normalization. Specifically, we store the
        # per-dimension 1st and 99th action quantile. The values below correspond to "no normalization" for simplicity.
        self.dataset_statistics = {
            "PushCubeDataset": {
                "action": {"q01": np.zeros((7,), dtype=np.float32), "q99": np.ones((7,), dtype=np.float32)}
            }
        }

    def __len__(self):
        return self.LEN

    def __getitem__(self, idx):
        idx = self.permutation[idx]
        
        # cur_info = self.steps[idx]
        cur_image = self.images[idx]
        cur_action = self.actions[idx]
        
        # image = Image.fromarray(np.asarray(np.random.rand(224, 224, 3) * 255.0, dtype=np.uint8))
        # action = np.asarray(np.random.rand(7), dtype=np.float32)
        
        # print(cur_info["image"].shape)
        
        image = np.asarray(cur_image).astype(np.uint8)
        image = Image.fromarray(image)
        
        action = np.asarray(cur_action)
                
        instruction = self.instruction

        # Add instruction to VLA prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)[0]},
        ]
        
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
