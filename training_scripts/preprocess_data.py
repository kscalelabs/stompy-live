import os
import json
import numpy as np
from tqdm import tqdm
import h5py


def save_steps(file_path, save_dir):
    
    length = 0
    
    for filename in tqdm(os.listdir(file_path)):
        
        json_file_path = os.path.join(file_path, filename)
        
        if "episode" in filename:
            with open(json_file_path, 'r') as file:
                cur_episode = json.load(file)
            
            length += len(cur_episode)  
            
    print("=" * 20)
    print("total length", length)
    print("=" * 20)
    
    res_imgs = np.empty((length, 512, 512, 3))
    res_actions = np.empty((length, 7))
    print(res_imgs.shape)
    
    res_imgs = []
    res_actions = []

    print("Reading json files...")
    idx = 0
    for filename in tqdm(os.listdir(file_path)):
        
        json_file_path = os.path.join(file_path, filename)
        
        if "episode" in filename:

            num = int(filename.split('-')[1].split('.')[0])
            
            # print(filename, num)
            
            # if num >= 2500:
            #     continue
            
            with open(json_file_path, 'r') as file:
                cur_episode = json.load(file)
            
            for episode_step in cur_episode:     
                # res_imgs.append(episode_step["image"])
                # res_actions.append(episode_step["action"])
                
                res_imgs[idx] = episode_step["image"]
                res_actions[idx] = episode_step["action"][0]
                idx += 1
                
                
    
    with h5py.File(save_dir, 'w') as hdf:
        hdf.create_dataset('images', data=np.array(res_imgs))
        hdf.create_dataset('actions', data=np.array(res_actions)) 
        
    print(save_dir)
    
def load_steps(file_path):
    with h5py.File(file_path, 'r') as hdf:
        images = hdf['images'][:]
        # images = hdf['masked_images'][:]
        
        actions = hdf['actions'][:]
        
        return (images, actions)



if __name__ == '__main__':
    PATH = '/ephemeral/users/tgao/data/random_start_v2.h5'
    save_steps('/ephemeral/users/tgao/data', PATH)
    
    images, actions = load_steps(PATH)
    print(images.shape, actions.shape)