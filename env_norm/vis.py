import os
import json
import numpy as np
from tqdm import tqdm
import h5py


def save_steps(file_path, save_dir):
    res_imgs = []
    res_actions = []

    print("Reading json files...")
    for filename in tqdm(os.listdir(file_path)):
        
        json_file_path = os.path.join(file_path, filename)
        
        if "episode" in filename:
            
            with open(json_file_path, 'r') as file:
                cur_episode = json.load(file)
            
            for episode_step in cur_episode:     
                res_imgs.append(episode_step["image"])
                res_actions.append(episode_step["action"])
                
                
    
    with h5py.File(save_dir, 'w') as hdf:
        hdf.create_dataset('images', data=np.array(res_imgs))
        hdf.create_dataset('actions', data=np.array(res_actions)) 
        
    print(save_dir)
    
def load_steps(file_path):
    with h5py.File(file_path, 'r') as hdf:
        # images = hdf['images'][:]
        images = hdf['masked_images'][:]
        actions = hdf['actions'][:]
        
        return (images, actions)


from PIL import Image

if __name__ == '__main__':
    PATH = '/ephemeral/users/tgao/data/cube_with_masked_images.h5'
    # save_steps('/ephemeral/users/tgao/data', PATH)
    
    images, actions = load_steps(PATH)
    
    image_to_display = images[0]
    
    image = np.asarray(image_to_display).astype(np.uint8)
    image = Image.fromarray(image)
    
    image.save("example.png")
        