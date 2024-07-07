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

def create_image_grid(images, rows, cols, thumbnail_size=(100, 100)):
    # Calculate the grid size
    grid_width = cols * thumbnail_size[0]
    grid_height = rows * thumbnail_size[1]
    grid_image = Image.new('RGB', (grid_width, grid_height))

    # Resize images and paste them into the grid
    index = 0
    for image in images:
        
        image = np.asarray(image).astype(np.uint8)
        image = Image.fromarray(image)
    
        if index >= rows * cols:
            break  # Stop if we have filled the grid
        # Resize the image
        img = image.resize(thumbnail_size)
        # Calculate the position to paste
        x_offset = (index % cols) * thumbnail_size[0]
        y_offset = (index // cols) * thumbnail_size[1]
        grid_image.paste(img, (x_offset, y_offset))
        index += 1

    return grid_image

if __name__ == '__main__':
    # PATH = '/ephemeral/users/tgao/data/cube_with_masked_images_v1.h5'
    PATH = '/ephemeral/users/tgao/data/cube_step_angles_brown_table.h5'
    
    # save_steps('/ephemeral/users/tgao/data', PATH)
    
    images, actions = load_steps(PATH)
    
    image = create_image_grid(images, 4, 5)
    image.save("example_grid.png")
    
    # image_to_display = images[]
    # # image_to_display = images[420]
    
    # image = np.asarray(image_to_display).astype(np.uint8)
    # image = Image.fromarray(image)
    
    # image.save("example.png")
        