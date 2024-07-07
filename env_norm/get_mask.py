from GroundingDINOSAM import *
import h5py
from tqdm import tqdm

def get_mask_from_image(img : Image.Image, labels = ["a robot", "a blue block", "a striped target"], threshold = 0.3):

    detector_id = "IDEA-Research/grounding-dino-tiny"
    segmenter_id = "facebook/sam-vit-base"

    image_array, detections = grounded_segmentation(
        image=img,
        labels=labels,
        threshold=threshold,
        polygon_refinement=True,
        detector_id=detector_id,
        segmenter_id=segmenter_id
    )
    
    return annotate_numpy(image_array, detections)
    
    
def load_steps(file_path):
    with h5py.File(file_path, 'r') as hdf:
        images = hdf['images'][:]
        actions = hdf['actions'][:]
        
        return (images, actions)
    
def convert_data(file_path, output_path):
    images, actions = load_steps(file_path)
    res_h5 = []
    
    for img in tqdm(images):
        image = np.asarray(img).astype(np.uint8)
        image = Image.fromarray(image)
        
        res_h5.append(get_mask_from_image(image))
        
    with h5py.File(output_path, 'w') as hf:
        hf.create_dataset('masked_images', data=res_h5)
        hf.create_dataset('images', data=images)
        hf.create_dataset('actions', data=actions)
        

if __name__ == '__main__':
    convert_data('/ephemeral/users/tgao/data/cube_step_angles_brown_table.h5', 
                 '/ephemeral/users/tgao/data/cube_with_masked_images.h5')
        