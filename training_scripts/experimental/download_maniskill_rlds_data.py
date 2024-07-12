import tensorflow_datasets as tfds
import tqdm

# List of datasets to download
DATASET_NAMES = ['maniskill_dataset_converted_externally_to_rlds']
DOWNLOAD_DIR = '/ephemeral/users/tgao'

print(f"Downloading {len(DATASET_NAMES)} datasets to {DOWNLOAD_DIR}.")

for dataset_name in tqdm.tqdm(DATASET_NAMES):
    dataset, info = tfds.load(dataset_name, with_info=True, data_dir=DOWNLOAD_DIR, as_supervised=True, download=True )