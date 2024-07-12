# import h5py
# import numpy as np

# # Sample data: list of dictionaries
# data_list = [
#     {"action": np.random.random(10).tolist(), "image": np.random.random((100, 100)).tolist()},
#     {"action": np.random.random(10).tolist(), "image": np.random.random((100, 100)).tolist()},
#     # Add more dictionaries as needed
# ]

# # Create a new HDF5 file
# with h5py.File('data.h5', 'w') as hdf:
#     for i, data_dict in enumerate(data_list):
#         group = hdf.create_group(f'entry_{i}')
#         for key, value in data_dict.items():
#             group.create_dataset(key, data=value)

# print('Data has been written to data.h5')

import h5py

# Read the HDF5 file
data_list_read = []
with h5py.File('data.h5', 'r') as hdf:
    for group_name in hdf.keys():
        data_dict = {}
        group = hdf[group_name]
        for key in group.keys():
            data_dict[key] = group[key][:]
        data_list_read.append(data_dict)

print('Data has been read from data.h5')
print(data_list_read)
