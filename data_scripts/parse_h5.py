import h5py

# Open the H5 file
file_path = '/home/tgao/stompy-live/trajectory.h5'

def h5_to_dict(h5_file_path):
    data_map = {}
    with h5py.File(h5_file_path, 'r') as file:
        def recurse_through_groups(h5_element, container):
            if isinstance(h5_element, h5py.Dataset):
                container[h5_element.name.split('/')[-1]] = h5_element[:]
            elif isinstance(h5_element, h5py.Group):
                for key in h5_element.keys():
                    container[key] = {}
                    recurse_through_groups(h5_element[key], container[key])

        recurse_through_groups(file, data_map)
    return data_map


data_dict = h5_to_dict(file_path)

print(data_dict['traj_0'].keys())
print(data_dict['traj_0']['env_states']['actors'].keys())
