import h5py

filename = "features/matches.h5"

with h5py.File(filename, "r") as f:
    # Print all root-level object names (aka keys)
    keys = list(f.keys())
    print("Keys:", keys)

    for key in keys:
        try:
            obj = f[key]
            print(f"Processing key: {key}, Type: {type(obj)}")
            if isinstance(obj, h5py.Group):
                print(f"'{key}' is a group with the following objects:")
                print(list(obj.keys()))
            elif isinstance(obj, h5py.Dataset):
                print(f"'{key}' is a dataset with the following values:")
                print(obj[()])  # This will print the contents of the dataset
        except (KeyError, ValueError, OSError) as e:
            print(f"Error accessing '{key}': {e}")

# Example: if you want to process a specific group or dataset, you can do something like this:
try:
    a_group_key = keys[0]
    with h5py.File(filename, "r") as f:
        obj = f[a_group_key]
        if isinstance(obj, h5py.Group):
            # If the key is a group name, get the object names in the group and return as a list
            data = list(obj.keys())
            print(f"Group '{a_group_key}' contains:", data)
        elif isinstance(obj, h5py.Dataset):
            # If the key is a dataset name, get the dataset values and return as a list
            data = obj[()]  # returns as a numpy array
            print(f"Dataset '{a_group_key}' contains:", data)
except (KeyError, ValueError, OSError) as e:
    print(f"Error processing '{a_group_key}': {e}")
