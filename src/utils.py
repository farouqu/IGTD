import os
import numpy as np
import shutil

# Function to split data
def split_data(source, train_dir, val_dir, test_dir, train_size=0.7, val_size=0.2):
    files = [f for f in os.listdir(source) if os.path.isfile(os.path.join(source, f))]
    shuffled_files = np.random.permutation(files)

    train_end = int(train_size * len(shuffled_files))
    val_end = int(val_size * len(shuffled_files)) + train_end

    for file in shuffled_files[:train_end]:
        shutil.copy(os.path.join(source, file), os.path.join(train_dir, file))

    for file in shuffled_files[train_end:val_end]:
        shutil.copy(os.path.join(source, file), os.path.join(val_dir, file))

    for file in shuffled_files[val_end:]:
        shutil.copy(os.path.join(source, file), os.path.join(test_dir, file))