import os

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

import h5py

with h5py.File("v2/models/keras_model.h5", "r+") as f:
    config = f.attrs["model_config"]
    if '"groups": 1,' in config:
        config = config.replace('"groups": 1,', '')
        f.attrs.modify("model_config", config)
        
print("✅ Patched keras model!")
