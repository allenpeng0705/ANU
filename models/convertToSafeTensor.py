# Got a bunch of .ckpt files to convert?
# Here's a handy script to take care of all that for you!
# Original .ckpt files are not touched!
# Make sure you have enough disk space! You are going to DOUBLE the size of your models folder!
#
# First, run:
# pip install torch torchsde==0.2.5 safetensors==0.2.5
#
# Place this file in the **SAME DIRECTORY** as all of your .ckpt files, open a command prompt for that folder, and run:
# python convert_to_safe.py

# Original script https://gist.github.com/xrpgame/8f756f99b00b02697edcd5eec5202c59
# Edited by @Tumppi066 for use with folders https://github.com/Tumppi066/

import os
import torch
from safetensors.torch import save_file

files = os.listdir()

# Loop through all files in the folder to find the .ckpt files
models = []
safeTensors = []
for path, subdirs, files in os.walk(os.path.abspath(os.getcwd())):
    for name in files:
        if name.lower().endswith('.ckpt'):
            models.append(os.path.join(path, name))
        if name.lower().endswith('.safetensors'):
            safeTensors.append(os.path.join(path, name))

if len(models) == 0:
    print('\033[91m> No .ckpt files found in this directory ({}).\033[0m'.format(os.path.abspath(os.getcwd())))
    input('> Press enter to exit... ')
    exit()
print(f"\n\033[92m> Found {len(models)} .ckpt files to convert.\033[0m")
for model in models:
    print(str(models.index(model)+1) +": "+ model.split("\\")[-1])

input("> Press enter to continue... ")
print("\n")

for index in range(len(models)):
    f = models[index]
    modelName = f.split("\\")[-1] # This is for easy printing (without printing the full path)
    tensorName = f"{modelName.replace('.ckpt', '')}.safetensors"
    fn = f"{f.replace('.ckpt', '')}.safetensors"

    if fn in safeTensors:
        # Print the model name and skip it if it already exists in yellow
        print(f"\033[33m\n> Skipping {modelName}, as {tensorName} already exists.\033[0m")
        continue
    
    print(f'\n> Loading {modelName} ({index+1}/{len(models)})...')

    try:
        with torch.no_grad():
            map_location = torch.device('cpu')
            weights = torch.load(f, map_location=map_location)["state_dict"]
            fn = f"{f.replace('.ckpt', '')}.safetensors"
            print(f'Saving {tensorName}...')
            save_file(weights, fn)
    except Exception as ex:
        print(f'ERROR converting {modelName}: {ex}')

print("\n\033[92mDone!\033[0m")
input("> Press enter to exit... ")
exit()