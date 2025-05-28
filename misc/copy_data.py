import os
import shutil
import json

# Load the JSON file into a dictionary
with open('../burstpaths.json') as f:
    data = json.load(f)

for name, burstpath in data.items():
    source_dir = f"/dataz/dsa110/candidates/{burstpath}/Level3/"
    dest_dir = f"{name}_{burstpath}"
    
    # Option 1: If dest_dir should not exist
    if not os.path.exists(dest_dir):
        shutil.copytree(source_dir, dest_dir)
    else:
        print(f"Destination {dest_dir} already exists. Skipping or handle accordingly.")
