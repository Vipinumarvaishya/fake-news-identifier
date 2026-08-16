# download_assets.py
import os
import subprocess

assets = {
    "Fake.csv": "1jwCFUHKX2hIv50V6BBoxjym8AWgc2j6Q",
    "True.csv": "1b4Izukx_Sfheu6pbdycaM6ByUUQt2YGB"
}

for filename, file_id in assets.items():
    if not os.path.exists(filename):
        print(f"⬇️ Downloading {filename}...")
        subprocess.run(["gdown", "--id", file_id, "-O", filename])
    else:
        print(f"✅ {filename} already exists.")