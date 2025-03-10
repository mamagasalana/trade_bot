import os
import json
import datetime
import subprocess
import shutil
import time

def main():
    start= time.time()
    # Load configuration from config.json (assumed in same directory as main.py)
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Create a unique output folder using the current timestamp
    folder_name = "test_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(folder_name, exist_ok=True)
    
    script = config.get("script")
    pairs = config.get("pairs")
    na_threshold = config.get("na_threshold", 0.9)
    exclude_weekend = config.get("exclude_weekend", 1)
    
    main_params = [
        "--pairs", ",".join(pairs),
        "--na_threshold", str(na_threshold),
        "--exclude_weekend", str(exclude_weekend), 
        "--folder", folder_name
    ]
    
    shutil.copy(script, folder_name)
    shutil.copy(config_path, folder_name)
    test_script = os.path.join(os.path.dirname(__file__), script)
    
    # Run test.py (its output will be saved to the current working directory)
    subprocess.run(["python", test_script] + main_params, check=True)
    print("Time taken: %s" % (time.time()-start))

if __name__ == "__main__":
    main()
