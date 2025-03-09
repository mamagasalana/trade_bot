import os
import json
import datetime
import subprocess
import shutil

def main():
    # Load configuration from config.json (assumed in same directory as main.py)
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Create a unique output folder using the current timestamp
    folder_name = "test_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(folder_name, exist_ok=True)
    
    script = config.get("script")
    pairs = config.get("pairs", ["CADCHF", "USDCAD"])
    na_threshold = config.get("na_threshold", 0.9)
    exclude_weekend = config.get("exclude_weekend", 1)
    
    main_params = [
        "--pairs", ",".join(pairs),
        "--na_threshold", str(na_threshold),
        "--exclude_weekend", str(exclude_weekend)
    ]
    
    shutil.copy(script, folder_name)
    # Change current working directory into the newly created folder
    os.chdir(folder_name)
    print("Changed working directory to:", os.getcwd())
    # Construct the absolute path to test.py (assumed to be in the same directory as main.py)
    test_script = os.path.join(os.path.dirname(__file__), "test.py")
    
    # Run test.py (its output will be saved to the current working directory)
    subprocess.run(["python", test_script] + main_params, check=True)
    

if __name__ == "__main__":
    main()
