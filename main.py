import os
import json
import shutil
import subprocess
from datetime import datetime

class OutputTracker:
    def __init__(self, config_file="config.json"):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self):
        """Loads the configuration from the JSON file."""
        with open(self.config_file, "r") as f:
            return json.load(f)

    def execute_script(self, script, params):
        """Executes the given script with parameters."""
        command = ["python", script] + params
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            print(f"Executed {script} successfully:\n{result.stdout}")
        except subprocess.CalledProcessError as e:
            print(f"Error executing {script}: {e.stderr}")

    def save_config_and_script(self, preprocess_script):
        """Saves the current config and preprocessing script in a new folder inside 'output/'."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = os.path.join("output", f"preprocess_output_{timestamp}")
        os.makedirs(folder_name, exist_ok=True)

        # Save config.json
        config_path = os.path.join(folder_name, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=4)

        # Save preprocess script
        script_path = os.path.join(folder_name, os.path.basename(preprocess_script))
        shutil.copy(preprocess_script, script_path)

        print(f"Saved config and script to {folder_name}")

    def run(self):
        """Runs the preprocessing steps as per the config file."""
        preprocess_script = self.config["preprocess"]["script"]
        preprocess_params = self.config["preprocess"]["params"]

        self.execute_script(preprocess_script, preprocess_params)
        self.save_config_and_script(preprocess_script)

if __name__ == "__main__":
    manager = OutputTracker()
    manager.run()
