import os
import yaml
import itertools
import subprocess
import re
import pandas as pd
from tqdm import tqdm


search_space = {
    # Core Parameters
    "BATCH_SIZE": [24],
    "LR": [0.00001],
    "MAX_SAMPLE_INTERVAL": [150, 200],
    # Data Augmentation Parameters
    "SEARCH_CENTER_JITTER": [3, 4],
    # 'TEMPLATE_CENTER_JITTER': [0.0, 0.1],
    "SCALE_JITTER": [0.5, 0.75],
    "EPOCH": [25],
    "BACKBONE_MULTIPLIER": [0.1],
    "DROP_PATH_RATE": [0.1, 0.2],
    "GIOU_WEIGHT": [2.0],
    "L1_WEIGHT": [5.0],
    "SAMPLE_PER_EPOCH": [20_000],
}

BASE_CONFIG_PATH = "experiments/abavitrack/abavit_patch16_224.yaml"
LOG_DIR = "outputs/logs/abavitrack"


def generate_combinations(space):
    keys, values = zip(*space.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


def update_yaml(base_path, new_path, params):
    with open(base_path, "r") as file:
        config = yaml.safe_load(file)

    config["TRAIN"]["BATCH_SIZE"] = params["BATCH_SIZE"]
    config["TRAIN"]["LR"] = params["LR"]
    config["DATA"]["MAX_SAMPLE_INTERVAL"] = params["MAX_SAMPLE_INTERVAL"]
    config["DATA"]["SEARCH"]["SCALE_JITTER"] = params["SCALE_JITTER"]

    config["DATA"]["SEARCH"]["CENTER_JITTER"] = params["SEARCH_CENTER_JITTER"]
    # config['DATA']['TEMPLATE']['CENTER_JITTER'] = params['TEMPLATE_CENTER_JITTER']

    config["TRAIN"]["BACKBONE_MULTIPLIER"] = params["BACKBONE_MULTIPLIER"]
    config["TRAIN"]["DROP_PATH_RATE"] = params["DROP_PATH_RATE"]
    config["TRAIN"]["GIOU_WEIGHT"] = params["GIOU_WEIGHT"]
    config["TRAIN"]["L1_WEIGHT"] = params["L1_WEIGHT"]
    config["TRAIN"]["EPOCH"] = params["EPOCH"]

    with open(new_path, "w") as file:
        yaml.dump(config, file, default_flow_style=False)


def parse_best_iou(log_path):
    best_iou = 0.0
    best_epoch = 0
    # Updated Regex
    regex = r"\[val:\s*(\d+)[^\]]*\].*?IoU:\s*([\d\.]+)"

    if not os.path.exists(log_path):
        return None, None

    with open(log_path, "r") as file:
        for line in file:
            match = re.search(regex, line)
            if match:
                epoch = int(match.group(1))
                iou = float(match.group(2))
                if iou > best_iou:
                    best_iou = iou
                    best_epoch = epoch

    return best_iou, best_epoch


def main():
    combinations = generate_combinations(search_space)
    results = []

    print("\n=======================================================")
    print(f" Starting Grid Search for {len(combinations)} combinations")
    print("=======================================================\n")

    pbar = tqdm(total=len(combinations), desc="Grid Search Progress", unit="run")

    for i, params in enumerate(combinations):
        config_name = f"abavit_gs_{i}"
        yaml_path = f"experiments/abavitrack/{config_name}.yaml"
        log_path = os.path.join(LOG_DIR, f"abavitrack-{config_name}.log")

        update_yaml(BASE_CONFIG_PATH, yaml_path, params)


        cmd = [
            "python",
            "tracking/train.py",
            "--script",
            "abavitrack",
            "--config",
            config_name,
            "--save_dir",
            "./outputs",
            "--mode",
            "single",
        ]

        subprocess.run(cmd, stdout=subprocess.DEVNULL)

        best_iou, best_epoch = parse_best_iou(log_path)

        params["Config_Name"] = config_name
        params["Best_Val_IoU"] = best_iou
        params["Best_Epoch"] = best_epoch
        results.append(params.copy())

        df = pd.DataFrame(results)
        df.to_csv("grid_search_results.csv", index=False)

        pbar.set_postfix({"Last_IoU": best_iou, "Config": config_name})
        pbar.update(1)

    pbar.close()
    print("\nGrid Search Complete! Results saved to grid_search_results_gs_0.csv")
    print("\nTop 3 Configurations:")
    print(df.sort_values(by="Best_Val_IoU", ascending=False).head(3))


if __name__ == "__main__":
    main()
