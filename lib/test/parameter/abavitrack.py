from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.abavitrack.config import cfg, update_config_from_file

import os


def parameters(yaml_name: str, test_epoch):
    params = TrackerParams()

    prj_dir = os.environ.get('PRJ_DIR', '/app')
    yaml_file = os.path.join(prj_dir, f'experiments/abavitrack/{yaml_name}.yaml')

    # Fallback to relative path just in case
    if not os.path.exists(yaml_file):
        yaml_file = os.path.join('experiments', 'abavitrack', f'{yaml_name}.yaml')

    update_config_from_file(yaml_file)
    cfg.TEST.EPOCH = test_epoch
    params.cfg = cfg
    print("test config: ", cfg)

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    env_checkpoint = os.environ.get('CHECKPOINT_PATH')

    if env_checkpoint and os.path.exists(env_checkpoint):
        params.checkpoint = env_checkpoint
        print(f"--> [Docker Mode] Using checkpoint from ENV: {params.checkpoint}")
    else:
        # Fallback to the original logic if not running in Docker
        save_dir = env_settings().save_dir
        params.checkpoint = os.path.join(
            save_dir,
            "checkpoints/train/abavitrack/%s/AbaViTrack_ep%04d.pth.tar" % (yaml_name, cfg.TEST.EPOCH)
        )
        print(f"--> [Local Mode] Using default checkpoint: {params.checkpoint}")

    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
