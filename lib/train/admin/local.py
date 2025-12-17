class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '{{PROJECT PATH}}'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '{{PROJECT PATH}}/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '{{PROJECT PATH}}/pretrained_networks'
        self.lasot_dir = '{{PROJECT PATH}}/data/LaSOT/zip'
        self.got10k_dir = '{{PROJECT PATH}}/data/GOT-10k/train'
        self.got10k_val_dir = '{{PROJECT PATH}}/data/got10k/val'
        self.lasot_lmdb_dir = '{{PROJECT PATH}}/data/lasot_lmdb'
        self.got10k_lmdb_dir = '{{PROJECT PATH}}/data/got10k_lmdb'
        self.trackingnet_dir = '{{PROJECT PATH}}/data/trackingnet'
        self.trackingnet_lmdb_dir = '{{PROJECT PATH}}/data/trackingnet_lmdb'
        self.coco_dir = '{{PROJECT PATH}}/data/coco/coco'
        self.coco_lmdb_dir = '{{PROJECT PATH}}/data/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '{{PROJECT PATH}}/data/vid'
        self.imagenet_lmdb_dir = '{{PROJECT PATH}}/data/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
