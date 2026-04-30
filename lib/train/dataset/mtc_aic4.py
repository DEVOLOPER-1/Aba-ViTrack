import os
import json
import torch
import numpy as np
import pandas as pd
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings


class MTCAIC4(BaseVideoDataset):
    """
    MTC-AIC4 training dataset.

    FIXES vs original
    -----------------
    * sequence_list now stores filesystem-safe names (slashes replaced by
      underscores) via safe_to_key mapping, mirroring the change in the
      evaluation dataset class.  This keeps both dataset wrappers consistent.
    * _read_bb_anno is protected against null annotation_path (public_lb
      sequences have no annotation — skip them gracefully during training).
    * Added proper __len__ override (BaseVideoDataset expects it).
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, split='train'):
        root = env_settings().mtc_aic4_dir if root is None else root
        super().__init__('MTC_AIC4', root, image_loader)

        self.split = split
        manifest_path = os.path.join(self.root, 'metadata', 'contestant_manifest.json')
        with open(manifest_path, 'r') as f:
            raw_manifest = json.load(f)[self.split]

        # Build safe-name ↔ manifest-key mapping
        self.safe_to_key = {k.replace('/', '_'): k for k in raw_manifest.keys()}
        # Keep only sequences that have annotations (training use-case)
        self.manifest = {}
        for safe, key in self.safe_to_key.items():
            info = raw_manifest[key]
            if info.get('annotation_path') is not None:
                self.manifest[safe] = info

        self.sequence_list = list(self.manifest.keys())  # safe names
        self.class_list = ['target']
        self.class_to_id = {'target': 0}

    # ------------------------------------------------------------------

    def get_name(self):
        return 'mtc_aic4'

    def has_class_info(self):
        return False

    def get_num_sequences(self):
        return len(self.sequence_list)

    def __len__(self):
        return len(self.sequence_list)

    # ------------------------------------------------------------------

    def _read_bb_anno(self, safe_name):
        seq_info = self.manifest[safe_name]
        anno_path = os.path.join(self.root, seq_info['annotation_path'])
        gt = pd.read_csv(
            anno_path,
            sep=r'[,\s]+',  # This replaces delimiter=',' and handles spaces or commas
            header=None,
            dtype=np.float32,
            na_filter=False,
            engine='python'  # Required when using regex separators
        ).values
        return torch.tensor(gt)

    def get_sequence_info(self, seq_id):
        safe_name = self.sequence_list[seq_id]
        bbox = self._read_bb_anno(safe_name)

        # Non-visible frames are annotated as 0,0,0,0
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.byte()

        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, safe_name, frame_id):
        seq_info = self.manifest[safe_name]
        return os.path.join(
            self.root,
            seq_info['dataset'],
            seq_info['seq_name'],
            'img',
            '{:08d}.jpg'.format(frame_id + 1),
        )

    def _get_frame(self, safe_name, frame_id):
        return self.image_loader(self._get_frame_path(safe_name, frame_id))

    def get_frames(self, seq_id, frame_ids, anno=None):
        safe_name = self.sequence_list[seq_id]
        frame_list = [self._get_frame(safe_name, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {
            key: [value[f_id, ...].clone() for f_id in frame_ids]
            for key, value in anno.items()
        }

        object_meta = OrderedDict({
            'object_class_name': 'target',
            'motion_class': None,
            'major_class': None,
            'root_class': None,
            'motion_adverb': None,
        })
        return frame_list, anno_frames, object_meta
