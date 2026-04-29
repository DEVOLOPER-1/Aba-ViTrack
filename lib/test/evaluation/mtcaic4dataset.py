import os
import json
import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text

class MTCAIC4Dataset(BaseDataset):
    def __init__(self, split='public_lb'):
        super().__init__()
        self.base_path = self.env_settings.mtc_aic4_dir
        self.split = split
        
        manifest_path = os.path.join(self.base_path, 'metadata', 'contestant_manifest.json')
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)[self.split]
            
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _get_sequence_list(self):
        return list(self.manifest.keys())

    def _construct_sequence(self, sequence_name):
        seq_info = self.manifest[sequence_name]
        
        # Load the annotation if it exists, otherwise generate dummy zeros
        if seq_info.get('annotation_path') is not None:
            anno_path = os.path.join(self.base_path, seq_info['annotation_path'])
            ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)
        else:
            ground_truth_rect = np.zeros((seq_info['n_frames'], 4)) 

        # --- THE FIX ---
        # If the annotation file only has one line (e.g., initial bounding box only), 
        # numpy loads it as a 1D array. PyTracking expects a 2D array (N, 4).
        if ground_truth_rect.ndim == 1:
            ground_truth_rect = ground_truth_rect.reshape(-1, 4)
        # ---------------

        frames_path = os.path.join(self.base_path, seq_info['dataset'], seq_info['seq_name'], 'img')
        frames_list = [os.path.join(frames_path, f"{(i+1):08d}.jpg") for i in range(seq_info['n_frames'])]

        return Sequence(sequence_name, frames_list, 'mtc_aic4', ground_truth_rect)