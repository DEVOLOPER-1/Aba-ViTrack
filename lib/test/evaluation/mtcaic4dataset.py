import os
import json
import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class MTCAIC4Dataset(BaseDataset):
    """
    MTC-AIC4 competition dataset (evaluation wrapper).

    Supports both 'public_lb' (no annotations, for inference/submission) and
    'train' (with annotations, for local evaluation) splits.

    IMPORTANT: Sequence names in the manifest use the form "datasetN/seq_name"
    (containing a forward-slash).  Keeping that slash inside seq.name would
    cause running.py to silently create nested sub-directories that it never
    explicitly mkdir-s, breaking result saving.  We therefore replace '/' with
    '_' to produce a flat, filesystem-safe name while keeping the original
    manifest key available for frame-path look-ups and submission generation.
    """

    def __init__(self, split='public_lb'):
        super().__init__()
        self.base_path = '/dataset'
        self.split = split

        manifest_path = '/dataset/metadata/contestant_manifest.json'

        if not os.path.exists(manifest_path):
            print(f"CRITICAL: Manifest not found at {manifest_path}")
            # List files to debug live
            print(f"Contents of {self.base_path}: {os.listdir(self.base_path)}")

        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)[self.split]

        # Build a bidirectional mapping between safe names (used as seq.name
        # throughout PyTracking) and original manifest keys (needed for
        # frame paths and submission ID generation).
        #   safe_name  : 'dataset2_basketball_player1'  (no slash)
        #   manifest_key: 'dataset2/basketball_player1' (original)
        self.safe_to_key = {
            k.replace('/', '_'): k for k in self.manifest.keys()
        }
        self.sequence_list = list(self.safe_to_key.keys())

    # ------------------------------------------------------------------
    # BaseDataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.sequence_list)

    def get_sequence_list(self):
        return SequenceList(
            [self._construct_sequence(s) for s in self.sequence_list]
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_sequence_list(self):
        return list(self.safe_to_key.keys())

    def _construct_sequence(self, safe_name):
        """
        Args:
            safe_name: filesystem-safe name, e.g. 'dataset2_basketball_player1'
        """
        manifest_key = self.safe_to_key[safe_name]
        seq_info = self.manifest[manifest_key]

        # ---- Ground-truth annotations --------------------------------
        if seq_info.get('annotation_path') is not None:
            anno_path = os.path.join(
                self.base_path, seq_info['annotation_path']
            )
            ground_truth_rect = load_text(
                str(anno_path), delimiter=',', dtype=np.float64
            )
        else:
            # public_lb: no annotation provided → dummy zeros
            ground_truth_rect = np.zeros((seq_info['n_frames'], 4))

        # Guarantee 2-D shape (N, 4) even if annotation file had 1 row
        if ground_truth_rect.ndim == 1:
            ground_truth_rect = ground_truth_rect.reshape(-1, 4)

        # ---- Frame list ----------------------------------------------
        frames_path = os.path.join(
            self.base_path,
            seq_info['dataset'],
            seq_info['seq_name'],
            'img',
        )
        frames_list = [
            os.path.join(frames_path, f'{(i + 1):08d}.jpg')
            for i in range(seq_info['n_frames'])
        ]

        return Sequence(
            safe_name,          # filesystem-safe, slash-free name
            frames_list,
            'mtc_aic4',
            ground_truth_rect,
        )
