import math
from lib.models.abavitrack import build_abavitrack
from lib.test.tracker.basetracker import BaseTracker
import torch
import torch.nn.functional as F

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond



class AbaViTrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(AbaViTrack, self).__init__(params)
        network = build_abavitrack(params.cfg, training=False)
        print('net')

        # Safely load the checkpoint
        ckpt = torch.load(self.params.checkpoint, map_location='cuda' if torch.cuda.is_available() else 'cpu')

        # Check if it has the ['net'] wrapper, otherwise assume it is a raw state_dict
        if 'net' in ckpt:
            state_dict = ckpt['net']
        else:
            state_dict = ckpt

        # Load the weights safely
        network.load_state_dict(state_dict, strict=False)

        self.cfg = params.cfg

        # --- DOCKER/CPU FRIENDLY DEVICE CHECK ---
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network = network.to(self.device)
        self.network.eval()

        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE

        # --- DOCKER/CPU FRIENDLY HANN WINDOW ---
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).to(self.device)

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}

        # --- ADAPTIVE SYNERGY CONSTANTS ---
        self.target_memory = None
        self.memory_lr = 0.1

        # Safely store the highly-efficient baseline search factor
        self.base_search_factor = params.search_factor
        self.current_search_factor = self.base_search_factor

    def initialize(self, image, info: dict):
        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                                output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = template

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0

        # Reset memory and expansion on a new video sequence
        self.target_memory = None
        self.current_search_factor = self.base_search_factor

        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1

        # 1. USE DYNAMIC EXPANSION VARIABLE (Do not use params.search_factor)
        x_patch_arr, resize_factor, x_amask_arr = sample_target(
            image, self.state, self.current_search_factor, output_sz=self.params.search_size)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            x_dict = search
            # merge the template and the search
            # run the transformer

            # --- DOCKER/CPU FRIENDLY TENSOR CASTING ---
            out_dict = self.network.forward(
                template=self.z_dict1.tensors.to(self.device),
                search=x_dict.tensors.to(self.device)
            )

        # --- PREDICTION PASS ---
        pred_score_map = out_dict['score_map']

        # Apply the Hanning window penalty map
        response = self.output_window * pred_score_map

        # 1. EXTRACT FROM RESPONSE (Safety Check 2: use penalized map, not raw)
        max_confidence = response.max().item()

        # 2. DATA-INDEPENDENT FEATURE EXTRACTION (Safety Check 1: proportional projection)
        opt_feat = out_dict['opt_feat']  # Shape: (1, C, H_feat, W_feat)

        _, _, H_feat, W_feat = opt_feat.shape
        _, _, H_map, W_map = response.shape

        # Find the peak coordinate in the penalized response map
        max_idx = response.view(-1).argmax().item()
        peak_y_map = max_idx // W_map
        peak_x_map = max_idx % W_map

        # Proportionally project the coordinate back to the ViT feature tensor
        peak_y_feat = int(peak_y_map * (H_feat / H_map))
        peak_x_feat = int(peak_x_map * (W_feat / W_map))

        # Extract the target's semantic vector (Safety Check 3: detached below)
        current_target_feat = opt_feat[0, :, peak_y_feat, peak_x_feat].clone().detach()

        # 3. COSINE SIMILARITY VERIFICATION
        if self.target_memory is None:
            self.target_memory = current_target_feat.clone().detach()
            similarity_score = 1.0
        else:
            similarity_score = F.cosine_similarity(
                current_target_feat.unsqueeze(0),
                self.target_memory.unsqueeze(0)
            ).item()

        # Generate visual bbox from the heads
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()
        visual_bbox = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # --- 4. ADAPTIVE SYNERGY LOGIC ---

        # Base thresholds (Safe, conservative defaults)
        base_conf = 0.35
        base_sim = 0.50

        # Dynamic Relaxation
        # 1. If similarity is excellent, we can survive poor visual confidence (Motion Blur)
        dynamic_conf_thresh = base_conf - (max(0, similarity_score - 0.65) * 0.5)

        # 2. If visual confidence is excellent, we can survive poor similarity (Pose Rotation)
        dynamic_sim_thresh = base_sim - (max(0, max_confidence - 0.50) * 0.4)

        # 3. Absolute minimum limits to prevent tracking background noise
        dynamic_conf_thresh = max(0.15, dynamic_conf_thresh)
        dynamic_sim_thresh = max(0.35, dynamic_sim_thresh)

        # Final verification decision
        is_target_confirmed = (max_confidence > dynamic_conf_thresh) and (similarity_score > dynamic_sim_thresh)

        if is_target_confirmed:
            # FOCUS STATE: Target Confirmed
            self.state = visual_bbox

            # Snap the search window back to high-efficiency baseline
            self.current_search_factor = self.base_search_factor

            # Dynamic Memory Learning Rate (NEW)
            # Learn faster if we are confident, learn slower if we are borderline
            current_lr = self.memory_lr if similarity_score > 0.60 else (self.memory_lr / 2.0)

            # Update memory to account for drone rotation/scaling
            self.target_memory = ((1.0 - current_lr) * self.target_memory) + \
                                 (current_lr * current_target_feat.clone().detach())
        else:
            # PANIC STATE: Out-Of-View (OOV) OR Distractor Detected
            # Freeze the box at the last known location.

            # Multiply search factor to cast a massive net in the next frame
            if self.current_search_factor < 16.0:
                self.current_search_factor *= 1.5

        # for debug
        # self.debug = True  # <--- FORCE THE DEBUG BLOCK TO RUN
        self.use_visdom = False
        if self.debug:
            # Create separate folders for bounding boxes and ponder masks
            save_dir_bbox = os.path.join('debug', 'abavit_patch16_224', 'bbox')
            save_dir_ponder = os.path.join('debug', 'abavit_patch16_224', 'ponder')
            os.makedirs(save_dir_bbox, exist_ok=True)
            os.makedirs(save_dir_ponder, exist_ok=True)

            # 1. Draw and Save Bounding Box Image
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # Draw Tracker Prediction (Red)
            cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)

            # Draw Ground Truth (Green) if available
            if info.get('gt_bbox') is not None:
                x2, y2, w2, h2 = info['gt_bbox']
                cv2.rectangle(image_BGR, (int(x2), int(y2)), (int(x2 + w2), int(y2 + h2)), color=(0, 255, 0),
                              thickness=2)

            cv2.imwrite(os.path.join(save_dir_bbox, "%04d.jpg" % self.frame_id), image_BGR)

            # 2. Draw and Save Ponder Mask (Background Ignorance Heatmap)
            # AbaViT tracks token survival depth in the backbone's counter_token tensor
            if hasattr(self.network.backbone, 'counter_token') and self.network.backbone.counter_token is not None:
                import numpy as np

                # The search region is represented by the last 256 tokens in the tensor
                survived_layers = self.network.backbone.counter_token[0, -256:].view(16, 16).cpu().numpy()

                # Normalize the survival depths to 0-255 (Max ViT depth is 12)
                mask_normalized = (survived_layers / 12.0) * 255.0
                mask_normalized = mask_normalized.astype(np.uint8)

                # Resize the 16x16 token grid to match the 256x256 search image
                mask_resized = cv2.resize(mask_normalized, (x_patch_arr.shape[1], x_patch_arr.shape[0]),
                                          interpolation=cv2.INTER_NEAREST)

                # Apply a color map (Red = kept for all 12 layers, Blue = dropped early)
                heatmap = cv2.applyColorMap(mask_resized, cv2.COLORMAP_JET)

                # Blend the heatmap over the original search region image
                search_bgr = cv2.cvtColor(x_patch_arr.astype(np.uint8), cv2.COLOR_RGB2BGR)
                blended = cv2.addWeighted(search_bgr, 0.5, heatmap, 0.5, 0)

                cv2.imwrite(os.path.join(save_dir_ponder, "%04d.jpg" % self.frame_id), blended)

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)  # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return AbaViTrack
