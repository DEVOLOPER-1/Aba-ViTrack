from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate
import torch.nn.functional as F

class AbaViTrackActor(BaseActor):
    """ Actor for training AbaViTrack models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        if self.net.is_csl:
            out_dict, out_dict2 = self.forward_pass(data)
            loss, status = self.compute_losses(out_dict, data, out_dict2)
        else:
            out_dict = self.forward_pass(data)
            loss, status = self.compute_losses(out_dict, data)
        return loss, status

    def forward_pass(self, data):
        # currently only support 1 template and 1 search region
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1

        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            # template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
            template_list.append(template_img_i)

        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)
        # search_att = data['search_att'][0].view(-1, *data['search_att'].shape[2:])  # (batch, 320, 320)

        box_mask_z = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(self.cfg, template_list[0].shape[0], template_list[0].device,
                                            data['template_anno'][0])

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                                total_epochs=ce_start_epoch + ce_warm_epoch,
                                                ITERS_PER_EPOCH=1,
                                                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])

        if len(template_list) == 1:
            template_list = template_list[0]

        t_mask = data['t_masks']
        s_mask = data['s_masks']
        out_dict = self.net(template=template_list,
                            search=search_img,
                            t_mask=t_mask,
                            s_mask=s_mask)

        if self.net.is_csl:
            with torch.no_grad():
                out_dict_teacher = self.net_teacher(template=template_list,
                                                    search=search_img)

            out_dict2 = self.net2(template=template_list,
                                  search=search_img,
                                  t_mask=t_mask,
                                  s_mask=s_mask)

            feat_teacher = out_dict_teacher['backbone_feat']
            feat_student1 = out_dict['backbone_feat']
            distill_loss1 = torch.nn.functional.mse_loss(feat_teacher, feat_student1)
            out_dict['distill_loss'] = distill_loss1
            feat_student2 = out_dict2['backbone_feat']
            distill_loss2 = torch.nn.functional.mse_loss(feat_teacher, feat_student2)
            out_dict2['distill_loss'] = distill_loss2

            return out_dict, out_dict2
        return out_dict

    def compute_losses(self, pred_dict, gt_dict, pred_dict2=None, return_status=True):
        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)

        kd_weight = 5
        ml_weight = 1
        ponder_token_scale = 0.0001

        rho_token = pred_dict['rho_token']
        rho_token_weight = pred_dict['rho_token_weight']    ### our
        rho_token[:,1:] = rho_token[:,1:]*rho_token_weight   ### our
        ponder_loss_token = torch.mean(torch.sum(rho_token,1)/(torch.sum(rho_token_weight,1)+1))  ### our


        # Distributional prior
        distr_prior_alpha = 0.001
        kl_metric = pred_dict['kl_metric']
        distr_target = pred_dict['distr_target']
        halting_score_layer = pred_dict['halting_score_layer']
        if distr_prior_alpha > 0.:
            # KL loss
            halting_score_distr = torch.stack(halting_score_layer)
            halting_score_distr = halting_score_distr / torch.sum(halting_score_distr)
            halting_score_distr = torch.clamp(halting_score_distr, 0.01, 0.99)
            distr_prior_loss = kl_metric(halting_score_distr.log(), distr_target)

        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss + ponder_token_scale *ponder_loss_token + distr_prior_alpha*distr_prior_loss
        if self.net.is_csl:
            distill_loss = pred_dict['distill_loss']
            ml_loss = self.objective['loss_kl'](F.log_softmax(pred_dict['backbone_feat'].view(32, -1), dim=1),
                                                F.softmax(pred_dict2['backbone_feat'].detach().view(32, -1), dim=1))
            loss += kd_weight * distill_loss + ml_weight * ml_loss

            # Get boxes
            pred_boxes = pred_dict2['pred_boxes']
            if torch.isnan(pred_boxes).any():
                raise ValueError("Network outputs is NAN! Stop Training")
            num_queries = pred_boxes.size(1)
            pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
            gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                               max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
            # compute giou and iou
            try:
                giou_loss2, iou2 = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
            except:
                giou_loss2, iou2 = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
            # compute l1 loss
            l1_loss2 = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
            # compute location loss
            if 'score_map' in pred_dict2:
                location_loss2 = self.objective['focal'](pred_dict2['score_map'], gt_gaussian_maps)
            else:
                location_loss2 = torch.tensor(0.0, device=l1_loss.device)

            # ==========test===
            # now get the token rhos
            ponder_token_scale = 0.0001
            rho_token = pred_dict2['rho_token']
            rho_token_weight = pred_dict2['rho_token_weight']  ### our
            rho_token[:, 1:] = rho_token[:, 1:] * rho_token_weight  ### our
            ponder_loss_token2 = torch.mean(torch.sum(rho_token, 1) / (torch.sum(rho_token_weight, 1) + 1))  ### our

            # Distributional prior
            distr_prior_alpha = 0.001
            kl_metric = pred_dict2['kl_metric']
            distr_target = pred_dict2['distr_target']
            halting_score_layer = pred_dict2['halting_score_layer']
            if distr_prior_alpha > 0.:
                # KL loss
                halting_score_distr = torch.stack(halting_score_layer)
                halting_score_distr = halting_score_distr / torch.sum(halting_score_distr)
                halting_score_distr = torch.clamp(halting_score_distr, 0.01, 0.99)
                distr_prior_loss2 = kl_metric(halting_score_distr.log(), distr_target)

            # ====================
            # weighted sum
            loss2 = self.loss_weight['giou'] * giou_loss2 + self.loss_weight['l1'] * l1_loss2 + self.loss_weight[
                'focal'] * location_loss2 + ponder_token_scale * ponder_loss_token2 + distr_prior_alpha * distr_prior_loss2
            distill_loss2 = pred_dict2['distill_loss']
            ml_loss2 = self.objective['loss_kl'](F.log_softmax(pred_dict2['backbone_feat'].view(32, -1), dim=1),
                                                 F.softmax(pred_dict['backbone_feat'].detach().view(32, -1), dim=1))
            loss2 += kd_weight * distill_loss2 + ml_weight * ml_loss2

            if return_status:
                # status for log
                mean_iou = iou.detach().mean()
                mean_iou2 = iou2.detach().mean()
                status = {
                    "Loss/total1": loss.item(),
                    "Loss/kd1": distill_loss.item(),
                    "Loss/ml1": ml_loss.item(),
                    "IoU1": mean_iou.item(),

                    "Loss/total2": loss2.item(),
                    "Loss/kd2": distill_loss2.item(),
                    "Loss/ml2": ml_loss2.item(),
                    "IoU2": mean_iou2.item(),
                }
                return [loss, loss2], status
            else:
                return [loss, loss2]

        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "Loss/ponder_token": ponder_loss_token.item(),
                      "Loss/distr_prior": distr_prior_loss.item(),
                      "IoU": mean_iou.item()}
            return [loss], status
        else:
            return [loss]
