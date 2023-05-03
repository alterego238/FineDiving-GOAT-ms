import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../"))

import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
import time
import numpy as np
from utils.misc import segment_iou, cal_tiou, seg_pool_1d, seg_pool_3d



def goat(args, data, target, feature_1, feature_2, gcn, attn_encoder, device):
    if args.use_goat:
        if args.use_formation:
            video_1_fea = []
            video_2_fea = []
            video_1_fea_list = [feature_1[:, i:i + 60] for i in range(0, 540, 60)]  # [B,60,1024]
            video_2_fea_list = [feature_2[:, i:i + 60] for i in range(0, 540, 60)]  # [B,60,1024]
            formation_features_1 = data['formation_features']  # B,540,1024
            formation_features_1_list = [formation_features_1[:, i:i + 60] for i in range(0, 540, 60)]  # [B,60,1024]
            formation_features_2 = target['formation_features']  # B,540,1024
            formation_features_2_list = [formation_features_2[:, i:i + 60] for i in range(0, 540, 60)]  # [B,60,1024]

            for i in range(9):
                q1 = formation_features_1_list[i]
                k1 = q1
                feature_1_tmp = attn_encoder(q1, k1, video_1_fea_list[i])  # B,60,1024
                video_1_fea.append(feature_1_tmp.mean(1).unsqueeze(1))  # [B,1,1024]

                q2 = formation_features_2_list[i]
                k2 = q2
                feature_2_tmp = attn_encoder(q2, k2, video_2_fea_list[i])  # B,60,1024
                video_2_fea.append(feature_2_tmp.mean(1).unsqueeze(1))  # [B,1,1024]
            video_1_fea = ops.concat(video_1_fea, axis=1)  # B,9,1024
            video_2_fea = ops.concat(video_2_fea, axis=1)  # B,9,1024
        elif args.use_bp:
            video_1_fea = []
            video_2_fea = []
            video_1_fea_list = [feature_1[:, i:i + 60] for i in range(0, 540, 60)]  # [B,60,1024]
            video_2_fea_list = [feature_2[:, i:i + 60] for i in range(0, 540, 60)]  # [B,60,1024]
            bp_features_1 = data['bp_features']  # B,540,768
            bp_features_1_list = [bp_features_1[:, i:i + 60] for i in range(0, 540, 60)]  # [B,60,768]
            bp_features_2 = target['bp_features']  # B,540,768
            bp_features_2_list = [bp_features_2[:, i:i + 60] for i in range(0, 540, 60)]  # [B,60,768]

            for i in range(9):
                q1 = bp_features_1_list[i]
                k1 = q1
                feature_1_tmp = attn_encoder(q1, k1, video_1_fea_list[i])  # B,60,1024
                video_1_fea.append(feature_1_tmp.mean(1).unsqueeze(1))  # [B,1,1024]

                q2 = bp_features_2_list[i]
                k2 = q2
                feature_2_tmp = attn_encoder(q2, k2, video_2_fea_list[i])  # B,60,1024
                video_2_fea.append(feature_2_tmp.mean(1).unsqueeze(1))  # [B,1,1024]
            video_1_fea = ops.concat(video_1_fea, axis=1)  # B,9,1024
            video_2_fea = ops.concat(video_2_fea, axis=1)  # B,9,1024
        elif args.use_self:
            video_1_fea = []
            video_2_fea = []
            video_1_fea_list = [feature_1[:, i:i + 60] for i in range(0, 540, 60)]  # [B,60,1024]
            video_2_fea_list = [feature_2[:, i:i + 60] for i in range(0, 540, 60)]  # [B,60,1024]

            for i in range(9):
                q1 = video_1_fea_list[i]
                k1 = q1
                feature_1_tmp = attn_encoder(q1, k1, video_1_fea_list[i])  # B,60,1024
                video_1_fea.append(feature_1_tmp.mean(1).unsqueeze(1))  # [B,1,1024]

                q2 = video_2_fea_list[i]
                k2 = q2
                feature_2_tmp = attn_encoder(q2, k2, video_2_fea_list[i])  # B,60,1024
                video_2_fea.append(feature_2_tmp.mean(1).unsqueeze(1))  # [B,1,1024]
            video_1_fea = ops.concat(video_1_fea, axis=1)  # B,9,1024
            video_2_fea = ops.concat(video_2_fea, axis=1)  # B,9,1024
        else:
            if args.use_cnn_features:
                # video1
                video_1_fea = []
                video_1_fea_list = [feature_1[:, i:i + 60] for i in range(0, 540, 60)]  # [B,60,1024]
                boxes_features_1 = data['cnn_features']
                boxes_features_1_list = [boxes_features_1[:, i:i + 60] for i in range(0, 540, 60)]  # [B,60,N,1024]
                boxes_in_1 = data['boxes']  # B,T,N,4
                boxes_in_1_list = [boxes_in_1[:, i:i + 60] for i in range(0, 540, 60)]  # [B,60,N,4]

                # video2
                video_2_fea = []
                video_2_fea_list = [feature_2[:, i:i + 60] for i in range(0, 540, 60)]  # [B,60,1024]
                boxes_features_2 = target['cnn_features']
                boxes_features_2_list = [boxes_features_2[:, i:i + 60] for i in range(0, 540, 60)]  # [B,60,N,1024]
                boxes_in_2 = target['boxes']  # B,T,N,4
                boxes_in_2_list = [boxes_in_2[:, i:i + 60] for i in range(0, 540, 60)]  # [B,60,N,4]

                for i in range(9):
                    q1 = gcn(boxes_features_1_list[i], boxes_in_1_list[i])  # B,60,1024
                    k1 = q1
                    feature_1_tmp = attn_encoder(q1, k1, video_1_fea_list[i])  # B,60,1024
                    video_1_fea.append(feature_1_tmp.mean(1).unsqueeze(1))  # [B,1,1024]

                    q2 = gcn(boxes_features_2_list[i], boxes_in_2_list[i])  # B,60,1024
                    k2 = q2
                    feature_2_tmp = attn_encoder(q2, k2, video_2_fea_list[i])  # B,60,1024
                    video_2_fea.append(feature_2_tmp.mean(1).unsqueeze(1))  # [B,1,1024]
                video_1_fea = ops.concat(video_1_fea, axis=1)  # B,9,1024
                video_2_fea = ops.concat(video_2_fea, axis=1)  # B,9,1024
            else:
                # video1
                video_1_fea = []
                video_1_fea_list = [feature_1[:, i:i + 60] for i in range(0, 540, 60)]  # [B,60,1024]
                images_in_1 = data['video']  # B,T,C,H,W
                images_in_1_list = [images_in_1[:, i:i + 60] for i in range(0, 540, 60)]  # [B,60,C,H,W]
                boxes_in_1 = data['boxes']  # B,T,N,4
                boxes_in_1_list = [boxes_in_1[:, i:i + 60] for i in range(0, 540, 60)]  # [B,60,N,4]
                # video2
                video_2_fea = []
                video_2_fea_list = [feature_2[:, i:i + 60] for i in range(0, 540, 60)]  # [B,60,1024]
                images_in_2 = target['video']  # B,T,C,H,W
                images_in_2_list = [images_in_2[:, i:i + 60] for i in range(0, 540, 60)]  # [B,60,C,H,W]
                boxes_in_2 = target['boxes']  # B,T,N,4
                boxes_in_2_list = [boxes_in_2[:, i:i + 60] for i in range(0, 540, 60)]  # [B,60,N,4]
                for i in range(9):
                    q1 = gcn(images_in_1_list[i], boxes_in_1_list[i])  # B,60,1024
                    k1 = q1
                    feature_1_tmp = attn_encoder(q1, k1, video_1_fea_list[i])  # B,60,1024
                    video_1_fea.append(feature_1_tmp.mean(1).unsqueeze(1))  # [B,1,1024]

                    q2 = gcn(images_in_2_list[i], boxes_in_2_list[i])  # B,60,1024
                    k2 = q2
                    feature_2_tmp = attn_encoder(q2, k2, video_2_fea_list[i])  # B,60,1024
                    video_2_fea.append(feature_2_tmp.mean(1).unsqueeze(1))  # [B,1,1024]
                video_1_fea = ops.concat(video_1_fea, axis=1)  # B,9,1024
                video_2_fea = ops.concat(video_2_fea, axis=1)  # B,9,1024
    else:
        video_1_fea = ops.concat([feature_1[:, i:i + 60].mean(1).unsqueeze(1) for i in range(0, 540, 60)], 1)  # B,9,1024
        video_2_fea = ops.concat([feature_2[:, i:i + 60].mean(1).unsqueeze(1) for i in range(0, 540, 60)], 1)  # B,9,1024

    return video_1_fea, video_2_fea


def network_forward_train(base_model, psnet_model, decoder, regressor_delta, pred_scores,
                          feature_1, label_1_score, feature_2, label_2_score, mse, optimizer, opti_flag,
                          epoch, batch_idx, batch_num, args, label_1_tas, label_2_tas, bce,
                          pred_tious_5, pred_tious_75, feamap_1, feamap_2, data, target, gcn, attn_encoder, device, linear_bp):
    start = time.time()
    #optimizer.zero_grad()

    def forward_fn(pred_scores, feature_1, label_1_score, feature_2, label_2_score, label_1_tas, label_2_tas, 
                   pred_tious_5, pred_tious_75, feamap_1, feamap_2, data, target):
        ############# I3D featrue #############
        N, T, C, T_t, H_t, W_t = (args.bs_train, 9, 1024, 2, 4, 4)
        N = feature_1.shape[0]
        if not args.use_i3d_bb:
            feature_1 = linear_bp(feature_1)  # B,540,1024
            feature_2 = linear_bp(feature_2)  # B,540,1024

        # goat
        video_1_fea, video_2_fea = goat(args, data, target, feature_1, feature_2, gcn, attn_encoder, device)
        video_1_feamap_re = ops.concat([feamap_1[:, i:i + 60].mean(1).unsqueeze(1).mean(-3) for i in range(0, 540, 60)], 1).reshape(-1, 9, 1024)
        video_2_feamap_re = ops.concat([feamap_2[:, i:i + 60].mean(1).unsqueeze(1).mean(-3) for i in range(0, 540, 60)], 1).reshape(-1, 9, 1024)

        ############# Procedure Segmentation #############
        com_feature_12_u = ops.concat((video_1_fea, video_2_fea), 0)  # (2B, 9, 1024)
        com_feamap_12_u = ops.concat((video_1_feamap_re, video_2_feamap_re), 0)  # (32B, 9, 1024)

        u_fea_96, transits_pred = psnet_model(com_feature_12_u) # (2, 96, 64), (2, 96, 2)
        u_feamap_96, transits_pred_map = psnet_model(com_feamap_12_u) # (32, 96, 64), (32, 96, 2)
        u_feamap_96 = u_feamap_96.reshape(2 * N, u_feamap_96.shape[1], u_feamap_96.shape[2], H_t, W_t) # (2, 96, 64, 4, 4)

        label_12_tas = ops.concat((label_1_tas, label_2_tas), 0)
        label_12_pad = ops.zeros(transits_pred.shape, transits_pred.dtype)
        # one-hot
        for bs in range(transits_pred.shape[0]):
            label_12_pad[bs, int(label_12_tas[bs, 0]), 0] = 1
            label_12_pad[bs, int(label_12_tas[bs, -1]), -1] = 1

        loss_tas = bce(transits_pred, label_12_pad)

        num = round(transits_pred.shape[1] / transits_pred.shape[-1])
        transits_st_ed = ops.zeros(label_12_tas.shape, label_12_tas.dtype)
        for bs in range(transits_pred.shape[0]):
            for i in range(transits_pred.shape[-1]):
                transits_st_ed[bs, i] = transits_pred[bs, i * num: (i + 1) * num, i].argmax(0).item() + i * num
        label_1_tas_pred = transits_st_ed[:transits_st_ed.shape[0] // 2]
        label_2_tas_pred = transits_st_ed[transits_st_ed.shape[0] // 2:]

        ############# Procedure-aware Cross-attention #############
        u_fea_96_1 = u_fea_96[:u_fea_96.shape[0] // 2].transpose(0, 2, 1) # (1, 64, 96)
        u_fea_96_2 = u_fea_96[u_fea_96.shape[0] // 2:].transpose(0, 2, 1) # (1, 64, 96)

        u_feamap_96_1 = u_feamap_96[:u_feamap_96.shape[0] // 2].transpose(0, 2, 1, 3, 4) # (1, 64, 96, 4, 4)
        u_feamap_96_2 = u_feamap_96[u_feamap_96.shape[0] // 2:].transpose(0, 2, 1, 3, 4) # (1, 64, 96, 4, 4)

        if epoch / args.max_epoch <= args.prob_tas_threshold:
            video_1_segs = []
            for bs_1 in range(u_fea_96_1.shape[0]):
                video_1_st = int(label_1_tas[bs_1][0].item())
                video_1_ed = int(label_1_tas[bs_1][1].item())
                video_1_segs.append(seg_pool_1d(u_fea_96_1[bs_1].unsqueeze(0), video_1_st, video_1_ed, args.fix_size)) # (1, 64, 15)
            video_1_segs = ops.concat(video_1_segs, 0).transpose(0, 2, 1) # (1, 15, 64)

            video_2_segs = []
            for bs_2 in range(u_fea_96_2.shape[0]):
                video_2_st = int(label_2_tas[bs_2][0].item())
                video_2_ed = int(label_2_tas[bs_2][1].item())
                video_2_segs.append(seg_pool_1d(u_fea_96_2[bs_2].unsqueeze(0), video_2_st, video_2_ed, args.fix_size)) # (1, 64, 15)
            video_2_segs = ops.concat(video_2_segs, 0).transpose(0, 2, 1) # (1, 15, 64)

            video_1_segs_map = []
            for bs_1 in range(u_feamap_96_1.shape[0]):
                video_1_st = int(label_1_tas[bs_1][0].item())
                video_1_ed = int(label_1_tas[bs_1][1].item())
                video_1_segs_map.append(
                    seg_pool_3d(u_feamap_96_1[bs_1].unsqueeze(0), video_1_st, video_1_ed, args.fix_size)) # (1, 64, 15, 4, 4)
            video_1_segs_map = ops.concat(video_1_segs_map, 0) # (1, 64, 15, 4, 4)
            video_1_segs_map = video_1_segs_map.reshape(video_1_segs_map.shape[0], video_1_segs_map.shape[1],
                                                        video_1_segs_map.shape[2], -1).transpose(0, 1, 3, 2) # (1, 64, 16, 15)
            video_1_segs_map = ops.concat([video_1_segs_map[:, :, :, i] for i in range(video_1_segs_map.shape[-1])],
                                        2).transpose(0, 2, 1) # (1, 240, 64)

            video_2_segs_map = []
            for bs_2 in range(u_fea_96_2.shape[0]):
                video_2_st = int(label_2_tas[bs_2][0].item())
                video_2_ed = int(label_2_tas[bs_2][1].item())
                video_2_segs_map.append(
                    seg_pool_3d(u_feamap_96_2[bs_2].unsqueeze(0), video_2_st, video_2_ed, args.fix_size)) # (1, 64, 15, 4, 4)
            video_2_segs_map = ops.concat(video_2_segs_map, 0) # (1, 64, 15, 4, 4)
            video_2_segs_map = video_2_segs_map.reshape(video_2_segs_map.shape[0], video_2_segs_map.shape[1],
                                                        video_2_segs_map.shape[2], -1).transpose(0, 1, 3, 2) # (1, 64, 16, 15)
            video_2_segs_map = ops.concat([video_2_segs_map[:, :, :, i] for i in range(video_2_segs_map.shape[-1])],
                                        2).transpose(0, 2, 1) # (1, 240, 64)
        else:
            video_1_segs = []
            for bs_1 in range(u_fea_96_1.shape[0]):
                video_1_st = int(label_1_tas_pred[bs_1][0].item())
                video_1_ed = int(label_1_tas_pred[bs_1][1].item())
                if video_1_st == 0:
                    video_1_st = 1
                if video_1_ed == 0:
                    video_1_ed = 1
                video_1_segs.append(seg_pool_1d(u_fea_96_1[bs_1].unsqueeze(0), video_1_st, video_1_ed, args.fix_size))
            video_1_segs = ops.concat(video_1_segs, 0).transpose(0, 2, 1)

            video_2_segs = []
            for bs_2 in range(u_fea_96_2.shape[0]):
                video_2_st = int(label_2_tas_pred[bs_2][0].item())
                video_2_ed = int(label_2_tas_pred[bs_2][1].item())
                if video_2_st == 0:
                    video_2_st = 1
                if video_2_ed == 0:
                    video_2_ed = 1
                video_2_segs.append(seg_pool_1d(u_fea_96_2[bs_2].unsqueeze(0), video_2_st, video_2_ed, args.fix_size))
            video_2_segs = ops.concat(video_2_segs, 0).transpose(0, 2, 1)

            video_1_segs_map = []
            for bs_1 in range(u_feamap_96_1.shape[0]):
                video_1_st = int(label_1_tas_pred[bs_1][0].item())
                video_1_ed = int(label_1_tas_pred[bs_1][1].item())
                if video_1_st == 0:
                    video_1_st = 1
                if video_1_ed == 0:
                    video_1_ed = 1
                video_1_segs_map.append(
                    seg_pool_3d(u_feamap_96_1[bs_1].unsqueeze(0), video_1_st, video_1_ed, args.fix_size))
            video_1_segs_map = ops.concat(video_1_segs_map, 0)
            video_1_segs_map = video_1_segs_map.reshape(video_1_segs_map.shape[0], video_1_segs_map.shape[1],
                                                        video_1_segs_map.shape[2], -1).transpose(0, 1, 3, 2)
            video_1_segs_map = ops.concat([video_1_segs_map[:, :, :, i] for i in range(video_1_segs_map.shape[-1])],
                                        2).transpose(0, 2, 1)

            video_2_segs_map = []
            for bs_2 in range(u_fea_96_2.shape[0]):
                video_2_st = int(label_2_tas_pred[bs_2][0].item())
                video_2_ed = int(label_2_tas_pred[bs_2][1].item())
                if video_2_st == 0:
                    video_2_st = 1
                if video_2_ed == 0:
                    video_2_ed = 1
                video_2_segs_map.append(
                    seg_pool_3d(u_feamap_96_2[bs_2].unsqueeze(0), video_2_st, video_2_ed, args.fix_size))
            video_2_segs_map = ops.concat(video_2_segs_map, 0)
            video_2_segs_map = video_2_segs_map.reshape(video_2_segs_map.shape[0], video_2_segs_map.shape[1],
                                                        video_2_segs_map.shape[2], -1).transpose(0, 1, 3, 2)
            video_2_segs_map = ops.concat([video_2_segs_map[:, :, :, i] for i in range(video_2_segs_map.shape[-1])],
                                        2).transpose(0, 2, 1)

        decoder_video_12_map_list = []
        decoder_video_21_map_list = []
        for i in range(args.step_num):
            decoder_video_12_map = decoder(video_1_segs[:, i * args.fix_size:(i + 1) * args.fix_size, :],
                                        video_2_segs_map[:,
                                        i * args.fix_size * H_t * W_t:(i + 1) * args.fix_size * H_t * W_t,
                                        :])  # N,15,256/64
            decoder_video_21_map = decoder(video_2_segs[:, i * args.fix_size:(i + 1) * args.fix_size, :],
                                        video_1_segs_map[:,
                                        i * args.fix_size * H_t * W_t:(i + 1) * args.fix_size * H_t * W_t,
                                        :])  # N,15,256/64
            decoder_video_12_map_list.append(decoder_video_12_map)
            decoder_video_21_map_list.append(decoder_video_21_map)

        decoder_video_12_map = ops.concat(decoder_video_12_map_list, 1)
        decoder_video_21_map = ops.concat(decoder_video_21_map_list, 1)

        ############# Fine-grained Contrastive Regression #############
        decoder_12_21 = ops.concat((decoder_video_12_map, decoder_video_21_map), 0)
        delta = regressor_delta(decoder_12_21)
        delta = delta.mean(1)
        loss_aqa = mse(delta[:delta.shape[0] // 2], (label_1_score - label_2_score)) \
                + mse(delta[delta.shape[0] // 2:], (label_2_score - label_1_score))

        loss = loss_aqa + loss_tas
        return loss, delta, transits_pred, label_12_tas, transits_st_ed
    
    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
    (loss, delta, transits_pred, label_12_tas, transits_st_ed), grads = grad_fn(pred_scores, feature_1, label_1_score, feature_2, label_2_score, label_1_tas, label_2_tas, 
                                                                                pred_tious_5, pred_tious_75, feamap_1, feamap_2, data, target)

    '''loss.backward()
    optimizer.step()'''
    loss = ops.depend(loss, optimizer(grads))

    end = time.time()
    batch_time = end - start

    score = (delta[:delta.shape[0] // 2].detach() + label_2_score)
    pred_scores.extend([i.item() for i in score])

    tIoU_results = []
    for bs in range(transits_pred.shape[0] // 2):
        tIoU_results.append(segment_iou(np.array(label_12_tas.squeeze(-1))[bs],
                                        np.array(transits_st_ed.squeeze(-1))[bs],
                                        args))

    tiou_thresholds = np.array([0.5, 0.75])
    tIoU_correct_per_thr = cal_tiou(tIoU_results, tiou_thresholds)
    Batch_tIoU_5 = tIoU_correct_per_thr[0]
    Batch_tIoU_75 = tIoU_correct_per_thr[1]
    pred_tious_5.extend([Batch_tIoU_5])
    pred_tious_75.extend([Batch_tIoU_75])

    if batch_idx % args.print_freq == 0:
        print('[Training][%d/%d][%d/%d] \t Batch_time: %.2f \t Batch_loss: %.4f \t '
              'lr1 : %0.5f \t lr2 : %0.5f'
              % (epoch, args.max_epoch, batch_idx, batch_num, batch_time, loss.asnumpy().item(),
                 optimizer.get_lr()[0], optimizer.get_lr()[1]))


def network_forward_test(base_model, psnet_model, decoder, regressor_delta, pred_scores,
                         feature_1, feature_2_list, label_2_score_list,
                         args, label_1_tas, label_2_tas_list,
                         pred_tious_test_5, pred_tious_test_75, feamap_1, feamap_2_list, data, target, gcn, attn_encoder, device, linear_bp):
    score = 0
    tIoU_results = []
    if not args.use_i3d_bb:
        feature_1 = linear_bp(feature_1)  # B,540,1024
    for tar, feature_2, feamap_2, label_2_score, label_2_tas in zip(target, feature_2_list, feamap_2_list, label_2_score_list, label_2_tas_list):

        ############# I3D featrue #############
        N, T, C, T_t, H_t, W_t = (args.bs_test, 9, 1024, 2, 4, 4)
        N = feature_1.shape[0]
        if not args.use_i3d_bb:
            feature_2 = linear_bp(feature_2)  # B,540,1024

        # goat
        video_1_fea, video_2_fea = goat(args, data, tar, feature_1, feature_2, gcn, attn_encoder, device)
        video_1_feamap_re = ops.concat([feamap_1[:, i:i + 60].mean(1).unsqueeze(1).mean(-3) for i in range(0, 540, 60)], 1).reshape(-1, 9, 1024)
        video_2_feamap_re = ops.concat([feamap_2[:, i:i + 60].mean(1).unsqueeze(1).mean(-3) for i in range(0, 540, 60)], 1).reshape(-1, 9, 1024)

        ############# Procedure Segmentation #############
        com_feature_12_u = ops.concat((video_1_fea, video_2_fea), 0)
        com_feamap_12_u = ops.concat((video_1_feamap_re, video_2_feamap_re), 0)

        u_fea_96, transits_pred = psnet_model(com_feature_12_u) # (2, 96, 64), (2, 96, 2)
        u_feamap_96, transits_pred_map = psnet_model(com_feamap_12_u) # (32, 96, 64), (32, 96, 2)
        u_feamap_96 = u_feamap_96.reshape(2 * N, u_feamap_96.shape[1], u_feamap_96.shape[2], H_t, W_t) # (2, 96, 64, 4, 4)

        label_12_tas = ops.concat((label_1_tas, label_2_tas), 0)
        num = round(transits_pred.shape[1] / transits_pred.shape[-1])
        transits_st_ed = ops.zeros(label_12_tas.shape, label_12_tas.dtype)
        for bs in range(transits_pred.shape[0]):
            for i in range(transits_pred.shape[-1]):
                transits_st_ed[bs, i] = transits_pred[bs, i * num: (i + 1) * num, i].argmax(0).item() + i * num
        label_1_tas_pred = transits_st_ed[:transits_st_ed.shape[0] // 2]
        label_2_tas_pred = transits_st_ed[transits_st_ed.shape[0] // 2:]

        ############# Procedure-aware Cross-attention #############
        u_fea_96_1 = u_fea_96[:u_fea_96.shape[0] // 2].transpose(0, 2, 1) # (1, 64, 96)
        u_fea_96_2 = u_fea_96[u_fea_96.shape[0] // 2:].transpose(0, 2, 1) # (1, 64, 96)
        u_feamap_96_1 = u_feamap_96[:u_feamap_96.shape[0] // 2].transpose(0, 2, 1, 3, 4) # (1, 64, 96, 4, 4)
        u_feamap_96_2 = u_feamap_96[u_feamap_96.shape[0] // 2:].transpose(0, 2, 1, 3, 4) # (1, 64, 96, 4, 4)

        video_1_segs = []
        for bs_1 in range(u_fea_96_1.shape[0]):
            video_1_st = int(label_1_tas_pred[bs_1][0].item())
            video_1_ed = int(label_1_tas_pred[bs_1][1].item())
            if video_1_st == 0:
                video_1_st = 1
            if video_1_ed == 0:
                video_1_ed = 1
            video_1_segs.append(seg_pool_1d(u_fea_96_1[bs_1].unsqueeze(0), video_1_st, video_1_ed, args.fix_size)) # (1, 64, 15)
        video_1_segs = ops.concat(video_1_segs, 0).transpose(0, 2, 1) # (1, 15, 64)

        video_2_segs = []
        for bs_2 in range(u_fea_96_2.shape[0]):
            video_2_st = int(label_2_tas_pred[bs_2][0].item())
            video_2_ed = int(label_2_tas_pred[bs_2][1].item())
            if video_2_st == 0:
                video_2_st = 1
            if video_2_ed == 0:
                video_2_ed = 1
            video_2_segs.append(seg_pool_1d(u_fea_96_2[bs_2].unsqueeze(0), video_2_st, video_2_ed, args.fix_size)) # (1, 64, 15)
        video_2_segs = ops.concat(video_2_segs, 0).transpose(0, 2, 1) # (1, 15, 64)

        video_1_segs_map = []
        for bs_1 in range(u_feamap_96_1.shape[0]):
            video_1_st = int(label_1_tas_pred[bs_1][0].item())
            video_1_ed = int(label_1_tas_pred[bs_1][1].item())
            if video_1_st == 0:
                video_1_st = 1
            if video_1_ed == 0:
                video_1_ed = 1
            video_1_segs_map.append(
                seg_pool_3d(u_feamap_96_1[bs_1].unsqueeze(0), video_1_st, video_1_ed, args.fix_size)) # (1, 64, 15, 4, 4)
        video_1_segs_map = ops.concat(video_1_segs_map, 0) # (1, 64, 15, 4, 4)
        video_1_segs_map = video_1_segs_map.reshape(video_1_segs_map.shape[0], video_1_segs_map.shape[1],
                                                    video_1_segs_map.shape[2], -1).transpose(0, 1, 3, 2) # (1, 64, 16, 15)
        video_1_segs_map = ops.concat([video_1_segs_map[:, :, :, i] for i in range(video_1_segs_map.shape[-1])],
                                     2).transpose(0, 2, 1) # (1, 240, 64)

        video_2_segs_map = []
        for bs_2 in range(u_fea_96_2.shape[0]):
            video_2_st = int(label_2_tas_pred[bs_2][0].item())
            video_2_ed = int(label_2_tas_pred[bs_2][1].item())
            if video_2_st == 0:
                video_2_st = 1
            if video_2_ed == 0:
                video_2_ed = 1
            video_2_segs_map.append(
                seg_pool_3d(u_feamap_96_2[bs_2].unsqueeze(0), video_2_st, video_2_ed, args.fix_size)) # (1, 64, 15, 4, 4)
        video_2_segs_map = ops.concat(video_2_segs_map, 0) # (1, 64, 15, 4, 4)
        video_2_segs_map = video_2_segs_map.reshape(video_2_segs_map.shape[0], video_2_segs_map.shape[1],
                                                    video_2_segs_map.shape[2], -1).transpose(0, 1, 3, 2) # (1, 64, 16, 15)
        video_2_segs_map = ops.concat([video_2_segs_map[:, :, :, i] for i in range(video_2_segs_map.shape[-1])],
                                     2).transpose(0, 2, 1) # (1, 240, 64)

        decoder_video_12_map_list = []
        decoder_video_21_map_list = []
        for i in range(args.step_num):
            decoder_video_12_map = decoder(video_1_segs[:, i * args.fix_size:(i + 1) * args.fix_size, :],
                                           video_2_segs_map[:,
                                           i * args.fix_size * H_t * W_t:(i + 1) * args.fix_size * H_t * W_t,
                                           :])
            decoder_video_21_map = decoder(video_2_segs[:, i * args.fix_size:(i + 1) * args.fix_size, :],
                                           video_1_segs_map[:,
                                           i * args.fix_size * H_t * W_t:(i + 1) * args.fix_size * H_t * W_t,
                                           :])
            decoder_video_12_map_list.append(decoder_video_12_map)
            decoder_video_21_map_list.append(decoder_video_21_map)

        decoder_video_12_map = ops.concat(decoder_video_12_map_list, 1)
        decoder_video_21_map = ops.concat(decoder_video_21_map_list, 1)

        ############# Fine-grained Contrastive Regression #############
        decoder_12_21 = ops.concat((decoder_video_12_map, decoder_video_21_map), 0)
        delta = regressor_delta(decoder_12_21)
        delta = delta.mean(1)
        score += (delta[:delta.shape[0] // 2] + label_2_score)

        for bs in range(transits_pred.shape[0] // 2):
            tIoU_results.append(segment_iou(np.array(label_12_tas.squeeze(-1))[bs],
                                            np.array(transits_st_ed.squeeze(-1))[bs], args))

    pred_scores.extend([i.asnumpy().item() / len(feature_2_list) for i in score])

    tIoU_results_mean = [sum(tIoU_results) / len(tIoU_results)]
    tiou_thresholds = np.array([0.5, 0.75])
    tIoU_correct_per_thr = cal_tiou(tIoU_results_mean, tiou_thresholds)
    pred_tious_test_5.extend([tIoU_correct_per_thr[0]])
    pred_tious_test_75.extend([tIoU_correct_per_thr[1]])


def save_checkpoint(base_model, psnet_model, decoder, regressor_delta, optimizer, epoch,
                    epoch_best_aqa, rho_best, L2_min, RL2_min, prefix, args):
    ms.save_checkpoint(append_dict={
        # 'base_model': base_model.state_dict(),
        'psnet_model': psnet_model.parameters_dict(),
        'decoder': decoder.parameters_dict(),
        'regressor_delta': regressor_delta.parameters_dict(),
        'optimizer': optimizer.parameters_dict(),
        'epoch': epoch,
        'epoch_best_aqa': epoch_best_aqa,
        'rho_best': rho_best,
        'L2_min': L2_min,
        'RL2_min': RL2_min,
    }, ckpt_file_name=os.path.join(args.experiment_path, prefix + '.pth'))


def save_outputs(pred_scores, true_scores, args):
    save_path_pred = os.path.join(args.experiment_path, 'pred.npy')
    save_path_true = os.path.join(args.experiment_path, 'true.npy')
    np.save(save_path_pred, pred_scores)
    np.save(save_path_true, true_scores)
