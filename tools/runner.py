import numpy as np
import mindspore as ms
import mindspore.nn as nn

from scipy import stats
from tools import builder, helper
from utils import misc
import time
import pickle
#from models.cnn_model import GCNnet_artisticswimming
from models.cnn_simplified import GCNnet_artisticswimming_simplified
from models.group_aware_attention import Encoder_Blocks
from utils.goat_utils import *
from models.linear_for_bp import Linear_For_Backbone
from mindspore.dataset import GeneratorDataset
#from thop import profile


def train_net(args):
    '''if is_main_process():
        print('Trainer start ... ')'''
    print('Trainer start ... ')
    # build dataset
    train_dataset_generator, test_dataset_generator = builder.dataset_builder(args)
    if args.use_multi_gpu:
        raise NotImplementedError()
        train_dataloader = build_dataloader(train_dataset,
                                            batch_size=args.bs_train,
                                            shuffle=True,
                                            num_workers=args.workers,
                                            persistent_workers=True,
                                            seed=set_seed(args.seed))
    else:
        train_dataset = GeneratorDataset(train_dataset_generator, ["data", "target"], num_parallel_workers=args.workers)
        train_dataset = train_dataset.batch(batch_size=args.bs_train)
        train_dataloader = train_dataset.create_tuple_iterator()
        '''train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=args.bs_train,
                                                       shuffle=False,
                                                       num_workers=int(args.workers),
                                                       pin_memory=True)'''

    test_dataset = GeneratorDataset(test_dataset_generator, ["data", "target"], shuffle=False, num_parallel_workers=args.workers)
    test_dataset = test_dataset.batch(batch_size=args.bs_test)
    test_dataloader = test_dataset.create_tuple_iterator()
    '''test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=args.bs_test,
                                                  shuffle=False,
                                                  num_workers=int(args.workers),
                                                  pin_memory=True)'''

    '''# Set data position
    if torch.cuda.is_available():
        device = get_device()
    else:
        device = torch.device('cpu')'''
    device = None

    # build model
    base_model, psnet_model, decoder, regressor_delta = builder.model_builder(args)

    '''input1 = torch.randn(2, 9, 1024)
    input2 = torch.randn(1, 15, 64)
    input3 = torch.randn(1, 15, 64)
    input4 = torch.randn(1, 15, 64)
    flops, params = profile(psnet_model, inputs=(input1, ))
    print(f'[psnet_model]flops: ', flops, 'params: ', params)
    flops, params = profile(decoder, inputs=(input2, input3))
    print(f'[decoder]flops: ', flops, 'params: ', params)
    flops, params = profile(regressor_delta, inputs=(input4, ))
    print(f'[regressor_delta]flops: ', flops, 'params: ', params)'''

    if args.warmup:
        '''num_steps = len(train_dataloader) * args.max_epoch
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)'''
        num_steps = train_dataset.get_dataset_size() * args.max_epoch
        cosine_decay_lr = nn.CosineDecayLR(min = 1e-8, max = args.lr, decay_steps=num_steps)

    # Set models and optimizer(depend on whether to use goat)
    if args.use_goat:
        if args.use_cnn_features:
            gcn = GCNnet_artisticswimming_simplified(args)
        else:
            raise NotImplementedError()
            gcn = GCNnet_artisticswimming(args)
            gcn.loadmodel(args.stage1_model_path)
        attn_encoder = Encoder_Blocks(args.qk_dim, 1024, args.linear_dim, args.num_heads, args.num_layers, args.attn_drop)
        linear_bp = Linear_For_Backbone(args)
        optimizer = nn.Adam([
            {'params': gcn.trainable_params()},
            {'params': attn_encoder.trainable_params()},
            {'params': psnet_model.trainable_params()},
            {'params': decoder.trainable_params()},
            {'params': linear_bp.trainable_params()},
            {'params': regressor_delta.trainable_params()}
        ], learing_rate=cosine_decay_lr if args.warmup else args.lr, weight_decay=args.weight_decay)
        scheduler = None
        if args.use_multi_gpu:
            raise NotImplementedError()
            wrap_model(gcn, distributed=args.distributed)
            wrap_model(attn_encoder, distributed=args.distributed)
            wrap_model(psnet_model, distributed=args.distributed)
            wrap_model(decoder, distributed=args.distributed)
            wrap_model(linear_bp, distributed=args.distributed)
            wrap_model(regressor_delta, distributed=args.distributed)
        else:
            gcn = gcn
            attn_encoder = attn_encoder
            psnet_model = psnet_model
            decoder = decoder
            linear_bp = linear_bp
            regressor_delta = regressor_delta
    else:
        gcn = None
        attn_encoder = None
        linear_bp = Linear_For_Backbone(args)
        optimizer = nn.Adam([
            {'params': psnet_model.trainable_params()},
            {'params': decoder.trainable_params()},
            {'params': linear_bp.trainable_params()},
            {'params': regressor_delta.trainable_params()}
        ], learning_rate=cosine_decay_lr if args.warmup else args.lr, weight_decay=args.weight_decay)
        scheduler = None
        if args.use_multi_gpu:
            raise NotImplementedError()
            wrap_model(psnet_model, distributed=args.distributed)
            wrap_model(decoder, distributed=args.distributed)
            wrap_model(regressor_delta, distributed=args.distributed)
            wrap_model(linear_bp, distributed=args.distributed)
        else:
            psnet_model = psnet_model
            decoder = decoder
            regressor_delta = regressor_delta
            linear_bp = linear_bp

    start_epoch = 0
    global epoch_best_tas, pred_tious_best_5, pred_tious_best_75, epoch_best_aqa, rho_best, L2_min, RL2_min
    epoch_best_tas = 0
    pred_tious_best_5 = 0
    pred_tious_best_75 = 0
    epoch_best_aqa = 0
    rho_best = 0
    L2_min = 1000
    RL2_min = 1000

    # resume ckpts
    if args.resume:
        start_epoch, epoch_best_aqa, rho_best, L2_min, RL2_min = builder.resume_train(base_model, psnet_model, decoder,
                                                                                      regressor_delta, optimizer, args)
        print('resume ckpts @ %d epoch(rho = %.4f, L2 = %.4f , RL2 = %.4f)'
              % (start_epoch - 1, rho_best, L2_min, RL2_min))

    # DP
    # base_model = nn.DataParallel(base_model)
    # psnet_model = nn.DataParallel(psnet_model)
    # decoder = nn.DataParallel(decoder)
    # regressor_delta = nn.DataParallel(regressor_delta)

    # loss
    mse = nn.MSELoss()
    bce = nn.BCELoss()

    # training phase
    for epoch in range(start_epoch, args.max_epoch):
        if args.use_multi_gpu:
            raise NotImplementedError()
            train_dataloader.sampler.set_epoch(epoch)
        pred_tious_5 = []
        pred_tious_75 = []
        true_scores = []
        pred_scores = []

        # base_model.train()
        psnet_model.set_train()
        decoder.set_train()
        regressor_delta.set_train()
        linear_bp.set_train()
        if args.use_goat:
            gcn.set_train()
            attn_encoder.set_train()

        # if args.fix_bn:
        #     base_model.apply(misc.fix_bn)
        for idx, (data, target) in enumerate(train_dataloader):
            # num_iter += 1
            opti_flag = True

            # video_1 is query and video_2 is exemplar
            feature_1 = data['feature'].float()
            feature_2 = target['feature'].float()
            feamap_1 = data['feamap'].float()
            feamap_2 = target['feamap'].float()
            label_1_tas = data['transits'].float() + 1
            label_2_tas = target['transits'].float() + 1
            label_1_score = data['final_score'].float().reshape(-1, 1)
            label_2_score = target['final_score'].float().reshape(-1, 1)

            # forward

            helper.network_forward_train(base_model, psnet_model, decoder, regressor_delta, pred_scores,
                                         feature_1, label_1_score, feature_2, label_2_score, mse, optimizer,
                                         opti_flag, epoch, idx + 1, len(train_dataloader),
                                         args, label_1_tas, label_2_tas, bce,
                                         pred_tious_5, pred_tious_75, feamap_1, feamap_2, data, target, gcn,
                                         attn_encoder, device, linear_bp)
            true_scores.extend(data['final_score'].asnumpy())

        # evaluation results
        pred_scores = np.array(pred_scores)
        true_scores = np.array(true_scores)
        rho, p = stats.spearmanr(pred_scores, true_scores)
        L2 = np.power(pred_scores - true_scores, 2).sum() / true_scores.shape[0]
        RL2 = np.power((pred_scores - true_scores) / (true_scores.max() - true_scores.min()), 2).sum() / \
              true_scores.shape[0]
        pred_tious_mean_5 = sum(pred_tious_5) / len(train_dataset)
        pred_tious_mean_75 = sum(pred_tious_75) / len(train_dataset)

        '''if is_main_process():
            print('[Training] EPOCH: %d, tIoU_5: %.4f, tIoU_75: %.4f'
                  % (epoch, pred_tious_mean_5, pred_tious_mean_75))
            print(
                '[Training] EPOCH: %d, correlation: %.4f, L2: %.4f, RL2: %.4f, lr1: %.4f' % (epoch, rho, L2, RL2,
                                                                                             optimizer.param_groups[
                                                                                                 0]['lr']))
            validate(base_model, psnet_model, decoder, regressor_delta, test_dataloader, epoch, optimizer, args, gcn,
                 attn_encoder, device, linear_bp)

            print('[TEST] EPOCH: %d, best correlation: %.6f, best L2: %.6f, best RL2: %.6f' % (epoch_best_aqa,
                                                                                               rho_best, L2_min, RL2_min))
            print('[TEST] EPOCH: %d, best tIoU_5: %.6f, best tIoU_75: %.6f' % (epoch_best_tas,
                                                                               pred_tious_best_5, pred_tious_best_75))'''
        
        print('[Training] EPOCH: %d, tIoU_5: %.4f, tIoU_75: %.4f'
                % (epoch, pred_tious_mean_5, pred_tious_mean_75))
        print('[Training] EPOCH: %d, correlation: %.4f, L2: %.4f, RL2: %.4f, lr1: %.4f' % (epoch, rho, L2, RL2,
                                                                                            optimizer.get_lr_parameter(psnet_model.trainable_params())))
        validate(base_model, psnet_model, decoder, regressor_delta, test_dataloader, epoch, optimizer, args, gcn,
                attn_encoder, device, linear_bp)

        print('[TEST] EPOCH: %d, best correlation: %.6f, best L2: %.6f, best RL2: %.6f' % (epoch_best_aqa,
                                                                                            rho_best, L2_min, RL2_min))
        print('[TEST] EPOCH: %d, best tIoU_5: %.6f, best tIoU_75: %.6f' % (epoch_best_tas,
                                                                            pred_tious_best_5, pred_tious_best_75))

        '''# scheduler lr
        if scheduler is not None:
            scheduler.step()'''


def validate(base_model, psnet_model, decoder, regressor_delta, test_dataloader, epoch, optimizer, args, gcn,
             attn_encoder, device, linear_bp):
    print("Start validating epoch {}".format(epoch))
    global use_gpu
    global epoch_best_aqa, rho_best, L2_min, RL2_min, epoch_best_tas, pred_tious_best_5, pred_tious_best_75

    true_scores = []
    pred_scores = []
    pred_tious_test_5 = []
    pred_tious_test_75 = []

    # base_model.set_train(False)
    psnet_model.set_train(False)
    decoder.set_train(False)
    regressor_delta.set_train(False)
    linear_bp.set_train(False)
    if args.use_goat:
        gcn.set_train(False)
        attn_encoder.set_train(False)

    batch_num = len(test_dataloader)
    #with torch.no_grad():
    datatime_start = time.time()

    for batch_idx, (data, target) in enumerate(test_dataloader, 0):
        datatime = time.time() - datatime_start
        start = time.time()

        # video_1 = data['video'].float()
        feature_1 = data['feature'].float()
        feamap_1 = data['feamap'].float()
        # video_2_list = [item['video'].float() for item in target]
        feature_2_list = [item['feature'].float() for item in target]
        feamap_2_list = [item['feamap'].float() for item in target]
        label_1_tas = data['transits'].float() + 1
        label_2_tas_list = [item['transits'].float() + 1 for item in target]
        label_2_score_list = [item['final_score'].float().reshape(-1, 1) for item in target]

        helper.network_forward_test(base_model, psnet_model, decoder, regressor_delta, pred_scores,
                                    feature_1, feature_2_list, label_2_score_list,
                                    args, label_1_tas, label_2_tas_list,
                                    pred_tious_test_5, pred_tious_test_75, feamap_1, feamap_2_list, data, target,
                                    gcn, attn_encoder, device, linear_bp)

        batch_time = time.time() - start
        if batch_idx % args.print_freq == 0:
            print('[TEST][%d/%d][%d/%d] \t Batch_time %.6f \t Data_time %.6f'
                    % (epoch, args.max_epoch, batch_idx, batch_num, batch_time, datatime))
        datatime_start = time.time()
        true_scores.extend(data['final_score'].asnumpy())

        # evaluation results
        pred_scores = np.array(pred_scores)
        true_scores = np.array(true_scores)
        rho, p = stats.spearmanr(pred_scores, true_scores)
        L2 = np.power(pred_scores - true_scores, 2).sum() / true_scores.shape[0]
        RL2 = np.power((pred_scores - true_scores) / (true_scores.max() - true_scores.min()), 2).sum() / \
              true_scores.shape[0]
        pred_tious_test_mean_5 = sum(pred_tious_test_5) / (len(test_dataloader) * args.bs_test)
        pred_tious_test_mean_75 = sum(pred_tious_test_75) / (len(test_dataloader) * args.bs_test)

        if pred_tious_test_mean_5 > pred_tious_best_5:
            pred_tious_best_5 = pred_tious_test_mean_5
        if pred_tious_test_mean_75 > pred_tious_best_75:
            pred_tious_best_75 = pred_tious_test_mean_75
            epoch_best_tas = epoch
        print('[TEST] EPOCH: %d, tIoU_5: %.6f, tIoU_75: %.6f' % (epoch, pred_tious_best_5, pred_tious_best_75))

        if L2_min > L2:
            L2_min = L2
        if RL2_min > RL2:
            RL2_min = RL2
        if rho > rho_best:
            rho_best = rho
            epoch_best_aqa = epoch
            print('-----New best found!-----')
            # helper.save_outputs(pred_scores, true_scores, args)
            # helper.save_checkpoint(base_model, psnet_model, decoder, regressor_delta, optimizer, epoch, epoch_best_aqa,
            #                        rho_best, L2_min, RL2_min, 'last', args)
        if epoch == args.max_epoch - 1:
            log_best(rho_best, RL2_min, epoch_best_aqa, args)

        print('[TEST] EPOCH: %d, correlation: %.6f, L2: %.6f, RL2: %.6f' % (epoch, rho, L2, RL2))


def test_net(args):
    raise NotImplementedError()
    print('Tester start ... ')

    train_dataset_generator, test_dataset_generator = builder.dataset_builder(args)
    test_dataset = GeneratorDataset(test_dataset_generator, ["data", "target"], shuffle=False, num_parallel_workers=args.workers)
    test_dataset = test_dataset.batch(batch_size=args.bs_test)
    test_dataloader = test_dataset.create_tuple_iterator()
    '''train_dataset, test_dataset = builder.dataset_builder(args)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs_test,
                                                  shuffle=False, num_workers=int(args.workers),
                                                  pin_memory=True)'''

    '''# Set data position
    if torch.cuda.is_available():
        device = get_device()
    else:
        device = torch.device('cpu')'''
    device = None

    # build model
    base_model, psnet_model, decoder, regressor_delta = builder.model_builder(args)

    # Set models and optimizer(depend on whether to use goat)
    if args.use_goat:
        gcn = GCNnet_artisticswimming(args)
        attn_encoder = Encoder_Blocks(args.qk_dim, 1024, args.num_heads, args.num_layers, args.attn_drop)
        if args.use_multi_gpu:
            wrap_model(gcn, distributed=args.distributed)
            wrap_model(attn_encoder, distributed=args.distributed)
            wrap_model(psnet_model, distributed=args.distributed)
            wrap_model(decoder, distributed=args.distributed)
            wrap_model(regressor_delta, distributed=args.distributed)
        else:
            gcn = gcn
            attn_encoder = attn_encoder
            psnet_model = psnet_model
            decoder = decoder
            regressor_delta = regressor_delta
    else:
        gcn = None
        attn_encoder = None
        if args.use_multi_gpu:
            wrap_model(psnet_model, distributed=args.distributed)
            wrap_model(decoder, distributed=args.distributed)
            wrap_model(regressor_delta, distributed=args.distributed)
        else:
            psnet_model = psnet_model
            decoder = decoder
            regressor_delta = regressor_delta

    # load checkpoints
    builder.load_model(base_model, psnet_model, decoder, regressor_delta, args)

    '''# CUDA
    global use_gpu
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        # base_model = base_model
        psnet_model = psnet_model
        decoder = decoder
        regressor_delta = regressor_delta
        torch.backends.cudnn.benchmark = True'''

    # DP
    # base_model = nn.DataParallel(base_model)
    # psnet_model = nn.DataParallel(psnet_model)
    # decoder = nn.DataParallel(decoder)
    # regressor_delta = nn.DataParallel(regressor_delta)

    test(base_model, psnet_model, decoder, regressor_delta, test_dataloader, args, gcn, attn_encoder, device)


def test(base_model, psnet_model, decoder, regressor_delta, test_dataloader, args, gcn, attn_encoder, device):
    raise NotImplementedError()
    global use_gpu
    global epoch_best_aqa, rho_best, L2_min, RL2_min
    global epoch_best_tas, pred_tious_best_5, pred_tious_best_75

    true_scores = []
    pred_scores = []
    pred_tious_test_5 = []
    pred_tious_test_75 = []

    # base_model.set_train(False)
    psnet_model.set_train(False)
    decoder.set_train(False)
    regressor_delta.set_train(False)
    if args.use_goat:
        gcn.set_train(False)
        attn_encoder.set_train(False)

    batch_num = len(test_dataloader)
    with torch.no_grad():
        datatime_start = time.time()

        for batch_idx, (data, target) in enumerate(test_dataloader, 0):
            datatime = time.time() - datatime_start
            start = time.time()

            # video_1 = data['video'].float()
            feature_1 = data['feature'].float()
            feamap_1 = data['feamap'].float()
            # video_2_list = [item['video'].float() for item in target]
            feature_2_list = [item['feature'].float() for item in target]
            feamap_2_list = [item['feamap'].float() for item in target]
            label_1_tas = data['transits'].float() + 1
            label_2_tas_list = [item['transits'].float() + 1 for item in target]
            label_2_score_list = [item['final_score'].float().reshape(-1, 1) for item in target]

            helper.network_forward_test(base_model, psnet_model, decoder, regressor_delta, pred_scores,
                                        feature_1, feature_2_list, label_2_score_list,
                                        args, label_1_tas, label_2_tas_list,
                                        pred_tious_test_5, pred_tious_test_75, feamap_1, feamap_2_list, data, target,
                                        gcn, attn_encoder, device)

            batch_time = time.time() - start
            if batch_idx % args.print_freq == 0:
                print('[TEST][%d/%d] \t Batch_time %.2f \t Data_time %.2f'
                      % (batch_idx, batch_num, batch_time, datatime))
            datatime_start = time.time()
            true_scores.extend(data['final_score'].numpy())

        # evaluation results
        pred_scores = np.array(pred_scores)
        true_scores = np.array(true_scores)
        rho, p = stats.spearmanr(pred_scores, true_scores)
        L2 = np.power(pred_scores - true_scores, 2).sum() / true_scores.shape[0]
        RL2 = np.power((pred_scores - true_scores) / (true_scores.max() - true_scores.min()), 2).sum() / \
              true_scores.shape[0]
        pred_tious_test_mean_5 = sum(pred_tious_test_5) / (len(test_dataloader) * args.bs_test)
        pred_tious_test_mean_75 = sum(pred_tious_test_75) / (len(test_dataloader) * args.bs_test)

        print('[TEST] tIoU_5: %.6f, tIoU_75: %.6f' % (pred_tious_test_mean_5, pred_tious_test_mean_75))
        print('[TEST] correlation: %.6f, L2: %.6f, RL2: %.6f' % (rho, L2, RL2))
