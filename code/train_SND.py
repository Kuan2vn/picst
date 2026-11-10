import argparse
import os
import copy
import datetime
import logging
import time
import math
import numpy as np
from collections import OrderedDict

import torch
import random
import pickle
import torch.nn.functional as F
from PIL import Image
from core.datasets.transform import Compose

from core.configs import cfg
from core.configs.underwater import add_underwater_label_args, apply_underwater_label_args
from core.datasets import build_dataset, rand_mixer_v2
from core.models import build_model, build_feature_extractor, build_classifier 
from core.solver import adjust_learning_rate
from core.utils.misc import mkdir, AverageMeter, intersectionAndUnionGPU 
from core.utils.misc import get_color_pallete, strip_prefix_if_present, WeightEMA, denormalizeimage
from core.utils.losses import BinaryCrossEntropy, pseudo_labels_probs, update_running_conf, full2weak

from core.utils.logger import setup_logger
from core.utils.metric_logger import MetricLogger
from core.apis.inference import inference, multi_scale_inference, sample_val_images
from core.apis.inference import evel_stu, run_test, run_candidate
from core.apis.inference import soft_ND_measure, entropy_measure
import pandas as pd
from datasets.generate_city_label_info import gen_lb_info
import higher


def b2_classifier_parameters(classifier, base_lr, readout_lr_mult):
    if readout_lr_mult == 10:
        return classifier.parameters()
    readout = list(classifier.linear_pred.parameters())
    readout_ids = {id(p) for p in readout}
    return [{'params': [p for p in classifier.parameters() if id(p) not in readout_ids],
             'lr': base_lr * 10, 'lr_mult': 10},
            {'params': readout, 'lr': base_lr * readout_lr_mult, 'lr_mult': readout_lr_mult}]


def freeze_b2_source_readout(classifier, source_readout):
    """Reject an adapted warm head; canonicalize only EMA floating-point drift."""
    with torch.no_grad():
        for name, parameter in classifier.linear_pred.named_parameters():
            source = source_readout['linear_pred.' + name].to(parameter)
            if not torch.allclose(parameter, source, rtol=0, atol=1e-6):
                raise ValueError('Frozen-source mode requires a source-frozen warm checkpoint')
            parameter.copy_(source)
            parameter.requires_grad_(False)

      
def train(cfg, local_rank, distributed, runtime=None):
    # Optional dataset/runtime adapter; the SND optimization below is shared.
    from types import SimpleNamespace
    ops = runtime or SimpleNamespace(
        build_feature_extractor=build_feature_extractor, build_classifier=build_classifier,
        build_dataset=build_dataset, run_candidate=run_candidate,
        sample_val_images=sample_val_images, full2weak=full2weak,
    )
    logger = logging.getLogger("DTST.trainer")
    logger.info("Start training")

    feature_extractor = ops.build_feature_extractor(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    feature_extractor.to(device)
    
    classifier = ops.build_classifier(cfg)
    classifier.to(device)

    ## distributed training
    batch_size = cfg.SOLVER.BATCH_SIZE
    if distributed:
        pg1 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        
        batch_size = int(cfg.SOLVER.BATCH_SIZE / torch.distributed.get_world_size())//2
        if not cfg.MODEL.FREEZE_BN:
            feature_extractor = torch.nn.SyncBatchNorm.convert_sync_batchnorm(feature_extractor)
        feature_extractor = torch.nn.parallel.DistributedDataParallel(
            feature_extractor, device_ids=[local_rank], output_device=local_rank, 
            find_unused_parameters=True, process_group=pg1
        )
        pg2 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        classifier = torch.nn.parallel.DistributedDataParallel(
            classifier, device_ids=[local_rank], output_device=local_rank, 
            find_unused_parameters=True, process_group=pg2
        )
        torch.autograd.set_detect_anomaly(True)
        torch.distributed.barrier()
    
    ## optimizer 
    b2 = runtime is not None and getattr(runtime, 'architecture', '') == 'segformer_mit_b2_nvidia_mlp768'
    if b2:
        optimizer_fea = torch.optim.AdamW(feature_extractor.parameters(), lr=cfg.SOLVER.BASE_LR, weight_decay=.01)
        optimizer_cls = torch.optim.AdamW(b2_classifier_parameters(
            classifier, cfg.SOLVER.BASE_LR, getattr(runtime, 'readout_lr_mult', 10)),
            lr=cfg.SOLVER.BASE_LR*10, weight_decay=.01)
    else:
        optimizer_fea = torch.optim.SGD(feature_extractor.parameters(), lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
        optimizer_cls = torch.optim.SGD(classifier.parameters(), lr=cfg.SOLVER.BASE_LR*10, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer_fea.zero_grad()
    optimizer_cls.zero_grad()

    
    output_dir = cfg.OUTPUT_DIR
    local_rank = 0
    start_epoch = 0
    iteration = 0
    
    ####  resume the ckpt 
    if cfg.resume:
        logger.info("Loading checkpoint from {}".format(cfg.resume))
        checkpoint = (runtime.load_initial(cfg.resume) if runtime else
                      torch.load(cfg.resume, map_location=torch.device('cpu')))
        model_weights = checkpoint['feature_extractor'] if distributed else strip_prefix_if_present(checkpoint['feature_extractor'], 'module.')
        feature_extractor.load_state_dict(model_weights)
        classifier_weights = checkpoint['classifier'] if distributed else strip_prefix_if_present(checkpoint['classifier'], 'module.')
        classifier.load_state_dict(classifier_weights)

    resume_state = None
    if b2 and getattr(runtime, 'resume_path', None):
        from train_SRMP_remote import load_checkpoint
        resume_state = load_checkpoint(runtime.resume_path)
        if (resume_state.get('protocol_signature') != runtime.signature or
                resume_state.get('source_checkpoint_sha256') != runtime.source_sha or
                resume_state.get('stage') != runtime.output.name or
                resume_state.get('smoke_only') != runtime.smoke_only or
                not 0 < resume_state.get('iteration', 0) <= cfg.SOLVER.STOP_ITER):
            raise ValueError('SND resume checkpoint provenance mismatch')
        feature_extractor.load_state_dict(resume_state['student_feature_extractor'], strict=True)
        classifier.load_state_dict(resume_state['student_classifier'], strict=True)
        optimizer_fea.load_state_dict(resume_state['optimizer_feature_extractor'])
        optimizer_cls.load_state_dict(resume_state['optimizer_classifier'])
        iteration = resume_state['iteration']
        runtime.last_iteration = iteration
        runtime.teacher_inferences = resume_state['teacher_inferences']
        runtime.bank_refreshes = resume_state['bank_refreshes']
        runtime.meta_images = resume_state['meta_images']
        if getattr(runtime, 'dtst_settings', None):
            runtime.dtst_feedback.load_state_dict(resume_state['dtst_feedback'])
        if getattr(runtime, 'semantic_recovery', None) is not None:
            runtime.semantic_recovery.load_state_dict(resume_state['semantic_recovery'])

    frozen_readout = b2 and getattr(runtime, 'freeze_source_readout', False)
    if frozen_readout:
        freeze_b2_source_readout(classifier, runtime.source_readout)
        logger.info('Source readout fixed: weight/bias frozen; fusion remains trainable; meta readout fixed')
    
    ####  loss define
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255, reduction='none')
    binary_ce = BinaryCrossEntropy(ignore_index=255)
    
    ####  model define
    max_iters = cfg.SOLVER.MAX_ITER
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    feature_extractor.train()
    classifier.train()
    if b2:
        # Keep source decoder BN statistics fixed throughout target adaptation.
        for module in classifier.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()
    classifier_his = copy.deepcopy(classifier).to(device)
    feature_extractor_his = copy.deepcopy(feature_extractor).to(device)
    classifier_his.eval()
    feature_extractor_his.eval()
    if hasattr(getattr(runtime, 'gsd_supervision', None), 'scout'):
        classifier_his.scout_capture_logits = True
    if resume_state is not None:
        classifier_his.load_state_dict(resume_state['classifier'], strict=True)
        feature_extractor_his.load_state_dict(resume_state['feature_extractor'], strict=True)
        if getattr(runtime, 'aux', None) is not None:
            runtime.aux.load_state_dict(resume_state['aux'])
        if getattr(runtime, 'baseline', None) is not None:
            runtime.baseline.load_state_dict(resume_state['baseline'], feature_extractor, classifier)
    if frozen_readout:
        freeze_b2_source_readout(classifier_his, runtime.source_readout)
    if runtime:
        if b2:
            if resume_state is None:
                runtime.checkpoint(0, feature_extractor_his, classifier_his)
            else:
                curve = runtime.output/'evaluation_curve.jsonl'
                if curve.exists():
                    import json
                    runtime.evaluated_iterations = {json.loads(line)['iteration'] for line in curve.read_text().splitlines()}
        else:
            runtime.record_evaluation(0, feature_extractor_his, classifier_his)
    
    
    ###### Mixup and  rsc init 
    if os.path.exists(os.path.join(cfg.OUTPUT_DIR, 'CTR')) and \
        os.path.exists(os.path.join(cfg.OUTPUT_DIR, 'CTR_O')) and \
        os.path.exists(os.path.join(cfg.OUTPUT_DIR, 'CTR.p')) and \
        os.path.exists(os.path.join(cfg.OUTPUT_DIR, 'CTR_O.p')):
        start_from_None = False
        print('+++++++++++++++++++++++  mixup from stable')
    else:
        print('+++++++++++++++++++++++  mixup from None')
        start_from_None = True
        ops.run_candidate(cfg, feature_extractor_his, classifier_his, local_rank, distributed, init_candidate=True, update_meta=True)
        gen_lb_info(cfg, 'CTR')  ## for mixup 
        gen_lb_info(cfg, 'CTR_O')  ## for rsc 
    if resume_state is not None:
        start_from_None = resume_state['initial_bank']
    tgt_train_data = ops.build_dataset(cfg, mode='train', is_source=False, rsc='CTR_O.p')
    print('len(tgt_train_data)', len(tgt_train_data))
    if cfg.TCR.OPEN and runtime is None:
        tgt_train_data.mixer.label_to_file, _ = pickle.load(open(os.path.join(cfg.OUTPUT_DIR, 'CTR.p'), "rb"))

    if distributed:
        tgt_train_sampler = torch.utils.data.distributed.DistributedSampler(tgt_train_data)
    else:
        tgt_train_sampler = None
    tgt_train_loader = torch.utils.data.DataLoader(tgt_train_data, batch_size=batch_size, shuffle=(tgt_train_sampler is None), num_workers=(runtime.workers if runtime else 2), pin_memory=(device.type == "cuda"), sampler=tgt_train_sampler, drop_last=True)

    ###### confident  init 
    #default param in SAC (https://github.com/visinf/da-sac)
    THRESHOLD_BETA = 0.001
    running_conf = torch.zeros(cfg.MODEL.NUM_CLASSES).to(device)
    running_conf.fill_(THRESHOLD_BETA)
    if resume_state is not None:
        running_conf.copy_(resume_state['running_conf'].to(device))

    ###### Dynamic teacher init
    if cfg.DTU.DYNAMIC:
        stu_eval_list = []
        stu_score_buffer = []
        res_dict = {'stu_ori':[], 'stu_now':[], 'update_iter':[]}
        
        
    cls_his_optimizer = WeightEMA(
        [p for p in classifier_his.parameters() if p.requires_grad],
        [p for p in classifier.parameters() if p.requires_grad],
        alpha= cfg.DTU.EMA_WEIGHT,
    )  
    fea_his_optimizer = WeightEMA(
        list(feature_extractor_his.parameters()), 
        list(feature_extractor.parameters()),
        alpha= cfg.DTU.EMA_WEIGHT,
    )      
    
    
    if cfg.METAPL.OPEN:
        cls_meta = ops.build_classifier(cfg).to(device)
        if frozen_readout:
            cls_meta.linear_pred.requires_grad_(False)
        gen_lb_info(cfg, 'meta_val')
        gen_lb_info(cfg, 'meta_val_mixup')

    if resume_state is not None:
        from train_SRMP_remote import restore_rng
        restore_rng(resume_state['rng'])
        del resume_state
        

    start_training_time = time.time()
    end = time.time()
    for rebuild_id in range(255):
        if iteration == cfg.SOLVER.MAX_ITER:
            break
        if iteration == cfg.SOLVER.STOP_ITER:
            break
        for i, batch in enumerate(tgt_train_loader):
            tgt_input, y, names, tgt_trans_param, tgt_img_full, mix_label = batch[:6]
            tgt_unmixed = batch[6] if len(batch) == 7 else None
            if runtime and getattr(runtime, 'geometry_settings', None) is not None:
                runtime.geometry_last_record = {
                    'geometry_'+key+'_by_class': values.sum(0).tolist()
                    for key, values in tgt_trans_param['geometry_stats'].items()}
                runtime.geometry_last_record['geometry_recipient_scale_mean'] = float(tgt_trans_param['recipient_scale'].mean())
            
            data_time = time.time() - end
            
            if b2:
                schedule_step = iteration + 1 + getattr(runtime, 'stage_offset', 0)
                budget = getattr(runtime, 'total_budget', max_iters)
                current_lr = cfg.SOLVER.BASE_LR * (1e-6 + (1-1e-6)*schedule_step/1500) if schedule_step <= 1500 else cfg.SOLVER.BASE_LR * max(0, 1-schedule_step/budget)
            else:
                current_lr = adjust_learning_rate(cfg.SOLVER.LR_METHOD, cfg.SOLVER.BASE_LR, iteration, max_iters, power=cfg.SOLVER.LR_POWER)
            for index in range(len(optimizer_fea.param_groups)):
                optimizer_fea.param_groups[index]['lr'] = current_lr
            for index in range(len(optimizer_cls.param_groups)):
                optimizer_cls.param_groups[index]['lr'] = current_lr * optimizer_cls.param_groups[index].get('lr_mult', 10)

            optimizer_fea.zero_grad()
            optimizer_cls.zero_grad()

            tgt_input = tgt_input.to(device, non_blocking=True)
            tgt_size = tgt_input.shape[-2:]
            tgt_img_full = tgt_img_full.to(device, non_blocking=True)
            mix_label = mix_label.to(device)
            tgt_full_size = tgt_img_full.shape[-2:]
            
            pheno = getattr(runtime, 'phenology', None) if runtime else None
            if pheno is not None and pheno.c1:
                with torch.no_grad():
                    pheno_full = classifier_his(feature_extractor_his(tgt_img_full)[1], tgt_img_full.shape[-2:])
                    pheno_full = pheno.adjust(pheno_full, tgt_img_full, feature_extractor_his, classifier_his, iteration)
                    pheno_aligned = ops.full2weak(pheno_full, tgt_trans_param)
                tgt_input = pheno.counterfactual(tgt_input, pheno_aligned, mix_label,
                                                 tgt_trans_param['valid'].to(device).bool())
            ### stu forward
            baseline = getattr(runtime, 'baseline', None) if runtime else None
            if baseline is None:  # opt-in re-implemented baselines run their own student forward in loss()
                tgt_features = feature_extractor(tgt_input)[1]
                tgt_pred = classifier(tgt_features)
                tgt_pred = F.interpolate(tgt_pred, size=tgt_size, mode='bilinear', align_corners=not b2)

            ######### dy update
            if cfg.DTU.DYNAMIC and getattr(runtime, 'dtst_settings', None):
                runtime.dtst_feedback.observe(feature_extractor, classifier, tgt_img_full)
            elif cfg.DTU.DYNAMIC:
                with torch.no_grad():
                    tgt_pred_full = classifier(feature_extractor(tgt_img_full.clone().detach())[1])
                    output = F.softmax(tgt_pred_full.clone().detach(), dim=1).detach()
                    if cfg.DTU.PROXY_METRIC == 'ENT': 
                        entropy_val = entropy_measure(output)
                        stu_score_buffer.append(entropy_val)
                        stu_eval_list.append([tgt_img_full.clone().detach().cpu()])
                    elif cfg.DTU.PROXY_METRIC == 'Soft_ND':
                        soft_ND_val, soft_ND_state = soft_ND_measure(output, select_point=100)
                        stu_score_buffer.append(soft_ND_val)
                        stu_eval_list.append([tgt_img_full.clone().detach().cpu(), soft_ND_state.cpu()])
                    else:
                        print('no support')
                        return
            ###########
                    
            #### history model 
            with torch.no_grad():   
                
                size = tgt_img_full.shape[-2:]
                train_feas = feature_extractor_his(tgt_img_full)[1]
                recovery_full_features = train_feas if getattr(runtime, 'semantic_recovery', None) is not None else None
                aux_full_teacher_features = (train_feas if getattr(getattr(runtime, 'aux', None),
                                                                   'view_consistency', False) else None)
                tgt_pred_his_full = classifier_his(train_feas, tgt_img_full.shape[-2:])
                if pheno is not None:
                    tgt_pred_his_full = pheno.adjust(tgt_pred_his_full, tgt_img_full, feature_extractor_his,
                                                     classifier_his, iteration)
                tgt_prob_his = F.softmax(ops.full2weak(tgt_pred_his_full, tgt_trans_param), dim=1)
                train_feas = ops.align_features(train_feas, tgt_trans_param) if b2 else ops.full2weak(train_feas, tgt_trans_param, down_ratio=8)
                
                # pos label
                running_conf = update_running_conf(F.softmax(tgt_pred_his_full, dim=1), running_conf, THRESHOLD_BETA)
                #psd_label, _, _ = pseudo_labels_probs(tgt_prob_his, running_conf, THRESHOLD_BETA)
                if cfg.METAPL.OPEN or (b2 and getattr(runtime, 'meta_off', False)):
                    psd_label = tgt_prob_his.max(1)[1].detach()
                else:
                    psd_label, _, _ = pseudo_labels_probs(tgt_prob_his, running_conf, THRESHOLD_BETA)
               
            
            if runtime:
                psd_label = runtime.mask_padding(psd_label, tgt_trans_param)
            gsd_supervision = getattr(runtime, 'gsd_supervision', None)
            if gsd_supervision is not None:
                extra = {'native_full_logits': tgt_pred_his_full} if hasattr(gsd_supervision, 'scout') else {}
                gsd_supervision.prepare(names, tgt_trans_param, tgt_prob_his,
                                        feature_extractor_his, classifier_his, **extra)

            if cfg.METAPL.OPEN:
                # and (iteration+1) > cfg.METAPL.VAL_UPDATE 
                # meta-pseudo labeling
                cls_meta.load_state_dict(classifier_his.state_dict())
                cls_meta.eval() if b2 else cls_meta.train()
                inner_optimiser = torch.optim.SGD((p for p in cls_meta.parameters() if p.requires_grad), lr=0.1)
                para_w = torch.ones_like(psd_label).float().to(device)
                para_w = torch.nn.Parameter(para_w, requires_grad=True)
                optimiser_para_w = torch.optim.Adam([para_w], lr=9.0e-1) 

                
                psd_label = psd_label * (mix_label==255) + mix_label * ((mix_label!=255))
                psd_label = psd_label.long()
                
                ## start inner-loop for calculate w
                optimize_cnt = 1
                for optimize_id in range(optimize_cnt):
                    inner_optimiser.zero_grad()
                    optimiser_para_w.zero_grad()
                    with higher.innerloop_ctx(cls_meta, inner_optimiser, copy_initial_weights=False) as (fcls, diffopt):
                        ####### pesudo optimize
                        virtual_pred = fcls(tuple(x.detach() for x in train_feas) if b2 else train_feas.detach(), psd_label.shape[-2:])
                        s_ce_loss = criterion(virtual_pred, psd_label.long()) * para_w 
                        s_ce_loss = s_ce_loss.mean()
                        diffopt.step(s_ce_loss)
                        virtual_psd = virtual_pred.max(1)[1]
                        #uni_psd = torch.unique(virtual_psd)
                        
                        ##### updating the w by unbaised data and meta cls
                        val_cnt = 1
                        val_loss = 0

                        #### mixup resampling 
                        for cnt in range(val_cnt):
                            val_input, val_label = ops.sample_val_images(cfg, psd_label, meta_val='meta_val')
                            val_input = val_input.to(device, non_blocking=True)
                            val_label = val_label.to(device, non_blocking=True).long()              
                            
                            v_size = val_label.shape[-2:]
                            with torch.no_grad():
                                val_feas = feature_extractor(val_input)[1]
                            val_pred = fcls(tuple(x.detach() for x in val_feas) if b2 else val_feas.detach(), v_size)
                            val_loss += criterion(val_pred, val_label).mean()
                        (val_loss/val_cnt).backward()
                        if b2 and (para_w.grad is None or not torch.isfinite(para_w.grad).all() or not para_w.grad.abs().any()):
                            raise FloatingPointError('SND meta pixel weights have zero or non-finite gradients')
                            
                    optimiser_para_w.step()            
                ## end inner-loop for calculate w
                uc_map_eln = para_w.clone().detach()
                if cfg.TCR.OPEN:
                    uc_map_eln[mix_label!=255] = 1
            else:
                uc_map_eln = torch.ones_like(psd_label).float()

            psd_label = psd_label * (mix_label==255) + mix_label * ((mix_label!=255))
            if hasattr(gsd_supervision, 'labels_for_loss'):
                psd_label = gsd_supervision.labels_for_loss(psd_label, mix_label)
            recovery = getattr(runtime, 'semantic_recovery', None)
            if recovery is not None:
                recovery.prepare(tgt_img_full, recovery_full_features, tgt_pred_his_full,
                                 names, tgt_trans_param, iteration+1)
                psd_label = recovery.apply(psd_label, mix_label, tgt_prob_his,
                                          tgt_trans_param['valid'].to(device))
            if baseline is not None:
                st_loss, pesudo_p_loss, pesudo_n_loss = baseline.loss(
                    iteration=iteration, encoder=feature_extractor, classifier=classifier, student_input=tgt_input,
                    unmixed=tgt_unmixed.to(device, non_blocking=True) if tgt_unmixed is not None else None,
                    full=tgt_img_full, params=tgt_trans_param, teacher_prob=tgt_prob_his,
                    teacher_full_logits=tgt_pred_his_full, pseudo=psd_label, mix_label=mix_label,
                    valid=tgt_trans_param['valid'].to(device), criterion=criterion,
                    negative_weight=runtime.dtst_settings['negative_weight'])
                runtime.dtst_last_record = {'loss_positive': float(pesudo_p_loss.detach()),
                                            'loss_negative': float(pesudo_n_loss.detach()),
                                            'loss': float(st_loss.detach())}
            else:
                st_loss = criterion(tgt_pred, psd_label.long())
                pesudo_p_loss = (st_loss * (0.5 + 0.5*(uc_map_eln-1).exp()) ).mean()
            if baseline is None and gsd_supervision is not None:
                pesudo_p_loss = gsd_supervision.loss(
                    st_loss, 0.5 + 0.5*(uc_map_eln-1).exp(), psd_label, mix_label)
            if baseline is None and recovery is not None:
                pesudo_p_loss = recovery.positive_loss(st_loss, 0.5 + 0.5*(uc_map_eln-1).exp())
            #pesudo_n_loss = binary_ce(tgt_pred.view(m_batchsize*C, 1, height, width), t_neg_label) * 0.5
            if baseline is None:
                st_loss = pesudo_p_loss
            if baseline is None and getattr(runtime, 'dtst_settings', None):
                from scripts.dtst_remote import negative_loss
                pesudo_n_loss = negative_loss(tgt_pred, tgt_prob_his, tgt_trans_param['valid'].to(device),
                                             runtime.dtst_settings['negative_weight'],
                                             protected_labels=recovery.protected if recovery is not None else
                                             (pheno.protected_labels(mix_label) if pheno is not None else None))
                st_loss = pesudo_p_loss + pesudo_n_loss
                runtime.dtst_last_record = {'loss_positive': float(pesudo_p_loss.detach()),
                                            'loss_negative': float(pesudo_n_loss.detach()),
                                            'loss': float(st_loss.detach())}
            if runtime and not torch.isfinite(st_loss):
                raise FloatingPointError("Non-finite SND loss")
            g2c_aux = getattr(runtime, 'g2c_aux', None)
            st_loss.backward(retain_graph=(g2c_aux is not None and iteration+1 > g2c_aux.prepare_updates))
            if runtime and getattr(runtime, 'context_settings', None) is not None:
                if tgt_unmixed is None:
                    raise RuntimeError('Masked context needs the unmixed recipient')
                from scripts.dtst_geometry import masked_context_loss
                context_loss, runtime.context_last_record = masked_context_loss(
                    feature_extractor, classifier, tgt_unmixed.to(device, non_blocking=True),
                    tgt_prob_his, tgt_trans_param['valid'].to(device), runtime.context_settings)
                if not torch.isfinite(context_loss):
                    raise FloatingPointError('Non-finite masked-context loss')
                context_loss.backward()
            if g2c_aux is not None:
                g2c_aux.backward_g2c(iteration+1, current_lr, feature_extractor,
                    feature_extractor_his, classifier_his, tgt_features,
                    tgt_img_full, tgt_trans_param, tgt_prob_his.argmax(1), mix_label)
            elif getattr(runtime, 'aux', None) is not None:
                if tgt_unmixed is None:
                    raise RuntimeError('Auxiliary view missing from target batch')
                runtime.aux.backward(iteration + 1 + getattr(runtime, 'aux_iteration_offset', 0), current_lr, st_loss, feature_extractor,
                                     classifier, feature_extractor_his, classifier_his,
                                     tgt_img_full, tgt_unmixed.to(device, non_blocking=True), tgt_trans_param,
                                     full_teacher_features=aux_full_teacher_features)
            
            ### update current model
            optimizer_fea.step()
            optimizer_cls.step()
            if baseline is not None:
                baseline.step(current_lr)
            if getattr(runtime, 'aux', None) is not None:
                runtime.aux.step()
            if g2c_aux is not None:
                g2c_aux.step()
        
            ### update history model
            ### eval student perfromance
            if cfg.DTU.DYNAMIC and getattr(runtime, 'dtst_settings', None):
                event = runtime.dtst_feedback.query(feature_extractor, classifier)
                if event:
                    if event['update']:
                        cls_his_optimizer.step(); fea_his_optimizer.step()
                    with (runtime.output/'teacher_feedback.jsonl').open('a') as handle:
                        import json
                        handle.write(json.dumps(dict(event, iteration=iteration+1))+'\n')
            elif cfg.DTU.DYNAMIC:
                if len(stu_score_buffer) >= cfg.DTU.Query_START and int(len(stu_score_buffer)-cfg.DTU.Query_START) % cfg.DTU.Query_STEP ==0:   
                    all_score = evel_stu(cfg, feature_extractor, classifier, stu_eval_list)
                    compare_res = np.array(all_score) - np.array(stu_score_buffer)
                    if np.mean(compare_res > 0) > 0.5 or len(stu_score_buffer) > cfg.DTU.META_MAX_UPDATE:
                        update_iter = len(stu_score_buffer)

                        cls_his_optimizer.step()
                        fea_his_optimizer.step()
                        if hasattr(runtime, 'record_teacher_update'):
                            runtime.record_teacher_update(iteration+1, len(stu_score_buffer), compare_res)
                        
                        res_dict['stu_ori'].append(np.array(stu_score_buffer).mean())
                        res_dict['stu_now'].append(np.array(all_score).mean())
                        res_dict['update_iter'].append(update_iter)
                        
                        df = pd.DataFrame(res_dict)
                        df.to_csv('dyIter_FN.csv')

                        ## reset
                        stu_eval_list = []
                        stu_score_buffer = []

            else:
                if iteration % cfg.DTU.FIX_ITERATION == 0:
                    cls_his_optimizer.step()
                    fea_his_optimizer.step()
           
            ## update
            if cfg.TCR.OPEN:
                tgt_train_data.mix_p = ((1-running_conf[tgt_train_data.mix_classes]) / ((1-running_conf[tgt_train_data.mix_classes]).sum() )).cpu().numpy()
            
            meters.update(loss_p_loss=pesudo_p_loss.item())
            if getattr(runtime, 'dtst_settings', None):
                meters.update(loss_n_loss=pesudo_n_loss.item())
            
            iteration = iteration + 1
            if hasattr(runtime, 'capture_state'):
                runtime.capture_state(iteration, feature_extractor, classifier,
                    feature_extractor_his, classifier_his, optimizer_fea, optimizer_cls,
                    running_conf, start_from_None, stu_score_buffer, stu_eval_list, res_dict)
            if runtime:
                runtime.record_step(iteration, pesudo_p_loss, uc_map_eln, psd_label, mix_label, running_conf, current_lr)

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            eta_seconds = meters.time.global_avg * (cfg.SOLVER.STOP_ITER - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            print_freq = 1 if cfg.SOLVER.MAX_ITER <= 100 else 20
            if iteration % print_freq == 0 or iteration == max_iters:
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "iter: {iter}",
                            "{meters}",
                                "lr: {lr:.6f}",
                                "max mem: {memory:.0f}",
                            ]
                        ).format(
                            eta=eta_string,
                            iter=iteration,
                            meters=str(meters),
                            lr=optimizer_fea.param_groups[0]["lr"],
                            memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                        )
                    )
                    
            checkpoint_due = (iteration == cfg.SOLVER.MAX_ITER or (iteration+1) % cfg.SOLVER.CHECKPOINT_PERIOD == 0)
            if runtime:
                checkpoint_due = (iteration % cfg.SOLVER.CHECKPOINT_PERIOD == 0 or iteration == cfg.SOLVER.STOP_ITER)
            if checkpoint_due:
                filename = os.path.join(output_dir, "model_iter{:06d}.pth".format(int(iteration)))
                if runtime:
                    runtime.checkpoint(iteration, feature_extractor_his, classifier_his)
                else:
                    torch.save({'iteration': iteration, 'feature_extractor': feature_extractor_his.state_dict(), 'classifier':classifier_his.state_dict()}, filename)
                    run_test(cfg, feature_extractor_his, classifier_his, local_rank, distributed)

            
            ### re-build candidate and dataloader
            if (cfg.TCR.OPEN and (iteration+1) % cfg.TCR.UPDATE_FREQUENCY == 0) or \
                ((cfg.METAPL.OPEN or (b2 and getattr(runtime, 'meta_off', False))) and
                 (iteration+1) % cfg.METAPL.VAL_UPDATE == 0):
                ops.run_candidate(cfg, feature_extractor_his, classifier_his, local_rank, distributed, update_meta=start_from_None)
                gen_lb_info(cfg, 'CTR')  ## for mixup 
                gen_lb_info(cfg, 'CTR_O')  ## for rsc 
                tgt_train_data = ops.build_dataset(cfg, mode='train', is_source=False, rsc='CTR_O.p')
                if distributed:
                    tgt_train_sampler = torch.utils.data.distributed.DistributedSampler(tgt_train_data)
                else:
                    tgt_train_sampler = None
                tgt_train_loader = torch.utils.data.DataLoader(tgt_train_data, batch_size=batch_size, shuffle=(tgt_train_sampler is None), num_workers=(runtime.workers if runtime else 2), pin_memory=(device.type == "cuda"), sampler=tgt_train_sampler, drop_last=True)
                if runtime is None:
                    tgt_train_data.mixer.label_to_file, _ = pickle.load(open(os.path.join(cfg.OUTPUT_DIR, 'CTR.p'), "rb"))
                if cfg.METAPL.OPEN:
                    ## meta_val
                    gen_lb_info(cfg, 'meta_val')
                    ## meta_val_mixup
                    gen_lb_info(cfg, 'meta_val_mixup')
        
                if b2 and checkpoint_due:
                    runtime.save_training_state(iteration, feature_extractor, classifier,
                                                feature_extractor_his, classifier_his,
                                                optimizer_fea, optimizer_cls, running_conf, start_from_None)
                break

            if b2 and checkpoint_due:
                runtime.save_training_state(iteration, feature_extractor, classifier,
                                            feature_extractor_his, classifier_his,
                                            optimizer_fea, optimizer_cls, running_conf, start_from_None)
            
            if iteration == cfg.SOLVER.STOP_ITER:
                break
    
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / max(iteration, 1)
        )
    )

    return feature_extractor_his, classifier_his          

            


def main():
    parser = argparse.ArgumentParser(description="PyTorch Semantic Segmentation Training")
    parser.add_argument("-cfg",
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--temp_train", action="store_true", help="testing with small iterations")
    parser.add_argument("--local_rank", type=int, default=0)
    add_underwater_label_args(parser)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # Disable cudnn.benchmark on Windows to prevent deadlocks with dilated convs, 
    # but keep it True on Linux (HPC) for performance unless doing a tiny temp_train.
    if os.name == 'nt' or (hasattr(args, "temp_train") and args.temp_train):
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        RANK = int(os.environ["RANK"])
        if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
            NGPUS_PER_NODE = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
        else:
            NGPUS_PER_NODE = torch.cuda.device_count()
        assert NGPUS_PER_NODE > 0, "CUDA is not supported"
        GPU = RANK % NGPUS_PER_NODE
        torch.cuda.set_device(GPU)
        master_address = os.environ['MASTER_ADDR']
        master_port = int(os.environ['MASTER_PORT'])
        WORLD_SIZE = int(os.environ['WORLD_SIZE'])
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='tcp://{}:{}'.format(
                                                 master_address, master_port),
                                             rank=RANK, world_size=WORLD_SIZE)
        NUM_GPUS = WORLD_SIZE
        print(f"RANK and WORLD_SIZE in environ: {RANK}/{WORLD_SIZE}")
        # torch.cuda.set_device(args.local_rank)
        # torch.distributed.init_process_group(
        #     backend="nccl", init_method="env://"
        # )

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    apply_underwater_label_args(cfg, args)
    if args.temp_train:
        cfg.SOLVER.MAX_ITER = 10
        cfg.SOLVER.STOP_ITER = 10
        cfg.SOLVER.CHECKPOINT_PERIOD = 10
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("DTST", output_dir, args.local_rank)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    fea, cls = train(cfg, args.local_rank, args.distributed)

    if not args.skip_test:
        run_test(cfg, fea, cls, args.local_rank, args.distributed)


if __name__ == "__main__":
    main()
