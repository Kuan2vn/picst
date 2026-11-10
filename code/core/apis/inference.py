# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import errno
import os
import numpy as np
from PIL import Image

import torch
from torch import nn
import torch.nn.init as initer

import logging
from core.utils.misc import mkdir, AverageMeter, intersectionAndUnionGPU 
from core.models import build_model, build_feature_extractor, build_classifier 
from core.datasets import build_dataset, rand_mixer_v2
from core.utils.misc import get_color_pallete
import time
import random
from tqdm import tqdm
import torch.nn.functional as F
import shutil
import pickle
import cv2


UNDERWATER_DATASET_TOKENS = ("suim", "uiis", "usis10k")


def _is_underwater_dataset(name):
    return any(token in name for token in UNDERWATER_DATASET_TOKENS)


def _palette_name_for_target(cfg):
    if "coralscapes" in cfg.DATASETS.TARGET_TRAIN:
        return "coralscapes"
    if _is_underwater_dataset(cfg.DATASETS.TARGET_TRAIN):
        return "suim"
    return "city"


def entropy_measure(output):
    out_max_prob = output.max(1)[0]
    uc_map_prob = 1- (-torch.mul(out_max_prob, torch.log2(out_max_prob)) * 2)    
    return uc_map_prob.mean().item()

def entropy(p, prob=True, mean=True):
    if prob:
        p = F.softmax(p, dim=1)
    en = -torch.sum(p * torch.log(p + 1e-5), 1)
    if mean:
        return torch.mean(en)
    else:
        return en
    
    
def soft_ND_measure(output, select_point=100):
    pred1 = output.permute(0, 2, 3, 1)
    pred1 = pred1.reshape(-1, pred1.size(3))
    pred1_rand = torch.randperm(pred1.size(0))
    #select_point = pred1_rand.shape[0]
    select_point = 100
    pred1 = F.normalize(pred1[pred1_rand[:select_point]])
    pred1_en = entropy(torch.matmul(pred1, pred1.t()) * 20)
    soft_ND_state = pred1_rand
    return pred1_en.item(), soft_ND_state
                        



def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def inference(feature_extractor, classifier, image, size, flip=True):
    bs = image.shape[0]
    if flip:
        image = torch.cat([image, torch.flip(image, [3])], 0)
    with torch.no_grad():
        output = classifier(feature_extractor(image)[1])
    output = F.interpolate(output, size=size, mode='bilinear', align_corners=True)
    output = F.softmax(output, dim=1)
    if flip:
        output = (output[:bs] + output[bs:].flip(2)) / 2
    else:
        output = output
    return output
    
def multi_scale_inference(feature_extractor, classifier, image, tsize, scales=[0.7,1.0,1.3], flip=True):
    feature_extractor.eval()
    classifier.eval()
    output = None
    size = image.shape[-2:]
    for s in scales:
        x = F.interpolate(image, size=(int(size[0]*s), int(size[1]*s)))
        pred = inference(feature_extractor, classifier, x, tsize, flip=False)
        if output is None:
            output = pred
        else:
            output = output + pred
        if flip:
            x_flip = torch.flip(x, [3])
            pred = inference(feature_extractor, classifier, x_flip, tsize, flip=False)
            output = output + pred.flip(3)
    if flip:
        return output/len(scales)/2
    return output/len(scales)
    
    
def sample_val_images(cfg, meta_train_psd, meta_val='meta_val'):
    ## meta_train_psd: b, w, h
    ## return batch val images
    if 'cityscapes' in cfg.DATASETS.TARGET_TRAIN:
        image_root = './datasets/cityscapes/' if os.path.exists('./datasets/cityscapes/') else 'data/cityscape/'
        if cfg.MODEL.NUM_CLASSES == 19:
            img_size = (1536, 768)
        if cfg.MODEL.NUM_CLASSES == 16:
            img_size = (1280, 640)
        meta_mixer = rand_mixer_v2(cfg, image_root)
    elif 'bdd' in cfg.DATASETS.TARGET_TRAIN:
        image_root = "datasets/BDD/"
        img_size = (1280, 640)
        meta_mixer = rand_mixer_v2(cfg, image_root)
    elif _is_underwater_dataset(cfg.DATASETS.TARGET_TRAIN):
        from core.datasets.dataset_path_catalog import DatasetCatalog
        attrs = DatasetCatalog.DATASETS[cfg.DATASETS.TARGET_TRAIN]
        image_root = os.path.join(DatasetCatalog.DATASET_DIR, attrs["data_dir"])
        img_size = tuple(cfg.INPUT.INPUT_SIZE_TEST)
        from core.datasets.suim import rand_mixer_suim
        meta_mixer = rand_mixer_suim(cfg, root=image_root)
    elif 'coralscapes' in cfg.DATASETS.TARGET_TRAIN:
        from core.datasets.dataset_path_catalog import DatasetCatalog
        attrs = DatasetCatalog.DATASETS[cfg.DATASETS.TARGET_TRAIN]
        image_root = os.path.join(DatasetCatalog.DATASET_DIR, attrs["data_dir"])
        img_size = tuple(cfg.INPUT.INPUT_SIZE_TEST)
        from core.datasets.coralscapes import rand_mixer_coralscapes
        meta_mixer = rand_mixer_coralscapes(cfg, root=image_root)

    if cfg.MODEL.NUM_CLASSES == 19:
        mix_classes = [4, 5, 6, 7, 12, 16, 17, 18]
    elif cfg.MODEL.NUM_CLASSES == 7:
        mix_classes = [1, 2, 3, 4, 5, 6]
    elif cfg.MODEL.NUM_CLASSES == 8:
        mix_classes = [1, 2, 3, 4, 5, 6, 7]
    elif cfg.MODEL.NUM_CLASSES == 39:
        mix_classes = list(range(39))
    else:
        mix_classes = [3, 5, 6, 7, 11, 14, 15]    
    uni_meta_label = torch.unique(meta_train_psd)
    uni_cnt = torch.zeros(cfg.MODEL.NUM_CLASSES).float()
    for meta_id in uni_meta_label:
        meta_id = int(meta_id.item()) if hasattr(meta_id, "item") else int(meta_id)
        if meta_id == 255 or meta_id < 0 or meta_id >= cfg.MODEL.NUM_CLASSES:
            continue
        uni_cnt[meta_id] += 1                   
    meta_mixer_p = uni_cnt[mix_classes]
    if meta_mixer_p.sum() <= 0:
        meta_mixer_p = torch.ones_like(meta_mixer_p) / len(mix_classes)
    else:
        meta_mixer_p = meta_mixer_p / meta_mixer_p.sum()

    bs, w, h = meta_train_psd.shape
    meta_val_files = os.path.join(cfg.OUTPUT_DIR, meta_val + '.p')
    meta_label_to_file, meta_file_to_label = pickle.load(open(meta_val_files, "rb"))
    if len(meta_file_to_label) == 0:
        raise RuntimeError("No valid pseudo-label files found in {}".format(meta_val_files))
    meta_val_images = []
    meta_val_labels = []
    for b in range(bs):
        uni_class_idx = torch.unique(meta_train_psd[b])
        uni_class_idx = uni_class_idx.cpu().numpy().astype(np.int64)
        uni_class_idx = uni_class_idx[
            (uni_class_idx >= 0)
            & (uni_class_idx < cfg.MODEL.NUM_CLASSES)
        ]
        uni_class_set = set(uni_class_idx.tolist())
        match_ = []
        name_list = []
        for name, label in meta_file_to_label.items():
            if len(label) > 0:
                match_.append(len(uni_class_set & set(label)))
                name_list.append(name)
        #print('match_', len(match_))
        #print('match top 10', match_[:10])
        #print('name_list_', len(name_list))
        match_ = np.array(match_).astype('float64')
        if len(name_list) == 0:
            name_list = list(meta_file_to_label.keys())
            match_ = np.ones(len(name_list), dtype='float64')
        elif match_.max() <= 0:
            match_ = np.ones_like(match_, dtype='float64')
        else:
            match_ = np.power(match_ / match_.max(), 3)
        match_sum = match_.sum()
        if (not np.isfinite(match_).all()) or (not np.isfinite(match_sum)) or match_sum <= 0:
            choice_p = np.ones(len(name_list), dtype='float64') / len(name_list)
        else:
            choice_p = match_ / match_sum
        selected_name = np.random.choice(name_list, size=1, p=choice_p)[0]

        if 'cityscapes' in cfg.DATASETS.TARGET_TRAIN:
            img_subdir = "leftImg8bit" if os.path.isdir(os.path.join(image_root, "leftImg8bit")) else "data"
            bare = selected_name.split('/')[-1]
            city = bare.split('_')[0]
            img_path = os.path.join(image_root, "%s/train/%s/%s" % (img_subdir, city, bare))
            label_path = os.path.join(cfg.OUTPUT_DIR, "{}/{}".format(meta_val, bare))
        elif 'bdd' in cfg.DATASETS.TARGET_TRAIN:
            img_path = os.path.join(image_root, "bdd-100k/bdd100k/images/10k/train/%s" % selected_name.split('.')[0] + '.jpg')
            label_path = os.path.join(cfg.OUTPUT_DIR, "{}/{}".format(meta_val, selected_name.split('.')[0] + '.png'))
        elif _is_underwater_dataset(cfg.DATASETS.TARGET_TRAIN):
            img_path = os.path.join(image_root, "%s" % selected_name.split('.')[0] + '.jpg')
            label_path = os.path.join(cfg.OUTPUT_DIR, "{}/{}".format(meta_val, selected_name.split('.')[0] + '.png'))
        elif 'coralscapes' in cfg.DATASETS.TARGET_TRAIN:
            img_path = os.path.join(image_root, selected_name.split('.')[0] + '.jpg')
            label_path = os.path.join(cfg.OUTPUT_DIR, "{}/{}".format(meta_val, selected_name.split('.')[0] + '.png'))

        image = Image.open(img_path).convert('RGB')   
        image = image.resize(img_size, Image.BILINEAR)
        label = Image.open(label_path)
        label = label.resize(img_size, Image.NEAREST)
        mix_image, mix_label =  meta_mixer.mix(np.asarray(image, dtype=np.uint8), mix_classes, meta_mixer_p)
        mix_label = np.asarray(label) * (mix_label==255) + mix_label * (mix_label!=255)
        mix_image = np.array(mix_image, dtype=np.float64) / 255
        mix_image -= [0.485, 0.456, 0.406]
        mix_image /= [0.229, 0.224, 0.225]
        mix_image = torch.from_numpy(mix_image.transpose(2, 0, 1)).float().unsqueeze(0)
        mix_label = torch.from_numpy(mix_label).unsqueeze(0)
        meta_val_images.append(mix_image)
        meta_val_labels.append(mix_label)
    meta_val_images = torch.cat(meta_val_images, 0)
    meta_val_labels = torch.cat(meta_val_labels, 0)
    return meta_val_images, meta_val_labels
    
    
def evel_stu(cfg, feature_extractor, classifier, stu_eval_list):
    feature_extractor.eval()
    classifier.eval()
    eval_result = []
    
    if cfg.DTU.PROXY_METRIC == 'GR_SHOW':
        ent_eval_result = []
        gt_eval_result = []
        nd_eval_result = []
        with torch.no_grad():
            for i, (x, permute_index, y) in enumerate(stu_eval_list):
                y=y.cuda()
                output = classifier(feature_extractor(x.cuda())[1], y.shape[-2:])
                output = F.softmax(output, dim=1)
                
                output_class = output.max(1)[1]
                gt_val = ((output_class == y) * (y != 255)).float().mean().item()
                gt_eval_result.append(gt_val)
                
                out_max_prob = output.max(1)[0]
                uc_map_prob = 1- (-torch.mul(out_max_prob, torch.log2(out_max_prob)) * 2)
                ent_eval_result.append(uc_map_prob.mean().item())
    
                pred1 = output.permute(0, 2, 3, 1)
                pred1 = pred1.reshape(-1, pred1.size(3))
                pred1_rand = permute_index
                #select_point = pred1_rand.shape[0]
                select_point = 100
                pred1 = F.normalize(pred1[pred1_rand[:select_point]])
                pred1_en = entropy(torch.matmul(pred1, pred1.t()) * 20)
                nd_eval_result.append(pred1_en.item())
                
                feature_extractor.train()
                classifier.train()
            return gt_eval_result, ent_eval_result, nd_eval_result 
            
            
    with torch.no_grad():
        for i, (x, permute_index) in enumerate(stu_eval_list):
            output = classifier(feature_extractor(x.cuda())[1])
            output = F.softmax(output, dim=1)
            if cfg.DTU.PROXY_METRIC == 'ENT':
                out_max_prob = output.max(1)[0]
                uc_map_prob = 1- (-torch.mul(out_max_prob, torch.log2(out_max_prob)) * 2)
                eval_result.append(uc_map_prob.mean().item())
            elif cfg.DTU.PROXY_METRIC == 'Soft_ND':
                pred1 = output.permute(0, 2, 3, 1)
                pred1 = pred1.reshape(-1, pred1.size(3))
                pred1_rand = permute_index
                #select_point = pred1_rand.shape[0]
                select_point = 100
                pred1 = F.normalize(pred1[pred1_rand[:select_point]])
                pred1_en = entropy(torch.matmul(pred1, pred1.t()) * 20)
                eval_result.append(pred1_en.item())
           
                
    feature_extractor.train()
    classifier.train()
    return eval_result
    

def run_test(cfg, feature_extractor, classifier, local_rank, distributed):
    logger = logging.getLogger("DTST.tester")
    print("local_rank", local_rank)
    if local_rank==0:
        logger.info('>>>>>>>>>>>>>>>> Start Testing >>>>>>>>>>>>>>>>')
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    
    if distributed:
        feature_extractor, classifier = feature_extractor.module, classifier.module
    torch.cuda.empty_cache()  # TODO check if it helps

    test_data = build_dataset(cfg, mode='test', is_source=False)
    if distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
    else:
        test_sampler = None
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True, sampler=test_sampler)
    feature_extractor.eval()
    classifier.eval()
    end = time.time()
    with torch.no_grad():
        for i, (x, y, _,) in enumerate(test_loader):
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True).long()
            
            size = y.shape[-2:]

            _, fea = feature_extractor(x)
            output = classifier(fea, size)
            output = output.max(1)[1]
            intersection, union, target = intersectionAndUnionGPU(output, y, cfg.MODEL.NUM_CLASSES, cfg.INPUT.IGNORE_LABEL)
            if distributed:
                torch.distributed.all_reduce(intersection), torch.distributed.all_reduce(union), torch.distributed.all_reduce(target)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            batch_time.update(time.time() - end)
            end = time.time()
    
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if local_rank==0:
        logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(cfg.MODEL.NUM_CLASSES):
            logger.info('Class_{} {} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, test_data.trainid2name[i], iou_class[i], accuracy_class[i]))
    return mIoU

## meta_val:  初始使用prob初始化，选择TOP50张图像，
## meta_val_mixup: 每个类别选择top-50
## CTR:  candidate for mixup
## CTR_O:  历史信息，用于更新CTR 和 rsc
def run_candidate(cfg, feature_extractor, classifier, local_rank, distributed, init_candidate=False, update_meta=False):
    # 1. stability calculated based on historical tags in CTR_O
    # 2. Select the TOP ranked samples, mask out the tail ranked samples, and save them to CTR; save all historical information to CTR_O
    # 3. Then gen_lb_info generates the CTR.p file

    # 1. 根据 CTR_O 中的历史标签 计算的Rel，
    # 2. 选择TOP ranked样本，mask掉尾部 ranked的样本，并save到 CTR 中； 全部的历史信息则save到 CTR_O
    # 3. 然后 gen_lb_info 生成 CTR.p 文件
    logger = logging.getLogger("tester")
    print("local_rank", local_rank)
    if local_rank==0:
        logger.info('>>>>>>>>>>>>>>>> Run candidate >>>>>>>>>>>>>>>>')
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    
    if distributed:
        feature_extractor, classifier = feature_extractor.module, classifier.module
    torch.cuda.empty_cache()  # TODO check if it helps
    dataset_name = cfg.DATASETS.TEST

    def _safe_threshold(values, ratio, default=0.0):
        if len(values) == 0:
            return default
        values = np.sort(values)
        index = int(np.round(len(values) * ratio))
        index = min(max(index, 0), len(values) - 1)
        return values[index]

    ### TTT for output CTR_O
    ### III for output GT
    if init_candidate:
        test_data = build_dataset(cfg, mode='III', is_source=False) 
    else:
        test_data = build_dataset(cfg, mode='TTT', is_source=False)

    if distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
    else:
        test_sampler = None
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True, sampler=test_sampler)

    feature_extractor.eval()
    classifier.eval()
    end = time.time()
    

    name_list = []
    predicted_label = None
    predicted_prob = None
    if not init_candidate:
        single_iou_list = np.zeros((len(test_loader), cfg.MODEL.NUM_CLASSES))
        
    with torch.no_grad():
        for i, (x, y, name,) in tqdm(enumerate(test_loader)):
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True).long()
            
            size = y.shape[-2:]

            _, fea = feature_extractor(x)
            output = classifier(fea, size)
            probs = F.softmax(output, dim=1)
            pred = output.max(1)[1]
            intersection, union, target = intersectionAndUnionGPU(pred.clone(), y, cfg.MODEL.NUM_CLASSES, cfg.INPUT.IGNORE_LABEL)
            if distributed:
                torch.distributed.all_reduce(intersection), torch.distributed.all_reduce(union), torch.distributed.all_reduce(target)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            
            if init_candidate:
                prob = probs.max(1)[0]
                prob_np = F.interpolate(prob.unsqueeze(0).float(), size=[256, 512]).cpu().numpy().squeeze()
                label_np = F.interpolate(pred.unsqueeze(0).float(), size=[256, 512]).cpu().numpy().squeeze()
                if predicted_prob is None:
                    predicted_prob = np.zeros((len(test_loader), 256, 512))
                    predicted_label = np.zeros((len(test_loader), 256, 512))
                predicted_prob[i] = prob_np
                predicted_label[i] = label_np
            else:
                single_iou = intersection / (union + 1e-8)
                single_iou_list[i] = single_iou
                label_np = F.interpolate(pred.unsqueeze(0).float(), size=[xx//2 for xx in size]).cpu().numpy().squeeze()
                if predicted_label is None:
                    predicted_label = np.zeros((len(test_loader), label_np.shape[0], label_np.shape[1]))
                predicted_label[i] = label_np
            name_list.append(name)
            
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)
            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            batch_time.update(time.time() - end)
            end = time.time()

    if init_candidate:
        save_folder = os.path.join(cfg.OUTPUT_DIR, 'CTR')
        mkdir(os.path.dirname(save_folder))
        mkdir(save_folder)
        save_folder_O = os.path.join(cfg.OUTPUT_DIR, 'CTR_O')
        mkdir(save_folder_O)
        save_folder_meta_val = os.path.join(cfg.OUTPUT_DIR, 'meta_val')
        mkdir(save_folder_meta_val)
        if update_meta:
            shutil.rmtree(save_folder_meta_val)
        mkdir(save_folder_meta_val)
        save_folder_meta_val_mixup = os.path.join(cfg.OUTPUT_DIR, 'meta_val_mixup')
        mkdir(save_folder_meta_val_mixup)
        if update_meta:
            shutil.rmtree(save_folder_meta_val_mixup)
        mkdir(save_folder_meta_val_mixup)
        thres = []
        meta_thres = []
        for i in range(cfg.MODEL.NUM_CLASSES):
            x = predicted_prob[predicted_label==i]
            thres.append(_safe_threshold(x, cfg.TCR.TOPK_CANDIDATE))
            meta_thres.append(_safe_threshold(x, 0.9))
        thres = np.array(thres)
        meta_thres = np.array(meta_thres)
        print('init prob thres is', thres)
        print('init meta_thres thres is', meta_thres)
        palette_name = _palette_name_for_target(cfg)
        select_num_dict = {}
        select_name_list = []
        for index in range(len(name_list)):
            name = name_list[index]
            ### no mask -> CTR
            label = predicted_label[index]
            output = np.asarray(label, dtype=np.uint8)
            output = cv2.resize(output, dsize=None, fx=4.0, fy=4.0, interpolation=cv2.INTER_NEAREST)
            mask = get_color_pallete(output, palette_name)
            mask_filename = name[0] if len(name[0].split("/"))<2 else name[0].split("/")[1]
            mask.save(os.path.join(save_folder_O, mask_filename))
            ### mask -> CTR_O
            prob = predicted_prob[index]
            for i in range(cfg.MODEL.NUM_CLASSES):
                label[(prob<thres[i])*(label==i)] = 255
            output = np.asarray(label, dtype=np.uint8)
            output = cv2.resize(output, dsize=None, fx=4.0, fy=4.0, interpolation=cv2.INTER_NEAREST)
            mask = get_color_pallete(output, palette_name)
            mask_filename = name[0] if len(name[0].split("/"))<2 else name[0].split("/")[1]
            mask.save(os.path.join(save_folder, mask_filename))
            select_num_dict[mask_filename] = np.sum(output!=255)
            select_name_list.append(mask_filename)
            ### mask -> meta_val_mixup
            prob = predicted_prob[index]
            label = predicted_label[index]
            for i in range(cfg.MODEL.NUM_CLASSES):
                label[(prob<meta_thres[i])*(label==i)] = 255
            if np.sum(label==255)/np.sum(label>=0) > 0.98:
                continue
            output = np.asarray(label, dtype=np.uint8)
            output = cv2.resize(output, dsize=None, fx=4.0, fy=4.0, interpolation=cv2.INTER_NEAREST)
            mask = get_color_pallete(output, palette_name)
            mask_filename = name[0] if len(name[0].split("/"))<2 else name[0].split("/")[1]
            if update_meta:
                mask.save(os.path.join(save_folder_meta_val_mixup, mask_filename))            
        ### no mask -> using ranked CTR_O for meta_val
        ranked_name_list = sorted(select_name_list, key=lambda c: select_num_dict[c], reverse=True)
        topK = int(len(ranked_name_list) * cfg.METAPL.TOP_K)
        ranked_name_list = ranked_name_list[:topK]
        if update_meta:
            for index in range(len(ranked_name_list)):
                name = ranked_name_list[index]
                src_ = os.path.join(save_folder, name)
                dst_ = os.path.join(save_folder_meta_val, name)
                shutil.copy(src_, dst_)
        print('init_candidate over !!!')
        
    else:
        thres = []
        meta_thres = []
        save_folder = os.path.join(cfg.OUTPUT_DIR, 'CTR')
        save_folder_O = os.path.join(cfg.OUTPUT_DIR, 'CTR_O')
        save_folder_meta_val = os.path.join(cfg.OUTPUT_DIR, 'meta_val')
        if update_meta:
            shutil.rmtree(save_folder_meta_val)
        mkdir(save_folder_meta_val)
        save_folder_meta_val_mixup = os.path.join(cfg.OUTPUT_DIR, 'meta_val_mixup')
        if update_meta:
            shutil.rmtree(save_folder_meta_val_mixup)
        mkdir(save_folder_meta_val_mixup)
        for i in range(cfg.MODEL.NUM_CLASSES):
            x = single_iou_list[:, i]  
            x = x[x > 0]
            thres.append(_safe_threshold(x, cfg.TCR.TOPK_CANDIDATE))
            meta_thres.append(_safe_threshold(x, 0.9))
        thres = np.array(thres)
        meta_thres = np.array(meta_thres)
        print('ReL thres is', thres)
        palette_name = _palette_name_for_target(cfg)
        select_num_dict = {}
        ranked_name_list = []
        for index in range(len(name_list)):
            name = name_list[index]
            ### no mask -> CTR
            label = predicted_label[index]
            mask_filename = name[0] if len(name[0].split("/"))<2 else name[0].split("/")[1]
            t = np.asarray(label, dtype=np.uint8)
            t = cv2.resize(t, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
            mask = get_color_pallete(t, palette_name)
            mask.save(os.path.join(save_folder_O, mask_filename))
            ReL = single_iou_list[index]
            ### mask -> CTR_O
            for i in range(cfg.MODEL.NUM_CLASSES):  
                if ReL[i]<thres[i]:
                    label[label==i] = 255  
            output = np.asarray(label, dtype=np.uint8)
            output = cv2.resize(output, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
            mask = get_color_pallete(output, palette_name)
            mask_filename = name[0] if len(name[0].split("/"))<2 else name[0].split("/")[1]
            mask.save(os.path.join(save_folder, mask_filename))
            select_num_dict[mask_filename] = np.sum(output!=255)
            ranked_name_list.append(mask_filename)
            ### mask -> meta_val_mixup
            ReL = single_iou_list[index]
            label = predicted_label[index]
            for i in range(cfg.MODEL.NUM_CLASSES):  
                if ReL[i]<meta_thres[i]:
                    label[label==i] = 255  
            if np.sum(label==255)/np.sum(label>=0) > 0.98:
                continue
            output = np.asarray(label, dtype=np.uint8)
            output = cv2.resize(output, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
            mask = get_color_pallete(output, palette_name)
            mask_filename = name[0] if len(name[0].split("/"))<2 else name[0].split("/")[1]
            if update_meta:
                mask.save(os.path.join(save_folder_meta_val_mixup, mask_filename))     
            
        ### no mask -> using ranked CTR_O for meta_val
        ranked_name_list = sorted(ranked_name_list, key=lambda c: select_num_dict[c], reverse=True)
        topK = int(len(ranked_name_list) * cfg.METAPL.TOP_K)
        ranked_name_list = ranked_name_list[:topK]
        if update_meta:
            for index in range(len(ranked_name_list)):
                name = ranked_name_list[index]
                src_ = os.path.join(save_folder, name)
                dst_ = os.path.join(save_folder_meta_val, name)
                shutil.copy(src_, dst_)
            
        print('run_candidate over !!!')
    
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if local_rank==0:
        logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        print('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(cfg.MODEL.NUM_CLASSES):
            logger.info('Class_{} {} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, test_data.trainid2name[i], iou_class[i], accuracy_class[i]))
            print('Class_{} {} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, test_data.trainid2name[i], iou_class[i], accuracy_class[i]))

