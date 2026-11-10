"""Read-only source-readout audit; optional target-eval labels are diagnostics only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from scripts.snd_remote import SNDRemoteRuntime, normalize, write_json
from scripts.snd_srmp import source_readout
from scripts.diagnose_crots_remote import EvaluationImages, summarize_matrix
from train_SRMP_remote import load_checkpoint, make_model, sha256_file


def confidence_counts(probability, labels):
    """Per-bin [total, correct] on native valid pixels."""
    confidence, prediction = probability.max(1)
    valid = labels != 255
    bins = (confidence[valid] * 10).long().clamp(max=9)
    total = torch.bincount(bins, minlength=10)
    correct = torch.bincount(bins[prediction[valid] == labels[valid]], minlength=10)
    return torch.stack((total, correct), dim=1).cpu()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--warm', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--split', choices=('target_train', 'target_eval'), default='target_train')
    parser.add_argument('--confidence-threshold', type=float, default=.7)
    args = parser.parse_args()
    if args.limit < 1 or not 0 <= args.confidence_threshold <= 1:
        parser.error('limit must be positive and confidence threshold in [0, 1]')
    torch.set_num_threads(4)
    source, warm = load_checkpoint(args.source), load_checkpoint(args.warm)
    source_sha = sha256_file(args.source)
    if warm.get('source_checkpoint_sha256') != source_sha and sha256_file(args.warm) != source_sha:
        raise ValueError('Source/warm provenance mismatch')
    weight, bias, directions, gram = source_readout(source)
    eigen = torch.linalg.eigvalsh(gram.double())
    if eigen.min() < -1e-6:
        raise ValueError('Source Gram is not positive semidefinite')
    encoder, decoder = make_model(6, architecture=warm['architecture'])
    encoder.load_state_dict(warm['feature_extractor'], strict=True)
    decoder.load_state_dict(warm['classifier'], strict=True)
    encoder.to(args.device).eval(); decoder.to(args.device).eval()
    weight, bias, directions = weight.to(args.device), bias.to(args.device), directions.to(args.device)
    if args.split == 'target_train':
        data = SNDRemoteRuntime(args.data_root, args.output.parent, args.device,
                                architecture='segformer_mit_b2_nvidia_mlp768')
        names_all = sorted(data.images)
        names = [names_all[i * len(names_all) // min(args.limit, len(names_all))]
                 for i in range(min(args.limit, len(names_all)))]
        samples = ((name, normalize(data.image(name)), None) for name in names)
    else:
        data = EvaluationImages(args.data_root, 'target_eval', 6, 0, args.limit)
        names = [data.base.rows[i][0].name for i in data.indices]
        samples = ((name, *data[i]) for i, name in enumerate(names))
    counts_src, counts_main = torch.zeros(6), torch.zeros(6)
    agreement = entropy_src = entropy_main = embedding_norm = pixels = 0.
    feature_variance = near_zero_embeddings = 0.
    matrices = {'source_readout': torch.zeros(6, 6, dtype=torch.int64),
                'adapted_head': torch.zeros(6, 6, dtype=torch.int64)}
    calibration = {key: torch.zeros(10, 2, dtype=torch.int64) for key in matrices}
    joint = torch.zeros(4, dtype=torch.int64)
    selected_gt = torch.zeros(6, dtype=torch.int64)
    total_gt = torch.zeros(6, dtype=torch.int64)
    selected_correct = selected_pixels = 0
    cases = []
    with torch.no_grad():
        for name, image, label in samples:
            image = image.unsqueeze(0).to(args.device)
            latent = decoder.fuse_features(encoder(image)[1])
            feature_variance += float(latent.float().var(dim=(0, 2, 3), unbiased=False).mean())
            src_logits = F.conv2d(latent, weight, bias)
            main_logits = decoder.linear_pred(latent)
            src, main = src_logits.softmax(1), main_logits.softmax(1)
            agreement += float((src.argmax(1) == main.argmax(1)).sum())
            entropy_src += float((-(src * src.clamp_min(1e-8).log()).sum(1)).sum())
            entropy_main += float((-(main * main.clamp_min(1e-8).log()).sum(1)).sum())
            norms = torch.einsum('bchw,cd->bdhw', src, directions).norm(dim=1)
            embedding_norm += float(norms.sum())
            near_zero_embeddings += float((norms < 1e-4).sum())
            counts_src += torch.bincount(src.argmax(1).flatten().cpu(), minlength=6)
            counts_main += torch.bincount(main.argmax(1).flatten().cpu(), minlength=6)
            pixels += src.shape[0] * src.shape[-2] * src.shape[-1]
            if label is not None:
                label = label.unsqueeze(0).to(args.device)
                native = {key: F.interpolate(logits, size=label.shape[-2:], mode='bilinear',
                                             align_corners=False).softmax(1)
                          for key, logits in [('source_readout', src_logits), ('adapted_head', main_logits)]}
                valid = label != 255
                per_image = {}
                for key, probability in native.items():
                    pred = probability.argmax(1)
                    cm = torch.bincount(label[valid] * 6 + pred[valid], minlength=36).reshape(6, 6).cpu()
                    matrices[key] += cm
                    calibration[key] += confidence_counts(probability, label)
                    per_image[key] = summarize_matrix(cm.numpy())
                source_correct = native['source_readout'].argmax(1)[valid] == label[valid]
                main_correct = native['adapted_head'].argmax(1)[valid] == label[valid]
                joint += torch.bincount(source_correct.long() * 2 + main_correct.long(), minlength=4).cpu()
                keep = valid & (native['source_readout'].max(1).values >= args.confidence_threshold)
                total_gt += torch.bincount(label[valid], minlength=6).cpu()
                selected_gt += torch.bincount(label[keep], minlength=6).cpu()
                selected_correct += int((native['source_readout'].argmax(1)[keep] == label[keep]).sum())
                selected_pixels += int(keep.sum())
                cases.append({'image': name, 'source_miou': per_image['source_readout']['miou'],
                              'main_miou': per_image['adapted_head']['miou'],
                              'source_iou': per_image['source_readout']['iou_by_class'],
                              'main_iou': per_image['adapted_head']['iou_by_class']})
    report = {'source_sha256': sha256_file(args.source), 'warm_sha256': sha256_file(args.warm),
              'images': names, 'feature_cells': int(pixels), 'source_main_agreement': agreement/pixels,
              'source_entropy': entropy_src/pixels, 'main_entropy': entropy_main/pixels,
              'source_embedding_norm_mean': embedding_norm/pixels,
              'source_embedding_near_zero_fraction': near_zero_embeddings/pixels,
              'fused_spatial_channel_variance_mean': feature_variance/len(names),
              'source_class_fraction': (counts_src/counts_src.sum()).tolist(),
              'main_class_fraction': (counts_main/counts_main.sum()).tolist(),
              'gram_eigenvalues': eigen.tolist(),
              'gram_condition': float(eigen.max()/eigen.clamp_min(1e-12).min()),
              'gram_off_diagonal_abs_mean': float((gram - torch.eye(6)).abs().sum()/30)}
    report.update(split=args.split, checkpoint_iteration=warm.get('iteration'), diagnostic_only=True,
                  target_labels_used_for_training=False, checkpoint=str(args.warm))
    if args.split == 'target_eval':
        report['metrics'] = {key: summarize_matrix(cm.numpy()) for key, cm in matrices.items()}
        report['confidence_bins'] = {key: [{'low': i/10, 'high': (i+1)/10, 'pixels': int(n),
                                          'accuracy': int(correct)/int(n) if n else None}
                                         for i, (n, correct) in enumerate(counts)]
                                     for key, counts in calibration.items()}
        report['correctness_joint_fraction'] = dict(zip(
            ('both_wrong', 'source_wrong_main_correct', 'source_correct_main_wrong', 'both_correct'),
            (joint / joint.sum().clamp_min(1)).tolist()))
        report['confidence_filter_diagnostic'] = {
            'threshold': args.confidence_threshold, 'coverage': selected_pixels/max(int(total_gt.sum()), 1),
            'selected_accuracy': selected_correct/selected_pixels if selected_pixels else None,
            'coverage_by_gt_class': (selected_gt / total_gt.clamp_min(1)).tolist(),
            'note': 'Native-pixel offline diagnostic; train filtering acts on masked feature cells.'}
        report['image_cases'] = sorted(cases, key=lambda x: x['source_miou'] - x['main_miou'])
    write_json(args.output, report)
    print(json.dumps({key: report[key] for key in ('checkpoint_iteration', 'split', 'source_main_agreement',
                                                  'gram_condition')} | {'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
