#!/usr/bin/env python3
"""Evaluate one SegFormer-B2 checkpoint on the 2016 PD target-eval chips: mIoU6, mIoU5 (clutter excluded), IoU/class."""
import argparse
import json
from pathlib import Path

import torch

from scripts.diagnose_crots_remote import EvaluationImages, evaluate_at_size
from train_SRMP_remote import load_checkpoint, make_model

CLASSES = ('impervious_surface', 'building', 'low_vegetation', 'tree', 'car', 'clutter')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=2)
    a = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state = load_checkpoint(a.checkpoint)
    encoder, classifier = make_model(6, architecture='segformer_mit_b2_nvidia_mlp768')
    encoder.load_state_dict(state['feature_extractor']); classifier.load_state_dict(state['classifier'])
    encoder.to(device).eval(); classifier.to(device).eval()
    metrics = evaluate_at_size(lambda x, size: classifier(encoder(x)[1], size),
                               EvaluationImages(a.data_root, 'target_eval', 6, 0), 6, device, a.workers, False)
    iou = metrics['iou_by_class']
    print(json.dumps({'checkpoint': str(a.checkpoint), 'iteration': state.get('iteration'),
                      'mIoU6': round(100*metrics['miou'], 2), 'mIoU5': round(100*metrics['miou5'], 2),
                      'iou': {c: round(100*iou[c], 2) for c in CLASSES}}, indent=2))


if __name__ == '__main__':
    with torch.no_grad():
        main()
