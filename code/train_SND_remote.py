"""Run author SND warm + bi-level adaptation on the fixed VH->PD manifests."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
import zipfile

import numpy as np
import torch

from core.configs import cfg as base_cfg
from train_SND import train
from train_SRMP_remote import load_checkpoint, sha256_file
from train_SRMP_remote import B2_PROTOCOL, make_model
from scripts.snd_remote import SNDRemoteRuntime, write_json
from scripts.diagnose_crots_remote import EvaluationImages, Model, evaluate_at_size

ROOT = Path(__file__).resolve().parent
CODE_FILES = ['train_SND.py', 'train_SND_remote.py', 'scripts/snd_remote.py',
              'core/utils/losses.py', 'core/utils/misc.py', 'core/models/feature_extractor.py',
              'core/models/resnet.py', 'core/models/classifier.py', 'core/datasets/transform.py',
              'datasets/generate_city_label_info.py', 'scripts/diagnose_crots_remote.py']
B2_CODE_FILES = CODE_FILES + ['train_SRMP_remote.py', 'core/models/segformer.py', 'scripts/remote_sensing_transforms.py',
                              'configs/remote_sensing/siamseg_sfuda_b2_v1.json']


def bank_digest(directory):
    digest = hashlib.sha256()
    count = 0
    for path in sorted(item for item in directory.rglob('*') if item.is_file()):
        name = path.relative_to(directory).as_posix().encode()
        digest.update(len(name).to_bytes(4, 'big'))
        digest.update(name)
        digest.update(bytes.fromhex(sha256_file(path)))
        count += 1
    return {'files': count, 'sha256': digest.hexdigest()}


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def configuration(recipe, stage, args, output):
    cfg = base_cfg.clone()
    cfg.defrost()
    cfg.MODEL.NUM_CLASSES = 6
    cfg.MODEL.NAME = ('segformer_mit_b2' if recipe['protocol'] == B2_PROTOCOL else 'deeplab_resnet101')
    cfg.MODEL.FREEZE_BN = True
    cfg.MODEL.DEVICE = args.device
    cfg.DATASETS.TARGET_TRAIN = 'isprs_pd_rgb_train'
    cfg.DATASETS.TEST = 'isprs_pd_rgb_eval'
    cfg.INPUT.TARGET_INPUT_SIZE_TRAIN = (recipe['size'], recipe['size'])
    cfg.INPUT.INPUT_SIZE_TEST = (recipe['size'], recipe['size'])
    cfg.INPUT.INPUT_SCALES_TRAIN = (.75, 1.25)
    cfg.INPUT.HORIZONTAL_FLIP_PROB_TRAIN = .5
    cfg.INPUT.BRIGHTNESS = cfg.INPUT.CONTRAST = cfg.INPUT.SATURATION = .5
    cfg.INPUT.HUE = .2
    cfg.SOLVER.BASE_LR = recipe['base_lr']
    cfg.SOLVER.BATCH_SIZE = recipe['batch_size']
    cfg.SOLVER.MOMENTUM = recipe['momentum']
    cfg.SOLVER.WEIGHT_DECAY = recipe['weight_decay']
    cfg.SOLVER.LR_POWER = recipe['lr_power']
    cfg.SOLVER.MAX_ITER = recipe[stage]['lr_max_iter']
    cfg.SOLVER.STOP_ITER = recipe[stage]['steps']
    if stage == 'warm' and getattr(args, 'warm_stop_updates', None) is not None:
        cfg.SOLVER.STOP_ITER = args.warm_stop_updates
    if stage == 'warm' and getattr(args, 'extend_warm_to', None) is not None:
        cfg.SOLVER.STOP_ITER = args.extend_warm_to
    if stage in ('snd', 'meta-off') and getattr(args, 'gsd_stop_updates', None) is not None:
        cfg.SOLVER.STOP_ITER = args.gsd_stop_updates
    cfg.SOLVER.CHECKPOINT_PERIOD = recipe['checkpoint_period']
    cfg.TCR.OPEN = True
    cfg.TCR.TOPK_CANDIDATE = recipe['candidate_top_k']
    cfg.TCR.UPDATE_FREQUENCY = recipe['candidate_update']
    cfg.METAPL.OPEN = recipe[stage]['meta']
    cfg.METAPL.TOP_K = recipe[stage]['meta_top_k']
    cfg.METAPL.VAL_UPDATE = recipe[stage]['meta_update']
    cfg.DTU.DYNAMIC = False
    cfg.DTU.EMA_WEIGHT = recipe[stage]['ema']
    cfg.DTU.FIX_ITERATION = recipe[stage]['ema_interval']
    if getattr(args, 'dtst_settings', None):
        dtst = args.dtst_settings
        cfg.DTU.DYNAMIC = dtst['dynamic_teacher']
        cfg.DTU.EMA_WEIGHT = dtst['ema']
        cfg.DTU.FIX_ITERATION = dtst['fixed_interval']
        if args.gsd_stop_updates is not None:
            cfg.SOLVER.STOP_ITER = args.gsd_stop_updates
    cfg.OUTPUT_DIR = str(output)
    # Author SND (DZhaoXd/SND run.sh) restarts the SND stage from the SOURCE weights with the warm stage's banks
    # copied in; --snd-init source reproduces that (opt-in; default keeps the historical warm-weight start).
    cfg.resume = str((args.warm_checkpoint if recipe['protocol'] == B2_PROTOCOL and stage != 'warm'
                      and getattr(args, 'snd_init', 'warm') == 'warm'
                      else args.source_checkpoint).resolve())
    cfg.freeze()
    return cfg


def portable(output):
    # Explicit report extensions only: no checkpoints, images or pickle caches.
    destination = output/'results_portable.zip'
    temporary = output/'results_portable.zip.tmp'
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(output.rglob('*')):
            if path.is_file() and path.suffix in ('.json', '.jsonl', '.csv', '.yaml', '.log', '.md'):
                bundle.write(path, path.relative_to(output))
    temporary.replace(destination)


def evaluate_checkpoint(path, args, split):
    state = load_checkpoint(path)
    if state.get('architecture') == 'segformer_mit_b2_nvidia_mlp768':
        encoder, classifier = make_model(6, architecture=state['architecture'])
        encoder.load_state_dict(state['feature_extractor'], strict=True)
        classifier.load_state_dict(state['classifier'], strict=True)
        encoder.to(args.device).eval(); classifier.to(args.device).eval()
        model = lambda image, size: classifier(encoder(image)[1], size)
    else:
        model = Model(6)
        model.encoder.load_state_dict(state['feature_extractor'], strict=True)
        model.classifier.load_state_dict(state['classifier'], strict=True)
        model.to(args.device).eval()
    # Never resize evaluation GT. Normal protocol uses native 512 inputs.
    data = EvaluationImages(args.data_root, split, 6, 0, args.eval_limit)
    metrics = evaluate_at_size(model, data, 6, torch.device(args.device), args.workers, False)
    del model, state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def refresh_b2_result(args, signature):
    path = args.output/'result_portable.json'
    saved = json.loads(path.read_text()) if path.is_file() else {}
    if saved and saved.get('protocol_signature') != signature:
        raise ValueError('Existing portable result has a different protocol signature')
    results = saved.get('results', {})
    if 'source' not in results:
        results['source'] = {'checkpoint_sha256': sha256_file(args.source_checkpoint),
                             'metrics': evaluate_checkpoint(args.source_checkpoint, args, 'target_eval')}
    for stage in ('warm', 'snd', 'meta-off'):
        checkpoint = args.output/stage/'last.pth'
        complete = args.output/stage/'complete.json'
        if not checkpoint.is_file() or not complete.is_file() or stage in results:
            continue
        state = json.loads(complete.read_text())
        digest = sha256_file(checkpoint)
        if state['protocol_signature'] != signature or state['checkpoint_sha256'] != digest:
            raise ValueError(f'{stage} checkpoint provenance mismatch')
        points = [json.loads(line) for line in (args.output/stage/'evaluation_curve.jsonl').read_text().splitlines()]
        final = next((point for point in reversed(points) if point['iteration'] == state['iteration']), None)
        results[stage] = {'checkpoint_sha256': digest,
                          'metrics': final['metrics'] if final else evaluate_checkpoint(checkpoint, args, 'target_eval')}
    write_json(path, {'protocol_signature': signature, 'smoke': args.smoke_only,
                      'task': 'vh2pd', 'split': 'target_eval', 'results': results})


def selected_warm_iteration(args, signature):
    warm = args.warm_checkpoint.parent
    state_path = warm/'complete.json'
    if not state_path.is_file():
        state_path = warm/'stopped_early.json'
    complete = json.loads(state_path.read_text())
    if (complete['protocol_signature'] != signature or
            complete['checkpoint_sha256'] != sha256_file(warm/'last.pth')):
        raise ValueError('Completed warm checkpoint provenance mismatch')
    checkpoint = load_checkpoint(args.warm_checkpoint)
    if (checkpoint.get('stage') != 'warm' or checkpoint.get('protocol_signature') != signature or
            checkpoint.get('source_checkpoint_sha256') != sha256_file(args.source_checkpoint) or
            checkpoint.get('smoke_only') != args.smoke_only):
        raise ValueError('Selected warm checkpoint provenance mismatch')
    iteration = checkpoint.get('iteration')
    if args.warm_checkpoint.name == 'last.pth':
        if iteration != complete['iteration']:
            raise ValueError('Warm final iteration mismatch')
    elif args.warm_checkpoint.name == 'best_target_val.pth':
        peak = json.loads((warm/'observed_peak.json').read_text())
        if peak['iteration'] != iteration or peak['protocol_signature'] != signature:
            raise ValueError('Warm peak checkpoint/curve mismatch')
    else:
        raise ValueError('Use warm/last.pth or warm/best_target_val.pth')
    selection = {'iteration': iteration, 'checkpoint_sha256': sha256_file(args.warm_checkpoint),
                 'checkpoint_name': args.warm_checkpoint.name}
    path = args.output/'warm_selection.json'
    if path.exists() and json.loads(path.read_text()) != selection:
        raise ValueError('SND and meta-off must use the same warm checkpoint')
    write_json(path, selection)
    return iteration


def refresh_aux_result(args, signature):
    if args.stage in ('warm', 'finalize-warm'):
        stage = args.output/'warm'
        curve = stage/'evaluation_curve.jsonl'
        points = [json.loads(line) for line in curve.read_text().splitlines()] if curve.is_file() else []
        write_json(args.output/'result_portable.json', {
            'protocol_signature': signature, 'method': 'snd_srmp_aux', 'aux_start': 'warm',
            'objective': args.objective,
            'source': {'checkpoint_sha256': sha256_file(args.source_checkpoint),
                       'metrics': points[0]['metrics'] if points else None},
            'warm_final': {'checkpoint_sha256': sha256_file(stage/'last.pth'),
                           'iteration': points[-1]['iteration'], 'metrics': points[-1]['metrics']} if points else None,
            'warm_peak': json.loads((stage/'observed_peak.json').read_text()) if points else None,
            'warm_peak_miou5': max(points, key=lambda point: point['metrics']['miou5']) if points else None,
            'warm_complete': (stage/'complete.json').is_file(),
            'warm_stopped_early': (stage/'stopped_early.json').is_file(),
            'warm_actual_updates': points[-1]['iteration'] if points else 0,
            'warm_planned_updates': args.warm_planned_updates,
            'warm_stop': json.loads((stage/'stopped_early.json').read_text()) if (stage/'stopped_early.json').is_file() else None,
            'runtime': json.loads((stage/'complete.json').read_text()) if (stage/'complete.json').is_file() else None})
        return
    handoff = json.loads((args.warm_checkpoint.parent.parent/'result_portable.json').read_text())
    if args.aux_start == 'warm':
        source, warm_final = handoff['source'], handoff['warm_final']
        if not (handoff['warm_complete'] or handoff.get('warm_stopped_early')):
            raise ValueError('Auxiliary warm run must be completed or explicitly finalized early')
    else:
        source, warm_final = handoff['results']['source'], handoff['results']['warm']
    if source['checkpoint_sha256'] != sha256_file(args.source_checkpoint):
        raise ValueError('Auxiliary source differs from completed warm run')
    warm_actual = (handoff['warm_actual_updates'] if 'warm_actual_updates' in handoff else
                   json.loads((args.warm_checkpoint.parent/'complete.json').read_text())['iteration'])
    stage = args.output/'snd'
    curve = stage/'evaluation_curve.jsonl'
    points = [json.loads(line) for line in curve.read_text().splitlines()] if curve.is_file() else []
    peak = max(points, key=lambda point: point['metrics']['miou']) if points else None
    post = max(points[1:], key=lambda point: point['metrics']['miou']) if len(points) > 1 else None
    write_json(args.output/'result_portable.json', {
        'protocol_signature': signature, 'method': 'snd_srmp_aux', 'objective': args.objective,
        'aux_start': args.aux_start, 'warm_protocol_signature': args.warm_signature,
        'warm_stopped_early': handoff.get('warm_stopped_early', False),
        'warm_actual_updates': warm_actual,
        'warm_planned_updates': handoff.get('warm_planned_updates', args.warm_planned_updates),
        'source': source, 'warm_final': warm_final,
        'warm_selection': json.loads((args.output/'warm_selection.json').read_text()),
        'warm_peak': json.loads((args.warm_checkpoint.parent/'observed_peak.json').read_text())['metrics'],
        'warm_peak_miou5': handoff.get('warm_peak_miou5'),
        'continuation_complete': (stage/'complete.json').is_file(),
        'runtime': json.loads((stage/'complete.json').read_text()) if (stage/'complete.json').is_file() else None,
        'continuation_last': points[-1] if points else None,
        'continuation_peak': peak, 'continuation_peak_miou5': max(points, key=lambda point: point['metrics']['miou5']) if points else None,
        'continuation_post_start_peak': post,
        'continuation_points': len(points)})


def finalize_aux_warm(args, recipe, signature, persist=True, reason=None):
    warm = args.output/'warm'
    if (warm/'complete.json').is_file():
        raise ValueError('Warm already completed its planned budget')
    marker = warm/'stopped_early.json'
    if marker.is_file():
        existing = json.loads(marker.read_text())
        if existing['protocol_signature'] != signature:
            raise ValueError('Existing early-stop marker belongs to another protocol')
        return existing
    started = json.loads((warm/'started.json').read_text())
    last = load_checkpoint(warm/'last.pth')
    resume = load_checkpoint(warm/'resume.pth')
    best = load_checkpoint(warm/'best_target_val.pth')
    anchor = load_checkpoint(warm/'best_aux_state.pth')
    peak = json.loads((warm/'observed_peak.json').read_text())
    curve = [json.loads(line) for line in (warm/'evaluation_curve.jsonl').read_text().splitlines()]
    iteration = resume['iteration']
    source_sha = sha256_file(args.source_checkpoint)
    if (started['protocol_signature'] != signature or
            not 0 < iteration == last['iteration'] == curve[-1]['iteration'] < recipe['warm']['steps'] or
            resume['protocol_signature'] != signature or resume['source_checkpoint_sha256'] != source_sha or
            last['protocol_signature'] != signature or last['source_checkpoint_sha256'] != source_sha or
            best['protocol_signature'] != signature or best['source_checkpoint_sha256'] != source_sha or
            best['iteration'] != peak['iteration'] or anchor['iteration'] != peak['iteration'] or
            anchor['protocol_signature'] != signature or anchor['source_checkpoint_sha256'] != source_sha or
            not 0 <= peak['iteration'] <= iteration):
        raise ValueError('Early-stop warm checkpoints, source or peak do not match')
    expected = args.target_limit or sum(1 for _ in (args.data_root/'manifests'/'target_train.csv').open()) - 1
    banks = {name: bank_digest(warm/name) for name in ('CTR', 'CTR_O', 'meta_val', 'meta_val_mixup')}
    banks.update({name: sha256_file(warm/name) for name in ('CTR.p', 'CTR_O.p')})
    if (banks['CTR']['files'] != expected or banks['CTR_O']['files'] != expected or
            not banks['meta_val']['files'] or not banks['meta_val_mixup']['files']):
        raise ValueError('Warm candidate or meta bank is incomplete')
    last_sha, resume_sha = sha256_file(warm/'last.pth'), sha256_file(warm/'resume.pth')
    candidates = [json.loads(line) for line in (warm/'candidates.jsonl').read_text().splitlines()]
    history = [json.loads(line) for line in (warm/'training_history.jsonl').read_text().splitlines()]
    record = {'protocol_signature': signature, 'iteration': iteration,
              'planned_iteration': recipe['warm']['steps'], 'checkpoint_sha256': last_sha,
              'resume_checkpoint_sha256': resume_sha, 'peak_iteration': peak['iteration'],
              'peak_checkpoint_sha256': sha256_file(warm/'best_target_val.pth'),
              'peak_aux_sha256': sha256_file(warm/'best_aux_state.pth'),
              'bank_last_update_iteration': candidates[-1]['iteration'], 'banks': banks,
              'last_logged_iteration': history[-1]['iteration'] if history else iteration,
              'reason': reason or 'User requested early stop after validation decline; only checkpointed updates are retained'}
    if last_sha != sha256_file(warm/'last.pth') or resume_sha != sha256_file(warm/'resume.pth'):
        raise RuntimeError('Warm checkpoints changed while finalizing early stop')
    if persist:
        write_json(marker, record)
    return record


def prepare_stage_output(output, signature):
    """Preserve an interrupted initialization before restarting from its input weights."""
    started_path = output/'started.json'
    if not started_path.is_file():
        return
    if json.loads(started_path.read_text())['protocol_signature'] != signature:
        raise RuntimeError(f'{output.name} is incomplete and belongs to a different protocol')
    resume_path = output/'resume.pth'
    if resume_path.is_file():
        logging.info('Resuming %s from %s', output.name, resume_path)
        return
    history = output/'training_history.jsonl'
    curve = output/'evaluation_curve.jsonl'
    last = output/'last.pth'
    trained = history.is_file() and history.stat().st_size > 0
    if curve.is_file():
        trained |= any(json.loads(line)['iteration'] != 0 for line in curve.read_text().splitlines())
    if last.is_file():
        state = load_checkpoint(last)
        trained |= (state.get('iteration') != 0 or state.get('protocol_signature') != signature or
                    state.get('stage') != output.name)
        del state
    if trained:
        raise RuntimeError(f'{output.name} has training results but no compatible resume.pth; '
                           'preserving the run rather than restarting automatically')
    if output.is_symlink():
        raise RuntimeError('Cannot archive a stage directory that is a symlink')
    archive = output.with_name(f'{output.name}_interrupted_init_{time.time_ns()}')
    output.rename(archive)
    logging.warning('Interrupted %s initialization has no training state; preserved at %s. '
                    'Restarting initialization from the configured input checkpoint.', output.name, archive)


def prepare_aux_warm_extension(args, signature):
    """Reopen a screened warm from full state; retain the original protocol and stop record."""
    warm = args.output/'warm'
    marker = warm/'stopped_early.json'
    request = warm/'extension.json'
    if (warm/'complete.json').is_file():
        raise ValueError('Warm already completed its planned budget')
    # An existing SND continuation references these banks and selected warm weights.
    if args.output.name.endswith('_warm') and (args.output.with_name(args.output.name[:-5])/'protocol.json').is_file():
        raise ValueError('A continuation already references this warm; preserve it instead of extending in place')
    resume_path = warm/'resume.pth'
    state = load_checkpoint(resume_path)
    iteration = state['iteration']
    if (state['protocol_signature'] != signature or state['stage'] != 'warm' or
            state['source_checkpoint_sha256'] != sha256_file(args.source_checkpoint) or
            state['smoke_only'] != args.smoke_only or
            not 0 < iteration <= args.extend_warm_to):
        raise ValueError('Warm extension has incompatible full training state or stop budget')
    del state
    if marker.is_file():
        stopped = json.loads(marker.read_text())
        if (stopped['protocol_signature'] != signature or stopped['iteration'] != iteration or
                stopped['resume_checkpoint_sha256'] != sha256_file(resume_path) or
                stopped['checkpoint_sha256'] != sha256_file(warm/'last.pth')):
            raise ValueError('Warm extension does not match the finalized screen')
        if iteration == args.extend_warm_to:
            logging.info('Warm already screened through step%d; nothing to extend', iteration)
            return False
        previous = warm/f'stopped_early_step{iteration:06d}.json'
        if previous.exists():
            raise ValueError('An archived stop record already exists; preserving both records')
        write_json(request, {'protocol_signature': signature, 'from_iteration': iteration,
                   'to_iteration': args.extend_warm_to, 'resume_checkpoint_sha256': stopped['resume_checkpoint_sha256'],
                   'reason': 'User-requested extra warm budget; original LR/EMA/aux schedule retained'})
        if (warm/'screening_runtime.json').is_file():
            shutil.copy2(warm/'screening_runtime.json', warm/f'screening_runtime_step{iteration:06d}.json')
        marker.rename(previous)
    else:
        recorded = json.loads(request.read_text())
        if recorded['protocol_signature'] != signature or recorded['to_iteration'] != args.extend_warm_to:
            raise ValueError('Interrupted extension belongs to another protocol or stop budget')
    logging.info('Extending warm from full training state at step%d to step%d', iteration, args.extend_warm_to)
    return True


def execute_stage(stage, recipe, args, signature):
    output = args.output/stage
    if stage == 'warm' and getattr(args, 'extend_warm_to', None) is not None:
        if not prepare_aux_warm_extension(args, signature):
            return
    if stage == 'warm' and (output/'stopped_early.json').is_file():
        raise RuntimeError('Warm was finalized early; use its selected peak for SND')
    if stage in ('snd', 'meta-off') and recipe['protocol'] == B2_PROTOCOL:
        selected_warm_iteration(args, getattr(args, 'warm_signature', signature))
    completed = output/'complete.json'
    if completed.exists():
        state = json.loads(completed.read_text())
        if state['protocol_signature'] != signature or sha256_file(output/'last.pth') != state['checkpoint_sha256']:
            raise ValueError(f'Completed {stage} has mismatched provenance')
        logging.info('Verified completed %s; skipping training', stage)
        return
    resume_path = output/'resume.pth'
    prepare_stage_output(output, signature)
    output.mkdir(parents=True, exist_ok=True)
    if stage in ('snd', 'meta-off'):
        warm = args.warm_checkpoint.parent if recipe['protocol'] == B2_PROTOCOL else args.output/'warm'
        if not ((warm/'complete.json').is_file() or (warm/'stopped_early.json').is_file()):
            raise RuntimeError('SND requires a completed or explicitly finalized warm stage')
        if not resume_path.is_file():
            for name in ('CTR', 'CTR_O', 'meta_val', 'meta_val_mixup'):
                shutil.copytree(warm/name, output/name, dirs_exist_ok=False)
            for name in ('CTR.p', 'CTR_O.p'):
                shutil.copy2(warm/name, output/name)
            if args.smoke_only:
                from PIL import Image
                for name in ('CTR', 'CTR_O', 'meta_val', 'meta_val_mixup'):
                    for path in (output/name).glob('*.png'):
                        with Image.open(path) as mask:
                            if mask.size != (recipe['size'], recipe['size']):
                                mask.resize((recipe['size'], recipe['size']), Image.Resampling.NEAREST).save(path)
    cfg = configuration(recipe, stage, args, output)
    (output/'config.yaml').write_text(cfg.dump(), encoding='utf-8')
    if not resume_path.is_file():
        write_json(output/'started.json', {'protocol_signature': signature, 'source_checkpoint': str(args.source_checkpoint)})
    seed_all(args.seed)
    architecture = 'segformer_mit_b2_nvidia_mlp768' if recipe['protocol'] == B2_PROTOCOL else 'resnet101_aspp'
    runtime = SNDRemoteRuntime(args.data_root, output, args.device, recipe['size'], args.target_limit,
                               args.workers, signature, architecture)
    if recipe['protocol'] == B2_PROTOCOL:
        runtime.checkpoint_portable = (lambda: (refresh_aux_result(args, signature), portable(args.output))
                                       if getattr(args, 'aux_settings', None) else lambda: portable(args.output))
        runtime.resume_path = resume_path if resume_path.is_file() else None
        runtime.source_sha = sha256_file(args.source_checkpoint)
        runtime.smoke_only = args.smoke_only
        runtime.warm_signature = getattr(args, 'warm_signature', signature)
        runtime.stage_offset = (0 if stage == 'warm' or getattr(args, 'snd_init', 'warm') == 'source'
                                else recipe['warm']['steps'])
        runtime.total_budget = recipe['warm']['steps'] + recipe['snd']['steps']
        runtime.meta_off = stage == 'meta-off'
        runtime.readout_lr_mult = recipe.get('readout_lr_mult', 10)
        runtime.freeze_source_readout = recipe.get('freeze_source_readout', False)
        if runtime.freeze_source_readout:
            source = load_checkpoint(args.source_checkpoint)
            runtime.source_readout = {name: source['classifier'][name].detach().cpu().clone()
                                     for name in ('linear_pred.weight', 'linear_pred.bias')}
            del source
        if getattr(args, 'aux_settings', None):
            from scripts.snd_srmp import RelationalAux
            source = load_checkpoint(args.source_checkpoint)
            runtime.aux = RelationalAux(source, args.aux_settings, args.device, args.seed, recipe['base_lr'])
            runtime.method = 'snd_srmp_aux'
            runtime.aux_iteration_offset = (recipe['warm']['steps'] if stage == 'snd' and args.aux_start == 'warm' else 0)
            runtime.aux_inherited_forward_count = 0
            if stage == 'snd' and args.aux_start == 'warm':
                anchor = load_checkpoint(args.warm_checkpoint.parent/'best_aux_state.pth')
                selected = json.loads((args.output/'warm_selection.json').read_text())
                if (anchor['iteration'] != selected['iteration'] or
                        anchor['protocol_signature'] != args.warm_signature or
                        anchor['source_checkpoint_sha256'] != runtime.source_sha):
                    raise ValueError('Selected warm weights and auxiliary rebuilder are not from the same step')
                runtime.aux_inherited_forward_count = anchor['aux']['forward_count']
                if runtime.resume_path is None:
                    runtime.aux.load_state_dict(anchor['aux'])
            del source
    runtime.eval_period, runtime.eval_limit, runtime.stop_iteration = args.eval_period, args.eval_limit, cfg.SOLVER.STOP_ITER
    if getattr(args, 'geometry_settings', None) is not None:
        runtime.geometry_settings = args.geometry_settings
        runtime.context_settings = args.geometry_settings.get('context')
    if getattr(args, 'recovery_settings', None) is not None:
        from scripts.dtst_semantic_recovery import SemanticRecovery
        runtime.semantic_recovery = SemanticRecovery(runtime, args.recovery_settings)
    if getattr(args, 'phenology_settings', None) is not None:
        from scripts.dtst_phenology import PhenologyCorrection
        runtime.phenology = PhenologyCorrection(runtime, args.phenology_settings)
    if getattr(args, 'baseline_settings', None) is not None:
        from scripts.rs_baselines import build_baseline
        runtime.baseline = build_baseline(runtime, args.baseline_settings, args.device)
        runtime.method = 'RS_baseline_' + args.baseline_settings['method']
    if getattr(args, 'dtst_settings', None):
        from scripts.dtst_remote import TeacherFeedback
        runtime.dtst_settings = args.dtst_settings
        runtime.dtst_feedback = TeacherFeedback(args.dtst_settings, runtime.device)
        runtime.method = 'DTST_B2'
    if getattr(args, 'gsd_settings', None) and (stage in ('snd', 'meta-off') or getattr(args, 'dtst_settings', None)):
        if args.gsd_settings['arm'] != 'raw':
            from scripts.gsd_reliability import GSDSupervision
            if 'scout' in args.gsd_settings:
                from scripts.gsd_scout import ScoutSupervision
                runtime.source_checkpoint = args.source_checkpoint
                runtime.scout_source_cache = args.gsd_settings['scout']['source_cache']
                runtime.gsd_supervision = ScoutSupervision(runtime, args.gsd_settings)
            else:
                runtime.gsd_supervision = GSDSupervision(runtime, args.gsd_settings)
            runtime.method = ('DTST_GSD_' if getattr(args, 'dtst_settings', None) else 'SND_GSD_') + args.gsd_settings['arm']
            if runtime.resume_path is not None:
                resume = load_checkpoint(runtime.resume_path)
                runtime.gsd_supervision.extra_forwards = resume.get('gsd_extra_teacher_images', 0)
                del resume
    started_at = time.perf_counter()
    encoder, classifier = train(cfg, 0, False, runtime=runtime)
    if runtime.last_iteration != cfg.SOLVER.STOP_ITER:
        raise RuntimeError('Trainer stopped before the requested number of iterations')
    runtime.checkpoint(runtime.last_iteration, encoder, classifier)
    elapsed = time.perf_counter() - started_at
    runtime_record = {'protocol_signature': signature, 'iteration': runtime.last_iteration,
                          'checkpoint_sha256': sha256_file(output/'last.pth'),
                          'wall_seconds': elapsed, 'gpu_hours': elapsed/3600 if args.device.startswith('cuda') else 0,
                          'peak_gpu_memory_mb': (torch.cuda.max_memory_allocated()/1048576 if torch.cuda.is_available() else None),
                          'teacher_inferences': runtime.teacher_inferences,
                          'meta_images': runtime.meta_images, 'bank_refreshes': runtime.bank_refreshes,
                          'aux_forward_count': (runtime.aux.forward_count - runtime.aux_inherited_forward_count
                                                if getattr(runtime, 'aux', None) else 0),
                          'aux_inherited_forward_count': getattr(runtime, 'aux_inherited_forward_count', 0)}
    if stage == 'warm' and cfg.SOLVER.STOP_ITER < recipe[stage]['steps'] and not getattr(args, 'dtst_settings', None):
        finalize_aux_warm(args, recipe, signature,
            reason=('User-requested warm extension; original LR/EMA/aux schedule retained'
                    if getattr(args, 'extend_warm_to', None) is not None else
                    'Fixed screening budget selected before training; planned LR/EMA schedule retained'))
        write_json(output/'screening_runtime.json', runtime_record)
    else:
        write_json(completed, runtime_record)
    del encoder, classifier, runtime
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    portable(args.output)


def main(dtst_settings=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['all', 'warm', 'finalize-warm', 'snd', 'meta-off', 'eval', 'source-check', 'smoke'])
    parser.add_argument('--protocol', choices=('legacy', B2_PROTOCOL), default=B2_PROTOCOL if dtst_settings else 'legacy')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/snd_vh2pd.json')
    parser.add_argument('--source-checkpoint', type=Path, required=True)
    parser.add_argument('--warm-checkpoint', type=Path)
    parser.add_argument('--snd-init', choices=('warm', 'source'), default='warm',
                        help='B2 snd stage: start from warm weights (historical) or source weights + warm banks (author SND)')
    parser.add_argument('--source-seed', type=int, default=1)
    parser.add_argument('--source-selection', choices=['checkpoint_metadata', 'target_eval'],
                        default='checkpoint_metadata',
                        help='Record whether source weights were chosen using target ground-truth metrics')
    parser.add_argument('--data-root', type=Path, default=ROOT/'data/remote_sensing/dgle')
    parser.add_argument('--gsd-config', type=Path, help='Opt-in GSD study; outer selector only in main stage')
    parser.add_argument('--geometry-config', type=Path, help='Opt-in DT-ST geometry/context scout')
    parser.add_argument('--recovery-config', type=Path, help='Opt-in target-only semantic recovery on DT-ST')
    parser.add_argument('--pheno-config', type=Path, help='Opt-in phenology C0/C1 arms on raw DT-ST')
    parser.add_argument('--baseline-config', type=Path, help='Opt-in re-implemented RS baselines (S3T-SFDA, St-CGN) on the DT-ST host')
    parser.add_argument('--gsd-stop-updates', type=int, help='Pilot main endpoint without changing the LR horizon')
    parser.add_argument('--gsd-target-limit', type=int, default=0)
    parser.add_argument('--gsd-eval-limit', type=int, default=0)
    parser.add_argument('--aux-config', type=Path)
    parser.add_argument('--aux-weight', type=float, help='Maximum auxiliary loss coefficient (default: aux config weight)')
    parser.add_argument('--aux-warmup-updates', type=int, help='Updates with zero auxiliary weight')
    parser.add_argument('--aux-ramp-updates', type=int, help='Updates to ramp auxiliary weight to its maximum')
    parser.add_argument('--aux-prepare-updates', type=int, help='R-only preparation inside zero-aux warmup (MRM)')
    parser.add_argument('--aux-soft-kl-weight', type=float, help='Soft KL coefficient inside source-relational auxiliary')
    parser.add_argument('--aux-encoder-grad-cap', type=float, help='Context/contrast aux encoder gradient norm / main encoder norm cap')
    parser.add_argument('--aux-teacher', choices=('ema', 'student'),
                        help='Source-contrast clean target; student uses current eval/no-grad weights after R preparation')
    parser.add_argument('--warm-stop-updates', type=int, help='Finalize auxiliary warm at a fixed checkpoint without changing the planned schedule')
    parser.add_argument('--extend-warm-to', type=int,
                        help='Resume an existing screened auxiliary warm to this absolute step, keeping its original protocol')
    parser.add_argument('--aux-mask-ratio', type=float)
    parser.add_argument('--aux-block-size', type=int, help='Mask block size in feature cells')
    parser.add_argument('--aux-confidence-threshold', type=float,
                        help='Keep masked cells with frozen-source teacher confidence >= threshold; 0 disables')
    parser.add_argument('--aux-view-consistency', type=int, choices=(0, 1),
                        help='Require full/crop EMA source-head argmax agreement for auxiliary targets')
    parser.add_argument('--readout-lr-mult', type=float,
                        help='B2 final Conv1x1 LR multiplier; default10; decoder fusion remains10')
    parser.add_argument('--freeze-source-readout', action='store_true',
                        help='Keep source weight/bias exact in student, EMA and meta inner-loop; requires source-frozen warm')
    parser.add_argument('--objective', choices=('relation', 'probability', 'masked-ce', 'no-mask', 'latent-cosine', 'mrm-ce', 'context-relation', 'source-contrast'))
    parser.add_argument('--aux-start', choices=('snd', 'warm'), default='snd')
    parser.add_argument('--smoke-only', action='store_true', help='Bound aux continuation to two updates')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--eval-period', type=int, default=500,
                        help='Diagnostic target curve; 0 disables monitoring. Never changes weights or stopping.')
    parser.add_argument('--smoke-size', type=int, choices=[64, 512], default=64)
    parser.add_argument('--smoke-batch-size', type=int, choices=[1, 4, 6], default=1)
    args = parser.parse_args()
    args.dtst_settings = dtst_settings
    if dtst_settings:
        if args.stage != 'warm' or args.protocol != B2_PROTOCOL or args.warm_checkpoint or args.aux_config:
            parser.error('DT-ST requires a fresh B2 adaptation stage initialized from source')
        if not args.gsd_config:
            args.gsd_config = ROOT/'configs/gsd_sfuda/raw_irrg.json'
    args.smoke_only = args.smoke_only or args.stage == 'smoke'
    if bool(args.aux_config) != bool(args.objective):
        parser.error('--aux-config and --objective must be set together')
    if args.aux_weight is not None and (not args.aux_config or not math.isfinite(args.aux_weight) or args.aux_weight < 0):
        parser.error('--aux-weight requires --aux-config and a finite nonnegative value')
    if args.aux_warmup_updates is not None and (not args.aux_config or args.aux_warmup_updates < 0):
        parser.error('--aux-warmup-updates requires --aux-config and a nonnegative integer')
    if args.aux_ramp_updates is not None and (not args.aux_config or args.aux_ramp_updates < 1):
        parser.error('--aux-ramp-updates requires --aux-config and a positive integer')
    if args.aux_prepare_updates is not None and (not args.aux_config or args.aux_prepare_updates < 0):
        parser.error('--aux-prepare-updates requires --aux-config and a nonnegative integer')
    if args.aux_soft_kl_weight is not None and (not args.aux_config or
            not math.isfinite(args.aux_soft_kl_weight) or args.aux_soft_kl_weight < 0):
        parser.error('--aux-soft-kl-weight requires --aux-config and a finite nonnegative coefficient')
    if args.aux_encoder_grad_cap is not None and (args.objective not in ('context-relation', 'source-contrast') or
            not math.isfinite(args.aux_encoder_grad_cap) or args.aux_encoder_grad_cap < 0):
        parser.error('--aux-encoder-grad-cap requires context-relation/source-contrast and a finite nonnegative value')
    if args.aux_teacher is not None and args.objective != 'source-contrast':
        parser.error('--aux-teacher requires source-contrast')
    if args.warm_stop_updates is not None and (not args.aux_config or args.aux_start != 'warm' or
            args.warm_stop_updates < 500 or args.warm_stop_updates % 500):
        parser.error('--warm-stop-updates requires auxiliary warm and a positive multiple of500')
    if args.extend_warm_to is not None and (args.protocol != B2_PROTOCOL or args.stage != 'warm' or
            not args.aux_config or args.aux_start != 'warm' or args.warm_stop_updates is None or
            args.extend_warm_to < 500 or args.extend_warm_to % 500):
        parser.error('--extend-warm-to requires B2 screened auxiliary warm and a positive multiple of500')
    if args.aux_mask_ratio is not None and (not args.aux_config or not 0 <= args.aux_mask_ratio < 1):
        parser.error('--aux-mask-ratio requires --aux-config and a value in [0, 1)')
    if args.aux_block_size is not None and (not args.aux_config or args.aux_block_size < 1):
        parser.error('--aux-block-size requires --aux-config and a positive integer')
    if args.aux_confidence_threshold is not None and (not args.aux_config or not 0 <= args.aux_confidence_threshold <= 1):
        parser.error('--aux-confidence-threshold requires --aux-config and a value in [0, 1]')
    if args.aux_view_consistency is not None and (not args.aux_config or
            (args.aux_view_consistency and args.objective in ('latent-cosine', 'mrm-ce', 'context-relation', 'source-contrast'))):
        parser.error('--aux-view-consistency requires a source-probability auxiliary objective')
    if args.readout_lr_mult is not None and (args.protocol != B2_PROTOCOL or
            not math.isfinite(args.readout_lr_mult) or args.readout_lr_mult < 0):
        parser.error('--readout-lr-mult requires B2 and a finite nonnegative value')
    if args.freeze_source_readout and (args.protocol != B2_PROTOCOL or args.readout_lr_mult not in (None, 0)):
        parser.error('--freeze-source-readout requires B2 and readout LR0')
    if args.aux_start == 'warm' and not args.aux_config:
        parser.error('--aux-start warm requires --aux-config and --objective')
    if args.aux_config and (args.protocol != B2_PROTOCOL or
                            args.stage not in (('snd', 'warm', 'finalize-warm') if args.aux_start == 'warm' else ('snd',))):
        parser.error('Auxiliary stage must be B2 snd, or B2 warm with --aux-start warm')
    if args.protocol == B2_PROTOCOL:
        if args.config == ROOT/'configs/snd_vh2pd.json':
            args.config = ROOT/'configs/snd_vh2pd_b2.json'
        if args.data_root == ROOT/'data/remote_sensing/dgle':
            args.data_root = ROOT/'data/remote_sensing/siamseg_sfuda_b2_v1'
        if args.stage in ('snd', 'meta-off') and (not args.warm_checkpoint or not args.warm_checkpoint.is_file()):
            parser.error('B2 snd/meta-off requires --warm-checkpoint')
        if args.stage == 'all':
            parser.error('B2 stages must be launched separately so warm can be verified before fork')
    elif args.stage == 'meta-off':
        parser.error('meta-off is only available for the B2 protocol')
    if args.eval_period < 0 or (args.eval_period and args.eval_period % 500):
        parser.error('--eval-period must be 0 or a multiple of the 500-step checkpoint period')
    if int(os.environ.get('WORLD_SIZE', '1')) > 1 or int(os.environ.get('SLURM_NTASKS', '1')) > 1:
        raise RuntimeError('Use one process/one GPU; do not use torchrun or multiple srun tasks')
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.output/'run.log', encoding='utf-8')])
    recipe = json.loads(args.config.read_text())
    recipe['eval_period'] = args.eval_period
    args.gsd_settings = None
    if args.gsd_config:
        if args.protocol != B2_PROTOCOL or args.aux_config or args.stage not in ('warm', 'snd', 'meta-off', 'smoke'):
            parser.error('GSD study requires B2 warm/main without auxiliary losses')
        args.gsd_settings = json.loads(args.gsd_config.read_text())
        study = args.gsd_settings
        mode, strength = study.get('loss_mode', 'hard'), study.get('intervention', 1.)
        if (mode not in ('hard', 'blend', 'soft_js') or not math.isfinite(strength) or
                not 0 <= strength <= 1 or
                (mode == 'soft_js' and (study['arm'] not in ('gsd', 'generic') or study['coverage'] != 1))):
            parser.error('Invalid GSD loss mode/intervention')
        if (study['arm'] not in ('raw', 'confidence', 'generic', 'gsd') or
                not 0 < study['coverage'] <= 1 or
                not math.isfinite(study['lambda']) or study['lambda'] < 0 or
                study['target_bands'] not in ('IRRG', 'RGB') or args.gsd_target_limit < 0 or args.gsd_eval_limit < 0):
            parser.error('Invalid GSD study configuration')
        pilot_budget = recipe['warm']['steps'] if dtst_settings else recipe['snd']['steps']
        if args.gsd_stop_updates is not None and not 0 < args.gsd_stop_updates <= pilot_budget:
            parser.error('GSD pilot endpoint must be within the main budget')
        if args.stage in ('snd', 'meta-off'):
            if args.warm_checkpoint is None:
                parser.error('GSD main requires a shared warm checkpoint')
            warm_protocol = json.loads((args.warm_checkpoint.parent.parent/'protocol.json').read_text())
            args.warm_signature = warm_protocol['protocol_signature']
            if warm_protocol['source_checkpoint_sha256'] != sha256_file(args.source_checkpoint):
                raise ValueError('GSD source differs from shared warm source')
            for split in ('target_train', 'target_eval'):
                if warm_protocol['manifests'][split] != sha256_file(args.data_root/'manifests'/f'{split}.csv'):
                    raise ValueError('GSD shared warm has another target manifest')
    elif args.gsd_stop_updates is not None or args.gsd_target_limit or args.gsd_eval_limit:
        parser.error('GSD pilot/scope options require --gsd-config')
    args.geometry_settings = None
    args.recovery_settings = None
    if args.recovery_config:
        if not dtst_settings or args.aux_config or args.gsd_settings['arm'] != 'raw' or args.geometry_config:
            parser.error('Semantic recovery requires raw DT-ST without other auxiliaries')
        from scripts.dtst_semantic_recovery import validate_recovery
        args.recovery_settings = validate_recovery(json.loads(args.recovery_config.read_text()))
    args.baseline_settings = None
    if args.baseline_config:
        if not dtst_settings or args.aux_config or args.gsd_settings['arm'] != 'raw' or args.geometry_config or \
                args.recovery_config or args.pheno_config:
            parser.error('Re-implemented baselines require the raw DT-ST host without other auxiliaries')
        from scripts.rs_baselines import load_settings
        args.baseline_settings = load_settings(args.baseline_config)
        if args.baseline_settings['method'] == 'stcgn':
            prototypes = Path(args.baseline_settings['prototypes'])
            if not prototypes.is_absolute():
                prototypes = (args.baseline_config.parent/prototypes).resolve()
            args.baseline_settings['prototypes'] = str(prototypes)
            args.baseline_prototypes_sha256 = sha256_file(prototypes)
    args.phenology_settings = None
    if args.pheno_config:
        if not dtst_settings or args.aux_config or args.gsd_settings['arm'] != 'raw' or args.geometry_config or args.recovery_config:
            parser.error('Phenology arms require raw DT-ST without other auxiliaries')
        from scripts.dtst_phenology import validate_phenology
        args.phenology_settings = validate_phenology(json.loads(args.pheno_config.read_text()))
    if args.geometry_config:
        bank_only = (args.gsd_settings.get('scout', {}).get('variant') == 'source_bank' and
                     args.gsd_settings['scout'].get('support') is False)
        if not dtst_settings or args.aux_config or (args.gsd_settings['arm'] != 'raw' and not bank_only):
            parser.error('Geometry scouts require raw DT-ST warm, without other auxiliary/gating modules')
        from scripts.dtst_geometry import validate_geometry
        args.geometry_settings = validate_geometry(json.loads(args.geometry_config.read_text()))
    if args.readout_lr_mult is not None:
        recipe['readout_lr_mult'] = args.readout_lr_mult
    if args.freeze_source_readout:
        recipe.update(freeze_source_readout=True, readout_lr_mult=0.)
    args.aux_settings = None
    if args.aux_config:
        args.aux_settings = json.loads(args.aux_config.read_text())
        if args.aux_weight is not None:
            args.aux_settings['weight'] = args.aux_weight
        if args.aux_warmup_updates is not None:
            args.aux_settings['warmup_updates'] = args.aux_warmup_updates
        if args.aux_ramp_updates is not None:
            args.aux_settings['ramp_updates'] = args.aux_ramp_updates
        for key in ('mask_ratio', 'block_size', 'confidence_threshold', 'prepare_updates', 'soft_kl_weight', 'encoder_grad_cap'):
            value = getattr(args, 'aux_' + key)
            if value is not None:
                args.aux_settings[key] = value
        if args.aux_view_consistency is not None:
            args.aux_settings['view_consistency'] = bool(args.aux_view_consistency)
        if args.aux_teacher is not None:
            args.aux_settings['target_teacher'] = args.aux_teacher
        args.aux_settings['objective'] = args.objective
        target_teacher = args.aux_settings.get('target_teacher', 'ema')
        if (target_teacher not in ('ema', 'student') or
                (target_teacher != 'ema' and args.objective != 'source-contrast')):
            parser.error('Current-student auxiliary teacher requires source-contrast')
        if target_teacher == 'student':
            args.aux_settings['target_teacher_view'] = 'clean_unmasked_eval_stop_gradient'
        if args.objective == 'context-relation':
            args.aux_settings.setdefault('context_temperature', .1)
            args.aux_settings.setdefault('context_source_weight', 1.)
            args.aux_settings.setdefault('encoder_grad_cap', .1)
            args.aux_settings.update(gradient_routing='encoder_rebuilder',
                context_metric='identity_plus_source_gram', rebuilder_update_weight=1.)
            if (args.aux_settings.get('rebuilder') != 'mrm' or
                    args.aux_settings.get('confidence_threshold', 0) or args.aux_settings.get('view_consistency', False) or
                    not math.isfinite(args.aux_settings['context_temperature']) or args.aux_settings['context_temperature'] <= 0 or
                    not math.isfinite(args.aux_settings['context_source_weight']) or args.aux_settings['context_source_weight'] < 0 or
                    not math.isfinite(args.aux_settings['encoder_grad_cap']) or args.aux_settings['encoder_grad_cap'] < 0):
                parser.error('Context relation requires MRM, no class gate, and finite valid metric/cap settings')
        if args.objective == 'source-contrast':
            args.aux_settings.setdefault('encoder_grad_cap', .1)
            args.aux_settings.update(gradient_routing='encoder_rebuilder', rebuilder_update_weight=1.,
                contrast_similarity='latent_cosine',
                contrast_negative_weight='one_minus_normalized_source_CRS_times_clean_latent_distance',
                contrast_positive='paired_clean_teacher_cell')
            a = args.aux_settings
            if (a.get('rebuilder') != 'mrm' or a.get('confidence_threshold', 0) or a.get('view_consistency', False) or
                    not math.isfinite(a['encoder_grad_cap']) or a['encoder_grad_cap'] < 0 or
                    not math.isfinite(a.get('contrast_temperature', 0)) or a.get('contrast_temperature', 0) <= 0 or
                    type(a.get('contrast_grid')) is not int or a['contrast_grid'] < 1 or
                    type(a.get('contrast_keys_per_class')) is not int or a['contrast_keys_per_class'] < 1):
                parser.error('Source contrast requires MRM, no class gate, and valid contrast/cap settings')
        preparation = args.aux_settings.get('prepare_updates', 0)
        soft_kl = args.aux_settings.get('soft_kl_weight', 0.)
        if (type(preparation) is not int or preparation < 0 or
                preparation > args.aux_settings['warmup_updates'] or
                (preparation and args.aux_settings.get('rebuilder') != 'mrm')):
            parser.error('R preparation requires MRM and prepare_updates within zero-aux warmup')
        if (not math.isfinite(soft_kl) or soft_kl < 0 or
                (soft_kl and args.objective not in ('relation', 'no-mask'))):
            parser.error('Soft KL requires relation/no-mask and a finite nonnegative coefficient')
        if args.aux_settings.get('rebuilder') == 'mrm':
            args.aux_settings['upstream_commit'] = '83d4bb5ebb518289cffba1032662733b68f21642'
            if (args.aux_settings['hidden_dim'] % 16 or args.aux_settings['hidden_dim'] % 4 or
                    args.aux_settings['depth'] < 1):
                parser.error('MRM embedding must be divisible by16 and depth positive')
        if args.stage == 'snd':
            warm_protocol = json.loads((args.warm_checkpoint.parent.parent/'protocol.json').read_text())
            args.warm_signature = warm_protocol['protocol_signature']
            if warm_protocol['source_checkpoint_sha256'] != sha256_file(args.source_checkpoint):
                raise ValueError('Auxiliary source differs from completed warm run')
            if (args.freeze_source_readout and
                    warm_protocol['recipe'].get('readout_lr_mult', 10) != 0):
                raise ValueError('Strict source-frozen SND requires a warm protocol with readout LR0')
            if args.aux_start == 'warm' and warm_protocol.get('aux_start') != 'warm':
                raise ValueError('SND requires a warm run with the same auxiliary phase')
            if args.aux_start == 'warm':
                args.warm_stop_updates = warm_protocol.get('warm_stop_updates')
    if recipe['task'] != 'vh2pd':
        raise ValueError('This integration is currently validated only for VH->PD')
    args.target_limit = args.eval_limit = 0
    if args.gsd_settings:
        args.target_limit, args.eval_limit = args.gsd_target_limit, args.gsd_eval_limit
    if args.smoke_only:
        args.warm_stop_updates = 1 if args.warm_stop_updates is not None else None
        args.extend_warm_to = 2 if args.extend_warm_to is not None else None
        recipe['size'], recipe['batch_size'] = args.smoke_size, args.smoke_batch_size
        recipe['checkpoint_period'] = 2
        recipe['candidate_update'] = 2
        for stage in ('warm', 'snd', 'meta-off'):
            if stage not in recipe:
                continue
            recipe[stage]['steps'] = 2
            recipe[stage]['lr_max_iter'] = 4
            recipe[stage]['meta_update'] = 2
        # The existing smoke warm bank contains the same four fixed target images.
        # Reuse that cohort even when a memory probe raises physical batch size.
        args.target_limit, args.eval_limit = 4, 2
        if args.aux_settings:
            if args.aux_settings.get('prepare_updates', 0):
                args.aux_settings['prepare_updates'] = 1
                args.aux_settings['warmup_updates'] = 1
            else:
                args.aux_settings['warmup_updates'] = 0
            args.aux_settings['ramp_updates'] = 1
    if args.aux_settings and args.aux_settings.get('target_teacher') == 'student':
        args.aux_settings['target_teacher_start_updates'] = args.aux_settings['warmup_updates']
    args.warm_planned_updates = recipe['warm']['steps']
    if args.warm_stop_updates is not None and args.warm_stop_updates >= args.warm_planned_updates:
        parser.error('--warm-stop-updates must be smaller than the planned warm budget')
    if args.extend_warm_to is not None and not args.warm_stop_updates < args.extend_warm_to <= args.warm_planned_updates:
        parser.error('--extend-warm-to must exceed the initial screen stop and not exceed the planned warm budget')
    if args.aux_settings and args.stage == 'snd' and args.aux_start == 'warm':
        if warm_protocol.get('auxiliary') != args.aux_settings:
            raise ValueError('SND must continue the same auxiliary warm objective and settings')
    source = SNDRemoteRuntime.load_source(args.source_checkpoint)
    source_split = 'source_train' if args.protocol == B2_PROTOCOL else 'isprs_vh_irrg_train'
    if source.get('source_manifest_sha256') != sha256_file(args.data_root/'manifests'/f'{source_split}.csv'):
        raise ValueError('Source checkpoint was trained on a different source manifest')
    if args.protocol == B2_PROTOCOL:
        if source.get('protocol_sha256') != sha256_file(ROOT/'configs/remote_sensing/siamseg_sfuda_b2_v1.json'):
            raise ValueError('B2 protocol config hash mismatch')
        if source.get('smoke_only') != args.smoke_only:
            raise ValueError('Smoke and full checkpoints cannot be mixed')
    source_provenance = {key: source.get(key) for key in (
        'iteration', 'train_policy', 'checkpoint_selection', 'training_settings',
        'source_manifest_sha256', 'source_val_manifest_sha256')}
    del source
    source_sha = sha256_file(args.source_checkpoint)
    protocol = {'method': 'SND', 'protocol': recipe['protocol'], 'recipe': recipe,
                'seed': args.seed, 'source_seed': args.source_seed, 'smoke': args.smoke_only, 'source_checkpoint_sha256': source_sha,
                'source_provenance': source_provenance,
                'source_selection': args.source_selection,
                'target_labels_used_for_source_selection': args.source_selection == 'target_eval',
                'source_initialization': ('Warm uses source; SND and meta-off fork the same warm checkpoint and bank'
                                          if args.protocol == B2_PROTOCOL else
                                          'Both warm and SND initialize all parameters from the same source checkpoint'),
                'meta_bank': 'SND inherits and freezes the stable meta banks from warm, following author commands',
                'target_labels_used_for_adaptation': False,
                'evaluation': 'Periodic diagnostic curve and final EMA, native-resolution GT, all six classes; fixed stopping',
                'target_labels_used_for_monitoring': args.eval_period > 0,
                'peak_reporting': ('target_validation_selected and final, with saved peak checkpoint'
                                   if args.protocol == B2_PROTOCOL else
                                   'Oracle observed peak reported separately; never used to select the main result'),
                'adaptations': ['512 square patches; half-resolution candidate caches restored to 512',
                                'all six classes eligible for object paste; no street co-occurrence rules',
                                'explicit scale/crop padding alignment and ignore padded pixels',
                                'nonempty-class sampling; bounded object selection; native source model retained',
                                'shared-memory copy-paste probabilities visible to DataLoader workers'],
                'manifests': {name: sha256_file(args.data_root/'manifests'/f'{name}.csv') for name in
                              (('source_train', 'target_train', 'target_eval') if args.protocol == B2_PROTOCOL else
                               ('isprs_vh_irrg_train', 'isprs_pd_rgb_train', 'isprs_pd_rgb_eval'))},
                'code_sha256': {p: sha256_file(ROOT/p) for p in (B2_CODE_FILES if args.protocol == B2_PROTOCOL else CODE_FILES)},
                'torch': str(torch.__version__), 'numpy': str(np.__version__)}
    if args.gsd_settings:
        metadata = json.loads((args.data_root/'metadata.json').read_text())
        if metadata.get('gsd_study_variant') != args.gsd_settings['target_bands']:
            raise ValueError('Study target bands differ from the prepared dataset')
        protocol['gsd_study'] = args.gsd_settings
        protocol['gsd_scope'] = {'main_stop_updates': args.gsd_stop_updates,
                                 'target_limit': args.target_limit, 'eval_limit': args.eval_limit}
        protocol['data_metadata_sha256'] = sha256_file(args.data_root/'metadata.json')
        cache = args.data_root / args.gsd_settings['view_root'] / 'view_manifest.json'
        protocol['gsd_view_manifest_sha256'] = sha256_file(cache)
        protocol['code_sha256']['scripts/gsd_reliability.py'] = sha256_file(ROOT/'scripts/gsd_reliability.py')
        if 'scout' in args.gsd_settings:
            protocol['code_sha256']['scripts/gsd_scout.py'] = sha256_file(ROOT/'scripts/gsd_scout.py')
        if args.stage in ('snd', 'meta-off'):
            protocol['warm_protocol_signature'] = args.warm_signature
            protocol['warm_checkpoint_sha256'] = sha256_file(args.warm_checkpoint)
            if args.snd_init == 'source':
                protocol['snd_init'] = 'source weights + warm banks; LR clock from 0 (author SND run.sh)'
            protocol['warm_bank_sha256'] = {name: bank_digest(args.warm_checkpoint.parent/name)
                                            for name in ('CTR', 'CTR_O', 'meta_val', 'meta_val_mixup')}
    if dtst_settings:
        protocol.update(method='DTST_B2', dtst=dtst_settings,
                        source_initialization='Student and EMA teacher start from source; adaptation clock starts at0',
                        meta_bank='Not used: DT-ST has no bi-level/meta optimization',
                        selector_scope='Positive recipient CE from adaptation step1; negative learning, DTU and TCR unchanged',
                        feedback_adaptation='Both history/current probes use eval mode; preserve source BN; same sampled100 pixel IDs',
                        lr_adaptation='Existing B2 AdamW, linear1500 warmup/32000 horizon; fixed20000 adaptation endpoint')
        protocol['code_sha256'].update({name: sha256_file(ROOT/name) for name in
                                       ('train_DTST_remote.py', 'scripts/dtst_remote.py')})
    if args.recovery_settings is not None:
        protocol['semantic_recovery'] = args.recovery_settings
        protocol['method'] = 'DTST_semantic_recovery'
        protocol['selector_scope'] = 'Target-only cross-image semantic correction on recipient pixels; corrected positive class protected from negative BCE; donor labels unchanged'
        protocol['extra_pretrained_model'] = args.recovery_settings['features'] == 'dinov2'
        protocol['code_sha256']['scripts/dtst_semantic_recovery.py'] = sha256_file(ROOT/'scripts/dtst_semantic_recovery.py')
    if getattr(args, 'baseline_settings', None) is not None:
        protocol['baseline'] = {k: v for k, v in args.baseline_settings.items() if k != 'prototypes'}
        protocol['method'] = 'RS_baseline_' + args.baseline_settings['method']
        protocol['code_sha256']['scripts/rs_baselines.py'] = sha256_file(ROOT/'scripts/rs_baselines.py')
        if hasattr(args, 'baseline_prototypes_sha256'):
            protocol['baseline_prototypes_sha256'] = args.baseline_prototypes_sha256
    if getattr(args, 'phenology_settings', None) is not None:
        protocol['phenology'] = args.phenology_settings
        protocol['method'] = 'DTST_phenology'
        protocol['selector_scope'] = 'C0 teacher tree-logit offset from leaf-off counterfactual; C1 class-conditional leaf-off student input edit'
        protocol['code_sha256']['scripts/dtst_phenology.py'] = sha256_file(ROOT/'scripts/dtst_phenology.py')
    if args.aux_settings:
        protocol['method'] = 'snd_srmp_aux'
        protocol['auxiliary'] = args.aux_settings
        if args.warm_stop_updates is not None:
            protocol['warm_stop_updates'] = args.warm_stop_updates
        if args.aux_start == 'warm':
            protocol['aux_start'] = 'warm'
        if args.stage == 'snd':
            protocol['warm_protocol_signature'] = args.warm_signature
            protocol['warm_checkpoint_sha256'] = sha256_file(args.warm_checkpoint)
            warm = args.warm_checkpoint.parent
            protocol['warm_bank_sha256'] = {name: bank_digest(warm/name) for name in
                                            ('CTR', 'CTR_O', 'meta_val', 'meta_val_mixup')}
            protocol['warm_bank_sha256'].update({name: sha256_file(warm/name) for name in ('CTR.p', 'CTR_O.p')})
            early = warm/'stopped_early.json'
            if early.is_file():
                stop = json.loads(early.read_text())
                if stop['banks'] != protocol['warm_bank_sha256']:
                    raise ValueError('Early-stopped warm bank changed after finalization')
                protocol['warm_early_stop_sha256'] = sha256_file(early)
                protocol['warm_actual_updates'] = stop['iteration']
                protocol['warm_schedule_offset'] = recipe['warm']['steps']
        protocol['aux_config_sha256'] = sha256_file(args.aux_config)
        protocol['code_sha256'].update({name: sha256_file(ROOT/name) for name in
                                        ('scripts/snd_srmp.py', 'train_SRMP.py')})
        if args.aux_settings.get('rebuilder') == 'mrm':
            import timm
            protocol['timm'] = timm.__version__
            protocol['code_sha256']['core/models/mrm_rebuilder.py'] = sha256_file(ROOT/'core/models/mrm_rebuilder.py')
    if args.geometry_settings is not None:
        protocol['geometry_study'] = args.geometry_settings
        protocol['geometry_config_sha256'] = sha256_file(args.geometry_config)
        protocol['code_sha256']['scripts/dtst_geometry.py'] = sha256_file(ROOT/'scripts/dtst_geometry.py')
    signature = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    path = args.output/'protocol.json'
    if path.exists():
        existing = json.loads(path.read_text())
        if existing['protocol_signature'] != signature:
            comparable = dict(existing)
            comparable.pop('protocol_signature')
            comparable['code_sha256'] = dict(comparable['code_sha256'])
            comparable['code_sha256']['train_SND_remote.py'] = protocol['code_sha256']['train_SND_remote.py']
            compatible = {'train_SND_remote.py', 'train_SND.py', 'scripts/snd_remote.py',
                          'scripts/diagnose_crots_remote.py', 'core/models/segformer.py'}
            # Default filtering is off; added gradient/readout diagnostics do not change old objectives.
            if (args.aux_settings and not args.aux_settings.get('view_consistency', False) and
                    not args.aux_settings.get('prepare_updates', 0) and not args.aux_settings.get('soft_kl_weight', 0) and
                    not args.aux_settings.get('confidence_threshold', 0) and
                    args.aux_settings.get('rebuilder', 'conv') == 'conv'):
                compatible.add('scripts/snd_srmp.py')
            # This exact pre-gate implementation remains numerically identical when the new gate is off.
            if (args.aux_settings and not args.aux_settings.get('view_consistency', False) and
                    not args.aux_settings.get('prepare_updates', 0) and not args.aux_settings.get('soft_kl_weight', 0) and
                    existing['code_sha256'].get('scripts/snd_srmp.py') ==
                    '8c72d3b08f96f4daae0075fdb17513bef873eb2ee452984ece0b884a9c95024e'):
                compatible.add('scripts/snd_srmp.py')
            # Exact pre-preparation/soft-KL version, including gated runs; new features must be off.
            if (args.aux_settings and not args.aux_settings.get('prepare_updates', 0) and
                    not args.aux_settings.get('soft_kl_weight', 0) and
                    existing['code_sha256'].get('scripts/snd_srmp.py') ==
                    'dcbbe879d28ba38866f1743ddc84dcac4bfcfa1523a7637bbed21992278fc4bc'):
                compatible.add('scripts/snd_srmp.py')
            # Context is a separate branch; the immediately preceding objectives are exact.
            if (args.aux_settings and args.objective != 'context-relation' and
                    existing['code_sha256'].get('scripts/snd_srmp.py') ==
                    '22a9c007f9f277f4eeb838ecf004532a20da0e5a83ef1d9668b5a21358ddd76c'):
                compatible.add('scripts/snd_srmp.py')
            # The contrastive objective adds a separate branch; old context/MRM paths are exact.
            if (args.aux_settings and args.objective != 'source-contrast' and
                    existing['code_sha256'].get('scripts/snd_srmp.py') ==
                    '528a1ee313dd87eb56048f257f8aa989772542b9e1c1f1590b066fe114fcd968'):
                compatible.add('scripts/snd_srmp.py')
            # The new target is opt-in; the exact preceding EMA path is unchanged.
            if (args.aux_settings and args.aux_settings.get('target_teacher', 'ema') == 'ema' and
                    existing['code_sha256'].get('scripts/snd_srmp.py') ==
                    '7c2203da7d9829b069489c19dbf37316fdb4b4218266c1d4c1ef36e206278d46'):
                compatible.add('scripts/snd_srmp.py')
            for name in compatible:
                if name in comparable['code_sha256']:
                    comparable['code_sha256'][name] = protocol['code_sha256'][name]
            compatible_run = ((args.stage in ('snd', 'meta-off') and
                               (not args.aux_settings or args.aux_start == 'snd')) or
                              (args.stage == 'snd' and args.aux_start == 'warm' and args.objective in ('context-relation', 'source-contrast')) or
                              (args.stage in ('warm', 'finalize-warm') and args.aux_start == 'warm'))
            if not compatible_run or comparable != protocol:
                raise ValueError('Output belongs to a different configuration/source/code; use a new RUN_NAME')
            write_json(args.output/'code_upgrade.json', {
                'warm_protocol_signature': existing['protocol_signature'],
                'runner_sha256': protocol['code_sha256']['train_SND_remote.py'],
                'reason': 'Compatible runner extension; training state and original protocol signature retained'})
            signature = existing['protocol_signature']
    else:
        write_json(path, dict(protocol, protocol_signature=signature))
    logging.info('Source SHA256 %s', source_sha)
    try:
        if args.protocol == B2_PROTOCOL:
            if args.aux_settings:
                if args.stage == 'finalize-warm':
                    finalize_aux_warm(args, recipe, signature)
                    refresh_aux_result(args, signature)
                    portable(args.output)
                    return
                execute_stage(args.stage, recipe, args, signature)
                refresh_aux_result(args, signature)
                portable(args.output)
                return
            if args.stage in ('warm', 'smoke'):
                execute_stage('warm', recipe, args, signature)
            if args.stage == 'smoke':
                args.warm_checkpoint = args.output/'warm/last.pth'
            if args.stage in ('snd', 'smoke'):
                execute_stage('snd', recipe, args, signature)
            if args.stage in ('meta-off', 'smoke'):
                execute_stage('meta-off', recipe, args, signature)
            if args.eval_period or args.stage == 'eval':
                refresh_b2_result(args, signature)
            return
        if args.stage == 'source-check':
            metrics = evaluate_checkpoint(args.source_checkpoint, args, 'isprs_vh_irrg_train')
            write_json(args.output/'source_fit.json', {'scope': 'Source TRAIN fit diagnostic, not held-out validation', 'metrics': metrics})
            logging.info('Source-domain train-fit mIoU %.4f%%', metrics['miou']*100)
            return
        if args.stage in ('all', 'warm', 'smoke'):
            execute_stage('warm', recipe, args, signature)
        if args.stage in ('all', 'snd', 'smoke'):
            execute_stage('snd', recipe, args, signature)
        if args.stage in ('all', 'eval', 'smoke'):
            results = {}
            for name, checkpoint in [('source', args.source_checkpoint), ('warm', args.output/'warm/last.pth'), ('snd', args.output/'snd/last.pth')]:
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                if name != 'source':
                    completed = json.loads((checkpoint.parent/'complete.json').read_text())
                    if completed['protocol_signature'] != signature or completed['checkpoint_sha256'] != sha256_file(checkpoint):
                        raise ValueError('Evaluation checkpoint provenance mismatch')
                results[name] = {'checkpoint_sha256': sha256_file(checkpoint),
                                 'metrics': evaluate_checkpoint(checkpoint, args, 'isprs_pd_rgb_eval')}
                write_json(args.output/'result_portable.json', {'protocol_signature': signature, 'smoke': args.stage == 'smoke',
                           'task': 'vh2pd', 'split': 'isprs_pd_rgb_eval', 'results': results})
                logging.info('%s mIoU %.4f%%', name, results[name]['metrics']['miou']*100)
    except Exception as error:
        write_json(args.output/'error.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        portable(args.output)
        logging.info('Portable report: %s', args.output/'results_portable.zip')


if __name__ == '__main__':
    main()
