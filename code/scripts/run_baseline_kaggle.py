"""One baseline experiment per Kaggle batch session, with checkpoint-safe time pauses (08/11/2026).

Baselines run on the same VH IRRG -> PD IRRG-file protocol as PIC-ST (siamseg_sfuda_b2_v1, source40k,
batch 6, seed 1, encoder LR 2e-5, readout x1, eval 1k, patience 2, cap 9k with the rising-cap rule).

snd_baseline: SND (Zhao et al., CVPR 2024) = the author two-stage pipeline: DT-ST warm, then the SND
stage (stable-neighbour banks + bi-level meta pseudo-label weighting, EMA 0.999, meta top-k 0.01,
bank refresh 3000). The warm stage is the existing Kaggle DT-ST IRRG null run (label-free last.pth at 9k,
attached as a kernel output); the SND stage uses the registered recipe.json 'snd' section unchanged
(LR schedule continues at warm offset 20000 of horizon 32000, as in every earlier B2 SND run).
stable_st: Stable-ST (Zhao et al., IJCV 2026; author train_stable_st.py = the SND trainer + the NFD proxy).
One stage from source40k, following the author's full config (deeplabv2_r101_StableST_G2C.yaml): dynamic
teacher with the NFD proxy (query start 10, step 5, max interval 21, EMA 0.99), bi-level meta pseudo-label
weighting (top-k 0.05, refresh 3000), TCR candidates 0.5/3000, no DT-ST negative learning. The author's
SegFormer config disables both DTU and METAPL because its NFD code needs a tensor feature; the B2 port
uses the last MiT stage as the NFD feature (see scripts/dtst_remote.py). Optimiser/LR/batch follow the
common B2 recipe, as for every row.
The stage-generic control below mirrors scripts/dtst_run_control.py (which is fixed to the DT-ST warm
stage and entry) without changing it.
"""
import argparse,datetime,json,os,signal,subprocess,sys,time,uuid
from pathlib import Path
from scripts.dtst_run_control import read_curve,write_json,stop_decision,digest,process_running
from scripts.dtst_pilot_budget import RISING_CAP_RULE

BUDGET_NAME='baseline_irrg_budget_20261108_v1'
SOURCE_SHA='10fbe026b5f2dcef331a99d97bab551f7cb0a2cf95feac2625b82ca3e3449d31'
# experiment -> (trainer entry, stage directory)
EXPERIMENTS={'snd_baseline':('train_SND_remote.py','snd'),'stable_st':('train_DTST_remote.py','warm'),
             'snd_author':('train_SND_remote.py','snd'),
             's3t_sfda':('train_DTST_remote.py','warm'),'stcgn':('train_DTST_remote.py','warm')}
NEEDS_WARM=('snd_baseline','snd_author')
CAP=9000


def _common_recipe(recipe):
    return (recipe['batch_size']==6 and recipe['base_lr']==2e-5 and recipe.get('readout_lr_mult')==1 and
            recipe['eval_period']==1000 and recipe['warm']['steps']==20000 and recipe['warm']['lr_max_iter']==32000 and
            recipe['snd']['steps']==12000 and recipe['snd']['lr_max_iter']==32000)


def _protocol_ok(phase,protocol,source_sha=SOURCE_SHA):
    recipe=protocol['recipe']
    if phase in ('s3t_sfda','stcgn'):
        dtst=protocol.get('dtst') or {};base=protocol.get('baseline') or {}
        expected=dict(s3t_sfda=(True,1.0,.99,12),stcgn=(False,0.0,.996,1))[phase]
        return (_common_recipe(recipe) and recipe['warm']['meta'] is False and base.get('method')==phase and
                protocol['source_checkpoint_sha256']==source_sha and protocol['gsd_study']['arm']=='raw' and
                protocol['gsd_study']['target_bands']=='IRRG' and 'phenology' not in protocol and
                (dtst.get('dynamic_teacher'),dtst.get('negative_weight'),dtst.get('ema'),dtst.get('fixed_interval'))==expected)
    if (not _common_recipe(recipe) or protocol['source_checkpoint_sha256']!=SOURCE_SHA or
            protocol['gsd_study']['arm']!='raw' or protocol['gsd_study']['target_bands']!='IRRG' or 'phenology' in protocol):
        return False
    if phase in NEEDS_WARM:
        return (recipe['snd']['meta'] is True and 'dtst' not in protocol and
                ('snd_init' in protocol)==(phase=='snd_author'))
    dtst=protocol.get('dtst') or {}
    return (recipe['warm']['meta'] is True and recipe['warm']['meta_top_k']==0.05 and recipe['warm']['meta_update']==3000 and
            dtst.get('proxy')=='nfd' and dtst.get('negative_weight')==0 and dtst.get('dynamic_teacher') is True and
            (dtst.get('query_start'),dtst.get('query_step'),dtst.get('max_interval'),dtst.get('ema'))==(10,5,21,.99))


def validate_budget(budget,policy,protocol=None,source_sha=SOURCE_SHA):
    if (budget.get('name')!=BUDGET_NAME or budget.get('phase') not in EXPERIMENTS or budget.get('max_updates')!=CAP or
            budget.get('original_policy')!=policy or budget.get('rising_cap_rule')!=RISING_CAP_RULE or
            policy.get('mode')!='no_improvement' or policy.get('patience_evaluations')!=2 or
            policy.get('eval_period')!=1000 or policy.get('max_updates')!=20000 or
            budget.get('user_requested') is not True or budget.get('applied_after_training_started') is not False):
        raise ValueError('Invalid baseline budget')
    if protocol is not None and not _protocol_ok(budget['phase'],protocol,source_sha):
        raise ValueError('Baseline protocol drift')
    return CAP


def cap_extension_active(curve,budget,policy):
    cap=validate_budget(budget,policy)
    at_cap=next((p for p in curve if p['iteration']==cap),None)
    before=[p['metrics']['miou'] for p in curve if p['iteration']<cap]
    return bool(at_cap and before and at_cap['metrics']['miou']>max(before))


def cap_decision(curve,budget,policy):
    cap=validate_budget(budget,policy)
    if not curve or curve[-1]['iteration']<cap:return None
    peak=max(curve,key=lambda point:point['metrics']['miou'])
    if cap_extension_active(curve,budget,policy):
        if curve[-1]['iteration']>=policy['max_updates'] or curve[-1]['metrics']['miou']>=peak['metrics']['miou']:return None
        return dict(iteration=curve[-1]['iteration'],peak_iteration=peak['iteration'],peak_miou=peak['metrics']['miou'],
                    recent_miou=[curve[-1]['metrics']['miou']],pilot_cap_extension=True,pilot_budget=budget,
                    reason='New peak at pilot cap; first subsequent validation strictly below peak')
    if curve[-1]['iteration']!=cap:raise RuntimeError('Saved pilot cap checkpoint window was missed')
    return dict(iteration=cap,peak_iteration=peak['iteration'],peak_miou=peak['metrics']['miou'],
                recent_miou=[curve[-1]['metrics']['miou']],pilot_budget_cap=True,pilot_budget=budget,
                reason='User-declared pilot budget reached; convergence not claimed')


def scientific_stop(curve,policy,budget):
    early=stop_decision(curve,policy);cap=cap_decision(curve,budget,policy)
    return cap if cap_extension_active(curve,budget,policy) else early or cap


def trim_uncommitted(run,stage_name,iteration):
    archive=run/'session_history_archive';archive.mkdir(exist_ok=True)
    for name in ('training_history.jsonl','teacher_feedback.jsonl'):
        path=run/stage_name/name
        if not path.exists():continue
        records=read_curve(path)
        if any(row.get('iteration',0)>iteration for row in records):
            (archive/(str(time.time_ns())+'_'+name)).write_bytes(path.read_bytes())
            path.write_text(''.join(json.dumps(row)+'\n' for row in records if row.get('iteration',0)<=iteration))


def checked_process(pid,run,allocation,entry):
    proc=Path('/proc')/str(pid)
    try:
        if proc.stat().st_uid!=os.getuid():raise RuntimeError('Refusing to control a process owned by someone else')
        command=[part.decode() for part in (proc/'cmdline').read_bytes().split(b'\0') if part]
        if not command:return False
        environment=(proc/'environ').read_bytes().split(b'\0')
        if ('DTST_LOCAL_RUN_ID='+allocation[6:]).encode() not in environment:
            raise RuntimeError('Trainer is outside the authorized session')
        if next((Path(part).name for part in command[1:] if part.endswith('.py')),None)!=entry:
            raise RuntimeError('Not the expected baseline trainer')
        if (proc/'comm').read_text().strip()=='pt_data_worker':raise RuntimeError('Refusing to signal a DataLoader worker')
        ppid=(proc/'stat').read_text().split(') ',1)[1].split()[1]
        parent=Path('/proc')/ppid/'cmdline'
        if parent.exists():
            parent_command=[part.decode() for part in parent.read_bytes().split(b'\0') if part]
            if next((Path(part).name for part in parent_command[1:] if part.endswith('.py')),None)==entry:
                raise RuntimeError('Refusing to signal a forked trainer worker')
        if '--output' not in command or Path(command[command.index('--output')+1]).resolve()!=run.resolve():
            raise RuntimeError('Trainer output differs from the intended run')
        return True
    except FileNotFoundError:return False
    except PermissionError:
        if not process_running(pid):return False
        raise


def stop_at_checkpoint(pid,run,stage_name,entry,allocation,decision,policy,stop_kind='early_stop'):
    if stop_kind not in ('early_stop','session_pause'):raise ValueError('Unknown checkpoint stop kind')
    import torch
    stage=run/stage_name
    if (stage/'complete.json').exists() or (stop_kind=='session_pause' and decision['iteration']>=policy['max_updates']):
        return False
    if not checked_process(pid,run,allocation,entry):return False
    os.kill(pid,signal.SIGSTOP);terminated=False
    try:
        if any(stage.glob('*.pth.tmp')):return False
        protocol=json.loads((run/'protocol.json').read_text())
        states=[torch.load(stage/name,map_location='cpu',weights_only=False) for name in ('last.pth','resume.pth','best_target_val.pth')]
        last,resume,peak=states
        if last['iteration']!=decision['iteration'] or resume['iteration']!=decision['iteration']:return False
        for state in states:
            if (state['protocol_signature']!=protocol['protocol_signature'] or
                    state['source_checkpoint_sha256']!=protocol['source_checkpoint_sha256']):
                raise ValueError('Stop checkpoint provenance mismatch')
        if peak['iteration']!=decision['peak_iteration']:return False
        latest=read_curve(stage/'training_history.jsonl')
        record=dict(decision,protocol_signature=protocol['protocol_signature'],planned_iteration=policy['max_updates'],
                    stopping_policy=policy,checkpoint_sha256=digest(stage/'last.pth'),
                    resume_checkpoint_sha256=digest(stage/'resume.pth'),peak_checkpoint_sha256=digest(stage/'best_target_val.pth'),
                    last_logged_iteration=latest[-1]['iteration'] if latest else decision['iteration'],
                    retained_iteration=decision['iteration'],target_labels_used_for_stopping=stop_kind=='early_stop',
                    stopped_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=7))).isoformat(),
                    complete_fixed_budget=False)
        if not checked_process(pid,run,allocation,entry):return False
        write_json(stage/'early_stop_intent.json',dict(record,trainer_pid=pid,allocation=allocation))
        os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT);terminated=True
        for _ in range(60):
            if not process_running(pid):break
            time.sleep(.5)
        else:raise RuntimeError('Trainer has not exited; no stop marker written')
        write_json(stage/('stopped_early.json' if stop_kind=='early_stop' else 'session_pause.json'),record)
        print(('Early stopped' if stop_kind=='early_stop' else 'Session paused'),run,'at',decision['iteration'],flush=True)
        return True
    finally:
        if not terminated and checked_process(pid,run,allocation,entry):os.kill(pid,signal.SIGCONT)


def export_manifest(run):
    files={str(p.relative_to(run)):dict(bytes=p.stat().st_size,sha256=digest(p))
           for p in run.rglob('*') if p.is_file() and not p.name.endswith('.tmp') and p.name!='SESSION_MANIFEST.json'}
    write_json(run/'SESSION_MANIFEST.json',dict(files=files,experiment=run.parent.name))


def verify_warm(warm,kind='dtst_null'):
    """dtst_null: label-free 9k endpoint of the Kaggle DT-ST IRRG null run.
    snd_warm: the completed author-style SND warm stage (fixed EMA 0.99/12, no meta, raw IRRG, source40k)."""
    if kind=='snd_warm':
        done=json.loads((warm/'complete.json').read_text());protocol=json.loads((warm.parent/'protocol.json').read_text())
        w=protocol['recipe']['warm']
        if (digest(warm/'last.pth')!=done['checkpoint_sha256'] or done['protocol_signature']!=protocol['protocol_signature'] or
                protocol['source_checkpoint_sha256']!=SOURCE_SHA or protocol.get('method')!='SND' or
                protocol['gsd_study']['arm']!='raw' or protocol['gsd_study']['target_bands']!='IRRG' or
                (w['ema'],w['ema_interval'],w['meta'])!=(.99,12,False)):
            raise ValueError('Warm checkpoint is not the verified author-style SND warm stage')
        for name in ('CTR','CTR_O','meta_val','meta_val_mixup'):
            if not (warm/name).is_dir() or not any((warm/name).iterdir()):raise FileNotFoundError(warm/name)
        return dict(kind=kind,iteration=done['iteration'],checkpoint_sha256=done['checkpoint_sha256'],
                    warm_protocol_signature=protocol['protocol_signature'])
    stop=json.loads((warm/'stopped_early.json').read_text())
    protocol=json.loads((warm.parent/'protocol.json').read_text())
    if (digest(warm/'last.pth')!=stop['checkpoint_sha256'] or stop['iteration']!=CAP or
            stop['protocol_signature']!=protocol['protocol_signature'] or protocol['source_checkpoint_sha256']!=SOURCE_SHA or
            protocol.get('method')!='DTST_phenology' or protocol['phenology']['c0'] or protocol['phenology']['c1'] or
            protocol['phenology'].get('fixed_delta') is not None or protocol['gsd_study']['target_bands']!='IRRG'):
        raise ValueError('Warm checkpoint is not the verified DT-ST IRRG null 9k endpoint')
    for name in ('CTR','CTR_O','meta_val','meta_val_mixup'):
        if not (warm/name).is_dir() or not any((warm/name).iterdir()):raise FileNotFoundError(warm/name)
    for name in ('CTR.p','CTR_O.p'):
        if not (warm/name).is_file():raise FileNotFoundError(warm/name)
    return dict(iteration=stop['iteration'],checkpoint_sha256=stop['checkpoint_sha256'],warm_protocol_signature=protocol['protocol_signature'])


def prepare_stcgn(args,run):
    """St-CGN source-side warm-up (once) and the resolved ST-stage settings with the anchor prototypes."""
    if args.source_root is None:raise ValueError('--source-root is required for stcgn')
    warm=run/'warmup';done=warm/'WARMUP_COMPLETE.json'
    if not done.exists():
        subprocess.run([sys.executable,'-u',str(args.workspace/'train_stcgn_warmup.py'),'--source-checkpoint',str(args.source),
                        '--source-root',str(args.source_root),'--output',str(warm),'--workers','2'],cwd=args.workspace,check=True)
    record=json.loads(done.read_text())
    if (record['parent_source_sha256']!=SOURCE_SHA or digest(warm/'stcgn_warmup.pth')!=record['checkpoint_sha256'] or
            digest(warm/'stcgn_prototypes.pt')!=record['prototypes_sha256']):
        raise ValueError('St-CGN warm-up provenance mismatch')
    settings=json.loads((args.settings/'stcgn.json').read_text());settings['prototypes']=str((warm/'stcgn_prototypes.pt').resolve())
    resolved=run/'stcgn_resolved.json'
    if resolved.exists():assert json.loads(resolved.read_text())==settings,'Resolved St-CGN settings changed'
    else:write_json(resolved,settings)
    return record['checkpoint_sha256']


def command(args,run):
    entry,_=EXPERIMENTS[args.experiment]
    if args.experiment=='stable_st':
        return [sys.executable,'-u',str(args.workspace/entry),'--config',str(args.settings/'recipe_stable_st.json'),
                '--dtst-config',str(args.settings/'dtst_stable_st.json'),'--source-checkpoint',str(args.source),
                '--data-root',str(args.data_root),'--workers','2','--seed','1','--readout-lr-mult','1',
                '--gsd-config',str(args.settings/'raw_irrg.json'),'--eval-period','1000','--output',str(run)]
    if args.experiment=='s3t_sfda':
        return [sys.executable,'-u',str(args.workspace/entry),'--config',str(args.settings/'recipe.json'),
                '--dtst-config',str(args.settings/'dtst_s3t.json'),'--source-checkpoint',str(args.source),
                '--data-root',str(args.data_root),'--workers','2','--seed','1','--readout-lr-mult','1',
                '--gsd-config',str(args.settings/'raw_irrg.json'),'--baseline-config',str(args.settings/'s3t_sfda.json'),
                '--eval-period','1000','--output',str(run)]
    if args.experiment=='stcgn':  # ST stage starts from the St-CGN warm-up teacher (source-side stage)
        return [sys.executable,'-u',str(args.workspace/entry),'--config',str(args.settings/'recipe.json'),
                '--dtst-config',str(args.settings/'dtst_stcgn.json'),'--source-checkpoint',str(run/'warmup/stcgn_warmup.pth'),
                '--data-root',str(args.data_root),'--workers','2','--seed','1','--readout-lr-mult','1',
                '--gsd-config',str(args.settings/'raw_irrg.json'),'--baseline-config',str(run/'stcgn_resolved.json'),
                '--eval-period','1000','--output',str(run)]
    if args.experiment=='snd_author':  # author SND: source weights + warm banks, LR clock from 0
        return [sys.executable,'-u',str(args.workspace/entry),'snd','--protocol','siamseg_sfuda_b2_v1','--snd-init','source',
                '--config',str(args.settings/'recipe.json'),'--source-checkpoint',str(args.source),
                '--warm-checkpoint',str(args.warm/'last.pth'),'--data-root',str(args.data_root),'--workers','2','--seed','1',
                '--readout-lr-mult','1','--gsd-config',str(args.settings/'raw_irrg.json'),'--eval-period','1000','--output',str(run)]
    return [sys.executable,'-u',str(args.workspace/entry),'snd','--protocol','siamseg_sfuda_b2_v1',
            '--config',str(args.settings/'recipe.json'),'--source-checkpoint',str(args.source),
            '--warm-checkpoint',str(args.warm/'last.pth'),'--data-root',str(args.data_root),'--workers','2','--seed','1',
            '--readout-lr-mult','1','--gsd-config',str(args.settings/'raw_irrg.json'),'--eval-period','1000','--output',str(run)]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--experiment',choices=tuple(EXPERIMENTS),required=True)
    p.add_argument('--workspace',type=Path,required=True);p.add_argument('--settings',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--warm',type=Path,help='snd_baseline only: DT-ST warm stage directory (contains last.pth and banks)')
    p.add_argument('--warm-kind',choices=('dtst_null','snd_warm'),default='dtst_null')
    p.add_argument('--source-root',type=Path,help='stcgn only: data root with the labelled source_train split')
    p.add_argument('--output-root',type=Path,required=True);p.add_argument('--seconds',type=float,required=True)
    args=p.parse_args();run=args.output_root/args.experiment/'seed1';run.mkdir(parents=True,exist_ok=True)
    entry,stage_name=EXPERIMENTS[args.experiment];stage=run/stage_name
    state_path=run/'RUN_STATE.json'
    def state(status,**extra):
        write_json(state_path,dict(status=status,experiment=args.experiment,
            time_bangkok=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=7))).isoformat(),**extra))
        print(json.dumps(json.loads(state_path.read_text())),flush=True)
    try:
        if args.seconds<900:raise ValueError('Less than15min left; start a fresh batch session instead')
        import torch
        assert torch.cuda.is_available(),'Select a GPU accelerator'
        free,_=torch.cuda.mem_get_info();assert free>=12*1024**3,'Need12GiB free on one GPU; keep batch6'
        assert digest(args.source)==SOURCE_SHA
        if (args.experiment in NEEDS_WARM)!=(args.warm is not None):raise ValueError('--warm is required for the SND experiments only')
        warm=verify_warm(args.warm,args.warm_kind) if args.warm is not None else None
        train_source_sha=SOURCE_SHA
        if args.experiment=='stcgn':
            train_source_sha=prepare_stcgn(args,run)
        policy=json.loads((args.settings/'early_stopping_scout_2eval.json').read_text())
        budget=dict(name=BUDGET_NAME,phase=args.experiment,max_updates=CAP,original_policy=policy,user_requested=True,
                    applied_after_training_started=False,rising_cap_rule=RISING_CAP_RULE,**({'warm':warm} if warm else {}))
        validate_budget(budget,policy)
        for path,value in [(run/'early_stopping_policy.json',policy),(run/'pilot_budget.json',budget)]:
            if path.exists():assert json.loads(path.read_text())==value,'Resume control differs from original'
            else:write_json(path,value)
        if any((stage/name).exists() for name in ('complete.json','stopped_early.json')):
            state('already_finished');export_manifest(run);return
        if (stage/'session_pause.json').exists():
            pause=json.loads((stage/'session_pause.json').read_text())
            assert digest(stage/'resume.pth')==pause['resume_checkpoint_sha256']
            assert digest(stage/'last.pth')==pause['checkpoint_sha256']
            trim_uncommitted(run,stage_name,pause['iteration'])
            (stage/'session_pause.json').rename(stage/('session_pause_'+str(time.time_ns())+'.json'))
        token=uuid.uuid4().hex;identity='local:'+token
        env=dict(os.environ,DTST_LOCAL_RUN_ID=token,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4');env.pop('SLURM_JOB_ID',None)
        cmd=command(args,run)
        write_json(run/'KAGGLE_LAUNCH.json',dict(command=cmd,torch=torch.__version__,gpu=torch.cuda.get_device_name(0),batch=6,
            session_seconds=args.seconds,target_bands='IRRG',source_sha256=digest(args.source),warm=warm,slurm=False))
        state('running',optimizer_progress='Check '+stage_name+'/training_history.jsonl')
        deadline=time.monotonic()+args.seconds;paused=finished=False;pause_boundary=None
        child=subprocess.Popen(cmd,cwd=args.workspace,env=env)
        try:
            while child.poll() is None:
                curve=read_curve(stage/'evaluation_curve.jsonl')
                if (run/'protocol.json').exists():validate_budget(budget,policy,json.loads((run/'protocol.json').read_text()),train_source_sha)
                decision=scientific_stop(curve,policy,budget)
                if decision and stop_at_checkpoint(child.pid,run,stage_name,entry,identity,decision,policy):finished=True;break
                if time.monotonic()>=deadline and pause_boundary is None:pause_boundary=curve[-1]['iteration'] if curve else 0
                if pause_boundary is not None and curve and curve[-1]['iteration']>pause_boundary:
                    peak=max(curve,key=lambda row:row['metrics']['miou'])
                    pause=dict(iteration=curve[-1]['iteration'],peak_iteration=peak['iteration'],peak_miou=peak['metrics']['miou'],
                        reason='Kaggle session time budget; resume from full saved state; not convergence or numerical early stop')
                    if stop_at_checkpoint(child.pid,run,stage_name,entry,identity,pause,policy,stop_kind='session_pause'):
                        trim_uncommitted(run,stage_name,pause['iteration']);paused=True;break
                time.sleep(2)
            code=child.wait()
        finally:
            if child.poll() is None:child.terminate();child.wait()
        if paused:state('paused_for_next_session',retained_iteration=pause['iteration'],target_labels_used_for_stopping=False)
        elif code==0 or finished:
            assert (stage/'complete.json').exists() or (stage/'stopped_early.json').exists(),'No verified completion marker'
            write_report(run,stage_name)
            state('finished')
        else:raise RuntimeError('Trainer exited '+str(code)+'; inspect error.json and notebook output')
    except Exception as error:
        state('failed',error=type(error).__name__+': '+str(error));raise
    finally:export_manifest(run)


CLASSES=('impervious_surface','building','low_vegetation','tree','car','clutter')


def write_report(run,stage_name):
    """Peak (target-label selected, declared) and actual last, with mIoU6/mIoU5 and per-class IoU from the same checkpoint."""
    curve=read_curve(run/stage_name/'evaluation_curve.jsonl')
    def row(point):
        iou=point['metrics']['iou'] if 'iou' in point['metrics'] else point['metrics'].get('per_class_iou')
        out=dict(iteration=point['iteration'],miou6=100*point['metrics']['miou'])
        if iou is not None:
            values=[iou[c] if isinstance(iou,dict) else iou[i] for i,c in enumerate(CLASSES)]
            out.update({c+'_iou':100*v for c,v in zip(CLASSES,values)},miou5=100*sum(values[:5])/5)
        return out
    peak=max(curve,key=lambda point:point['metrics']['miou'])
    report=dict(stage=stage_name,iteration0=row(curve[0]),peak=row(peak),last=row(curve[-1]),
                curve=[row(point) for point in curve],
                note=('Iteration 0 is the SND-stage start (warm weights for snd_baseline, source40k for snd_author). ' if stage_name=='snd' else 'Iteration 0 is source40k. ')+
                     'Peak selected with target-eval labels (declared).')
    (run/'report').mkdir(exist_ok=True);write_json(run/'report/peak_and_endpoint.json',report)
    lines=['# Baseline: peak and actual last','','| | iteration | mIoU6 | mIoU5 |','|---|---:|---:|---:|']
    for name in ('iteration0','peak','last'):
        r=report[name];lines.append(f"| {name} | {r['iteration']} | {r['miou6']:.4f} | {r.get('miou5',float('nan')):.4f} |")
    (run/'report/PEAK_AND_ENDPOINT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
