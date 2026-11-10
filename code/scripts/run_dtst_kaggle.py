"""One full-protocol experiment per Kaggle batch session, with checkpoint-safe time pauses."""
import argparse,datetime,json,os,shutil,subprocess,sys,time,uuid
from pathlib import Path
from scripts.dtst_run_control import read_curve,write_json,stop_decision,stop_at_checkpoint,digest
from scripts.dtst_pilot_budget import cap_decision,cap_extension_active,validate_budget

PHASES=('dino_recovery','dino_recovery_open','ema_recovery','dino_recovery_no_negative','bank_coupled','no_negative','raw_readout1_scout',
        'pheno_c0','pheno_c0_c1','pheno_c1','pheno_null','pheno_c0_protect','pheno_delta_replay','pheno_c0_c1_randdir','pheno_c0multi','pheno_c0multi_c1plaus','pheno_c0_c1plaus')
PHENO=('pheno_c0','pheno_c0_c1','pheno_c1','pheno_null','pheno_c0_protect','pheno_delta_replay','pheno_c0_c1_randdir','pheno_c0multi','pheno_c0multi_c1plaus','pheno_c0_c1plaus')

def scientific_stop(curve,policy,budget):
    early=stop_decision(curve,policy);cap=cap_decision(curve,budget,policy)
    return cap if cap_extension_active(curve,budget,policy) else early or cap

def trim_uncommitted(run,iteration):
    archive=run/'session_history_archive';archive.mkdir(exist_ok=True)
    for name in ('training_history.jsonl','teacher_feedback.jsonl'):
        path=run/'warm'/name
        if not path.exists():continue
        records=read_curve(path)
        if any(row.get('iteration',0)>iteration for row in records):
            shutil.copy2(path,archive/(str(time.time_ns())+'_'+name))
            path.write_text(''.join(json.dumps(row)+'\n' for row in records if row.get('iteration',0)<=iteration))

def export_manifest(run):
    files={str(p.relative_to(run)):dict(bytes=p.stat().st_size,sha256=digest(p))
           for p in run.rglob('*') if p.is_file() and not p.name.endswith('.tmp') and p.name!='SESSION_MANIFEST.json'}
    write_json(run/'SESSION_MANIFEST.json',dict(files=files,experiment=run.parent.name))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--experiment',choices=PHASES,required=True)
    p.add_argument('--workspace',type=Path,required=True);p.add_argument('--settings',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True);p.add_argument('--seconds',type=float,required=True)
    p.add_argument('--target-bands',choices=('IRRG','RGB'),default='IRRG')
    args=p.parse_args();run=args.output_root/args.experiment/'seed1';run.mkdir(parents=True,exist_ok=True)
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
        assert digest(args.source)=='10fbe026b5f2dcef331a99d97bab551f7cb0a2cf95feac2625b82ca3e3449d31'
        policy=json.loads((args.settings/'early_stopping_scout_2eval.json').read_text())
        budget=dict(name='dtst_scout_budget_20261015_v1' if args.experiment=='raw_readout1_scout' else
            'dtst_pheno_budget_20261031_v1' if args.experiment in PHENO else 'dtst_recovery_budget_20261016_v1',
            phase=args.experiment,max_updates=9000,original_policy=policy,user_requested=True,
            applied_after_training_started=False,rising_cap_rule='new_peak_at_cap_then_first_strict_decline_v1')
        for path,value in [(run/'early_stopping_policy.json',policy),(run/'pilot_budget.json',budget)]:
            if path.exists():assert json.loads(path.read_text())==value,'Resume control differs from original'
            else:write_json(path,value)
        stage=run/'warm'
        if any((stage/name).exists() for name in ('complete.json','stopped_early.json')):
            state('already_finished');export_manifest(run);return
        if (stage/'session_pause.json').exists():
            pause=json.loads((stage/'session_pause.json').read_text())
            assert digest(stage/'resume.pth')==pause['resume_checkpoint_sha256']
            assert digest(stage/'last.pth')==pause['checkpoint_sha256']
            trim_uncommitted(run,pause['iteration'])
            (stage/'session_pause.json').rename(stage/('session_pause_'+str(time.time_ns())+'.json'))
        token=uuid.uuid4().hex;identity='local:'+token
        env=dict(os.environ,DTST_LOCAL_RUN_ID=token,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        env.pop('SLURM_JOB_ID',None)
        common=[sys.executable,'-u',str(args.workspace/'train_DTST_remote.py'),'--config',str(args.settings/'recipe.json'),
            '--dtst-config',str(args.settings/('dtst_no_negative.json' if args.experiment in ('dino_recovery_no_negative','no_negative') else 'dtst.json')),
            '--source-checkpoint',str(args.source),'--data-root',str(args.data_root),'--workers','2','--seed','1',
            '--readout-lr-mult','1','--gsd-config',str(args.settings/('source_bank.json' if args.experiment=='bank_coupled' else 'raw_rgb.json' if args.target_bands=='RGB' else 'raw_irrg.json')),
            '--eval-period','1000','--output',str(run)]
        if args.experiment=='bank_coupled':common+=['--geometry-config',str(args.settings/'coupled_scale.json')]
        elif args.experiment in PHENO:common+=['--pheno-config',str(args.settings/(args.experiment+'.json'))]
        elif args.experiment not in ('no_negative','raw_readout1_scout'):common+=['--recovery-config',str(args.settings/(args.experiment+'.json'))]
        write_json(run/'KAGGLE_LAUNCH.json',dict(command=common,torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
            batch=6,session_seconds=args.seconds,target_bands=args.target_bands,source_sha256=digest(args.source),slurm=False))
        state('running',optimizer_progress='Check warm/training_history.jsonl')
        deadline=time.monotonic()+args.seconds;paused=False;finished=False;pause_boundary=None
        child=subprocess.Popen(common,cwd=args.workspace,env=env)
        try:
            while child.poll() is None:
                curve=read_curve(stage/'evaluation_curve.jsonl')
                if (run/'protocol.json').exists():validate_budget(budget,policy,json.loads((run/'protocol.json').read_text()))
                decision=scientific_stop(curve,policy,budget)
                if decision and stop_at_checkpoint(child.pid,run,identity,decision,policy):finished=True;break
                if time.monotonic()>=deadline and pause_boundary is None:
                    pause_boundary=curve[-1]['iteration'] if curve else 0
                if pause_boundary is not None and curve and curve[-1]['iteration']>pause_boundary:
                    peak=max(curve,key=lambda row:row['metrics']['miou'])
                    pause=dict(iteration=curve[-1]['iteration'],peak_iteration=peak['iteration'],peak_miou=peak['metrics']['miou'],
                        reason='Kaggle session time budget; resume from full saved state; not convergence or numerical early stop')
                    if stop_at_checkpoint(child.pid,run,identity,pause,policy,stop_kind='session_pause'):
                        trim_uncommitted(run,pause['iteration']);paused=True;break
                time.sleep(2)
            code=child.wait()
        finally:
            if child.poll() is None:child.terminate();child.wait()
        if paused:state('paused_for_next_session',retained_iteration=pause['iteration'],target_labels_used_for_stopping=False)
        elif code==0 or finished:
            assert (stage/'complete.json').exists() or (stage/'stopped_early.json').exists(),'No verified completion marker'
            subprocess.run([sys.executable,str(args.workspace/'scripts/summarize_dtst_results.py'),str(run),
                '--output',str(run/'report'),'--skip-continuation'],cwd=args.workspace,env=env,check=True)
            state('finished')
        else:raise RuntimeError('Trainer exited '+str(code)+'; inspect error.json and notebook output')
    except Exception as error:
        state('failed',error=type(error).__name__+': '+str(error));raise
    finally:export_manifest(run)

if __name__=='__main__':main()
