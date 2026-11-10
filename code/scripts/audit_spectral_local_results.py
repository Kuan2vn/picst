"""Independent NumPy audit of the exported paired spectral pilot results."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/training_free_subspace_20261008'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quality(cm):
    cm=np.asarray(cm); tp=cm.diagonal(axis1=-2,axis2=-1)
    union=cm.sum(-1)+cm.sum(-2)-tp
    with np.errstate(divide='ignore',invalid='ignore'):
        iou=np.where(union>0,tp/union,np.nan)
    return np.nanmean(iou,axis=-1),iou


def close(a,b):
    np.testing.assert_allclose(a,b,rtol=0,atol=1e-10)


def main():
    manifest=read(REPORT/'SPECTRAL_LOCAL_ARTIFACTS.json')
    for name,expected in manifest['portable_sha256'].items():
        path=(REPORT/name).resolve()
        assert REPORT.resolve() in path.parents and sha(path)==expected,name
    remote=read(REPORT/'SPECTRAL_POOLED_HPC_AUDIT.json')
    assert remote['passed']
    results={}; summary={}; draws_by_channel={}
    for channel in ('rgb','irrg'):
        folder=REPORT/('spectral_'+channel); parent=folder/'parent_85043'
        protocol=read(folder/'PROTOCOL.json'); old=read(parent/'PROTOCOL.json')
        m=read(folder/'METRICS_screen.json'); old_m=read(parent/'METRICS_screen.json')
        g=read(folder/'GEOMETRY.json'); a=read(folder/'COLLECTION.json'); gate=read(folder/'ELIGIBILITY.json')
        for p,d in ((folder,protocol),(parent,old)):
            assert hashlib.sha256(json.dumps(d['locked'],sort_keys=True,allow_nan=False).encode()).hexdigest()==d['signature']
            assert sha(p/'runner_snapshot.py')==d['locked']['code_sha256']['run_training_free_spectral_local.py']
            assert sha(p/'helper_snapshot.py')==d['locked']['code_sha256']['scripts/training_free_local_subspace.py']
            assert sha(p/'PLAN_LOCKED.md')==d['locked']['plan_sha256']
            assert read(p/'COMPLETE.json')['signature']==d['signature']
        assert protocol['locked']['repaired_from_signature']==old['signature']
        assert m['signature']==protocol['signature'] and len(m['images'])==112
        assert m['images']==protocol['locked']['roles']['screen']==old_m['images']
        assert m['state_unchanged'] and m['optimizer_updates']==0 and m['target_train_gt_opened']==0
        assert a['state_unchanged'] and a['target_train_gt_opened']==g['target_train_gt_opened']==0
        assert a['unlabeled_native_parity_checks']==2
        assert m['bank_sha256']==a['bank_sha256']==old_m['bank_sha256']
        assert m['scorers_sha256']==g['scorers_sha256']
        assert sha(folder/'COLLECTION.json')==sha(parent/'COLLECTION.json')
        assert read(folder/'COMPLETE.json')['full_evaluated'] is False
        role_tiles={r:{'_'.join(Path(n).stem.split('_')[:2]) for n in names} for r,names in protocol['locked']['roles'].items()}
        assert [len(role_tiles[r]) for r in ('fit','calibration','screen')]==[18,6,14]
        for r,s in (('fit','calibration'),('fit','screen'),('calibration','screen')):
            assert not role_tiles[r]&role_tiles[s]
        matrices={r:np.asarray(v['confusion_matrix'],dtype=np.int64) for r,v in m['routes'].items()}
        ids=sorted(m['group_confusion']); n=len(ids); assert n==14
        sample=np.random.default_rng(20261010).integers(n,size=(2000,n))
        weights=np.asarray([np.bincount(v,minlength=n) for v in sample])
        boot={}; route_summary={}
        for route,cm in matrices.items():
            assert cm.shape==(6,6) and (cm>=0).all()
            close(cm.sum(1),matrices['B1'].sum(1))
            q,iou=quality(cm); close(q,m['routes'][route]['miou']); close(iou,m['routes'][route]['iou'])
            assert int(cm.sum())==m['routes'][route]['valid_pixels']
            groups=np.asarray([m['group_confusion'][t][route] for t in ids])
            close(groups.sum(0),cm)
            boot[route]=quality((weights@groups.reshape(n,-1)).reshape(2000,6,6))[0]
            zsum=np.zeros((6,6),dtype=np.int64)
            for zone in ('boundary','interior'):
                z=m['zones'][zone][route]; zcm=np.asarray(z['confusion_matrix']); zsum+=zcm
                assert int(zcm.sum())==z['valid_pixels']
                if zcm.sum(): close(quality(zcm)[0],z['miou']); close(quality(zcm)[1],z['iou'])
                else: assert z['miou'] is None and z['iou']==[None]*6
            close(zsum,cm)
            if route!='NL': assert m['routes'][route]==old_m['routes'][route],route
            if route in m['changes_vs_B1']:
                change=m['changes_vs_B1'][route]; net=change['wrong_to_correct']-change['correct_to_wrong']
                assert net==int(np.trace(cm)-np.trace(matrices['B1']))
                assert all(v>=0 for v in change.values())
            else: net=None
            tile_delta={t:float(100*(quality(v[route])[0]-quality(v['B1'])[0])) for t,v in m['group_confusion'].items()}
            route_summary[route]={'miou_percent':float(100*q),'iou_percent':(100*iou).tolist(),
                'delta_B1_pp':float(100*(q-quality(matrices['B1'])[0])),
                'net_correct_pixels_vs_B1':net,'positive_tiles_vs_B1':sum(v>0 for v in tile_delta.values()),
                'tile_delta_vs_B1_pp':tile_delta}
        for pair,expected in m['bootstrap_ci95_pp'].items():
            left,right=pair.split('-minus-')
            close(100*np.quantile(boot[left]-boot[right],[.025,.975]),expected)
        for candidate,controls in [('SL',('B1','UL','FL','PL','RL')),('NL',('B1','UL','FL','PN','RN'))]:
            comparator=max(controls,key=lambda r:m['routes'][r]['miou'])
            delta=100*(m['routes'][candidate]['miou']-m['routes'][comparator]['miou'])
            net=route_summary[candidate]['net_correct_pixels_vs_B1']
            worst=float(np.min(100*(quality(matrices[candidate])[1]-quality(matrices['B1'])[1])))
            positive=sum(quality(v[candidate])[0]>quality(v[comparator])[0] for v in m['group_confusion'].values())
            checks={'gain_at_least_0_15pp':delta>=.15,'net_vs_B1_positive':net>0,
                    'worst_class_drop_within_0_5pp':worst>=-.5,'positive_in_at_least_two_tiles':positive>=2}
            assert gate[candidate]['checks']==checks and gate[candidate]['passed']==all(checks.values())
            assert gate[candidate]['comparator']==comparator and gate[candidate]['positive_tiles']==positive
            close(delta,gate[candidate]['delta_pp']); close(worst,gate[candidate]['worst_class_delta_pp'])
        if channel=='rgb':
            old_graph=read(REPORT/'graph_vh_sourceval/METRICS_screen.json')
            assert m['routes']['B0']['confusion_matrix']==old_graph['routes']['B0']['confusion_matrix']
        results[channel]={'passed':True,'signature_snapshot_verified':True,'confusion_group_zone_correction_audit':True,
            'all_bootstrap_ci_recomputed':True,'eligibility_recomputed':True,'nine_unchanged_routes_exact_replay':True,
            'geographic_role_tiles':{r:len(v) for r,v in role_tiles.items()},'boundary_valid_pixels':m['zones']['boundary']['B1']['valid_pixels']}
        drift=np.asarray([v['source_argmax_change_fraction'] for v in a['perturbation_audit']])
        summary[channel]={'routes':route_summary,'ci95_pp':m['bootstrap_ci95_pp'],
            'B1_minus_B0_ci95_pp':(100*np.quantile(boot['B1']-boot['B0'],[.025,.975])).tolist(),
            'supported_classes':g['supported_classes'],'class_counts':a['class_counts'],
            'semantic_rank':g['semantic_rank'],'nuisance':g['nuisance'],'eligibility':gate,
            'perturbation_argmax_drift_mean':float(drift.mean()),'perturbation_argmax_drift_max':float(drift.max()),
            'collection_seconds':a['seconds'],'collection_forwards':a['forwards'],
            'screen_seconds':m['seconds'],'screen_encoder_seconds':m['forward_seconds'],'peak_rss_mib':m['peak_rss_mib']}
        draws_by_channel[channel]=boot
    rgb=read(REPORT/'spectral_rgb/METRICS_screen.json'); irrg=read(REPORT/'spectral_irrg/METRICS_screen.json')
    assert rgb['images']==irrg['images']
    close(np.asarray(rgb['routes']['B0']['confusion_matrix']).sum(1),np.asarray(irrg['routes']['B0']['confusion_matrix']).sum(1))
    results['paired']={'passed':True,'identical_coordinates_truth_counts':True,'images':112,'tiles':14}
    summary['paired']={
        'IRRG_minus_RGB_B0_pp':summary['irrg']['routes']['B0']['miou_percent']-summary['rgb']['routes']['B0']['miou_percent'],
        'IRRG_minus_RGB_B0_ci95_pp':(100*np.quantile(draws_by_channel['irrg']['B0']-draws_by_channel['rgb']['B0'],[.025,.975])).tolist(),
        'B1_gain_IRRG_minus_RGB_pp':summary['irrg']['routes']['B1']['delta_B1_pp']-summary['irrg']['routes']['B0']['delta_B1_pp']-summary['rgb']['routes']['B1']['delta_B1_pp']+summary['rgb']['routes']['B0']['delta_B1_pp'],
        'B1_gain_IRRG_minus_RGB_ci95_pp':(100*np.quantile((draws_by_channel['irrg']['B1']-draws_by_channel['irrg']['B0'])-(draws_by_channel['rgb']['B1']-draws_by_channel['rgb']['B0']),[.025,.975])).tolist()}
    for filename,value in [('SPECTRAL_LOCAL_CPU_VALIDATION.json',results),('SPECTRAL_LOCAL_CPU_SUMMARY.json',summary)]:
        (REPORT/filename).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('PASS: portable hashes, protocol snapshots, nine-route replay, CMs/zones/groups/corrections, bootstrap and paired data')
    for channel in ('rgb','irrg'):
        print(channel,{r:round(v['miou_percent'],6) for r,v in summary[channel]['routes'].items()})


if __name__=='__main__': main()
