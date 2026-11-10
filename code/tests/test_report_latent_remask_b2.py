"""Portable-only paired report, including an intentionally early-stopped B0."""
import json
from pathlib import Path
import sys
import tempfile
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.report_latent_remask_b2 import summarize


def write_zip(path, files):
    with zipfile.ZipFile(path, 'w') as archive:
        for name, value in files.items():
            archive.writestr(name, json.dumps(value) if not isinstance(value, str) else value)


def point(step, miou):
    names = ['a', 'b']
    return {'iteration': step, 'metrics': {'miou': miou, 'miou5': miou, 'class_order': names,
            'iou_by_class': {'a': miou, 'b': miou}, 'prediction_fraction': [.5, .5]}}


def run():
    with tempfile.TemporaryDirectory() as folder:
        baseline, auxiliary = Path(folder)/'b.zip', Path(folder)/'a.zip'
        base_protocol = {'seed': 1, 'source_checkpoint_sha256': 'source', 'protocol_signature': 'warm',
                         'recipe': {'snd': {'steps': 12000}}, 'manifests': {'train': 'fixed'}}
        aux_protocol = {'seed': 1, 'source_checkpoint_sha256': 'source', 'method': 'snd_srmp_aux',
                        'warm_protocol_signature': 'warm', 'warm_checkpoint_sha256': 'selected',
                        'auxiliary': {'objective': 'relation', 'weight': .1}, 'recipe': base_protocol['recipe'],
                        'manifests': base_protocol['manifests']}
        write_zip(baseline, {'protocol.json': base_protocol,
                             'result_portable.json': {'results': {'warm': {'metrics': {'miou': .5}}}},
                             'warm_selection.json': {'checkpoint_sha256': 'selected'},
                             'snd/evaluation_curve.jsonl': '\n'.join(json.dumps(point(s, m)) for s, m in
                                                                   [(0, .5), (500, .48), (5000, .45)])})
        write_zip(auxiliary, {'protocol.json': aux_protocol,
                              'result_portable.json': {'warm_selection': {'checkpoint_sha256': 'selected'},
                                                       'warm_final': {'metrics': {'miou': .5}},
                                                       'continuation_complete': True},
                              'snd/evaluation_curve.jsonl': '\n'.join(json.dumps(point(s, m)) for s, m in
                                                                    [(0, .5), (500, .49), (5000, .47), (12000, .46)])})
        row = summarize(baseline, [auxiliary])['runs'][0]
        assert row['paired_last_iteration'] == 5000
        assert abs(row['paired_last_delta_pp'] - 2) < 1e-9
        assert abs(row['paired_last_miou5_delta_pp'] - 2) < 1e-9
        assert row['aux_weight'] == .1 and row['aux_peak_miou5'] == .5
        assert row['aux_peak_miou5_iteration'] == 0
        assert row['final_delta_pp'] is None and not row['full_budget_paired']
        assert row['baseline_status'] == 'early_stopped'
        assert row['comparison_scope'] == 'same_selected_warm'
        aux_protocol.update(aux_start='warm', warm_checkpoint_sha256='own_warm',
                            warm_protocol_signature='own_protocol')
        write_zip(auxiliary, {'protocol.json': aux_protocol,
                              'result_portable.json': {'warm_selection': {'checkpoint_sha256': 'own_warm'},
                                                       'warm_protocol_signature': 'own_protocol',
                                                       'warm_final': {'metrics': {'miou': .52}},
                                                       'continuation_complete': True},
                              'snd/evaluation_curve.jsonl': '\n'.join(json.dumps(point(s, m)) for s, m in
                                                                    [(0, .52), (500, .49), (5000, .47), (12000, .46)])})
        row = summarize(baseline, [auxiliary])['runs'][0]
        assert row['comparison_scope'] == 'different_warm_full_pipeline'
        assert abs(row['warm_final_delta_pp'] - 2) < 1e-9
        write_zip(auxiliary, {'protocol.json': aux_protocol,
                              'result_portable.json': {'warm_selection': {'checkpoint_sha256': 'own_warm'},
                                                       'warm_protocol_signature': 'own_protocol',
                                                       'warm_final': {'metrics': {'miou': .52}},
                                                       'warm_stopped_early': True, 'warm_actual_updates': 8500,
                                                       'warm_planned_updates': 20000,
                                                       'continuation_complete': True},
                              'snd/evaluation_curve.jsonl': '\n'.join(json.dumps(point(s, m)) for s, m in
                                                                    [(0, .52), (500, .49), (5000, .47), (12000, .46)])})
        row = summarize(baseline, [auxiliary])['runs'][0]
        assert row['comparison_scope'] == 'different_warm_early_stopped_pipeline'
        assert row['warm_actual_updates'] == 8500 and not row['full_budget_paired']
        aux_protocol['warm_checkpoint_sha256'] = 'wrong'
        write_zip(auxiliary, {'protocol.json': aux_protocol,
                              'result_portable.json': {'warm_selection': {'checkpoint_sha256': 'own_warm'},
                                                       'warm_protocol_signature': 'own_protocol',
                                                       'warm_final': {'metrics': {'miou': .52}}},
                              'snd/evaluation_curve.jsonl': json.dumps(point(500, .49))})
        try:
            summarize(baseline, [auxiliary])
        except ValueError:
            pass
        else:
            raise AssertionError('Mismatched warm checkpoint was accepted')


if __name__ == '__main__':
    run()
    print('Portable paired report OK')
