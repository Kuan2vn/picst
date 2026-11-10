"""Check gain arithmetic and final-versus-peak reporting on a tiny fixture."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.report_siamseg_sfuda_b2 import sha256, summarize


def test_report():
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root/'source/seed1/best_target_val.pth'
        source.parent.mkdir(parents=True)
        source.write_bytes(b'fixed source')
        run = root/'snd/seed1'
        run.mkdir(parents=True)
        (run/'protocol.json').write_text(json.dumps({'protocol': 'siamseg_sfuda_b2_v1',
            'source_checkpoint_sha256': sha256(source), 'seed': 1,
            'recipe': {'warm': {'steps': 20}, 'snd': {'steps': 12}}}))
        names = ['impervious_surface', 'building', 'low_vegetation', 'tree', 'car', 'clutter']
        metric = lambda value: {'miou': value, 'valid_pixels': 100,
                                'iou_by_class': dict.fromkeys(names, value)}
        results = {name: {'metrics': metric(value)} for name, value in
                   [('source', .4), ('warm', .45), ('snd', .47), ('meta-off', .44)]}
        (run/'result_portable.json').write_text(json.dumps({'results': results}))
        for name, points in {'warm': [(20, .45)], 'snd': [(26, .5), (32, .47)],
                             'meta-off': [(26, .46), (32, .44)]}.items():
            stage = run/name
            stage.mkdir()
            (stage/'evaluation_curve.jsonl').write_text(''.join(json.dumps({
                'iteration': iteration-20 if name != 'warm' else iteration,
                'total_iteration': iteration, 'stage': name, 'metrics': metric(value)})+'\n'
                for iteration, value in points))
        entry = summarize(root)['seeds'][0]
        assert round(entry['gain_final_pp'], 6) == 7
        assert round(entry['gain_vs_meta_off_pp'], 6) == 3
        assert round(entry['gain_peak_vs_meta_off_peak_pp'], 6) == 4
        assert round(entry['peak_to_final_drop_pp'], 6) == 3
        assert entry['peak_pipeline_iteration'] == 26
        (run/'warm_selection.json').write_text(json.dumps({'iteration': 5}))
        (run/'warm/evaluation_curve.jsonl').write_text(''.join(json.dumps({
            'iteration': iteration, 'total_iteration': iteration, 'stage': 'warm',
            'metrics': metric(value)})+'\n' for iteration, value in [(5, .48), (20, .45)]))
        for name, values in [('snd', [.5, .47]), ('meta-off', [.46, .44])]:
            (run/name/'evaluation_curve.jsonl').write_text(''.join(json.dumps({
                'iteration': iteration, 'total_iteration': iteration+20, 'stage': name,
                'metrics': metric(value)})+'\n'
                for iteration, value in zip((6, 12), values)))
        selected_entry = summarize(root)['seeds'][0]
        assert selected_entry['warm_selected_iteration'] == 5
        assert round(selected_entry['gain_vs_warm_selected_pp'], 6) == -1
        assert selected_entry['peak_meta_iteration'] == 26
        selected = source.with_name('selected_target_val_step003000.pth')
        selected.write_bytes(source.read_bytes())
        source.write_bytes(b'later best during ongoing source training')
        assert summarize(root)['source_checkpoint_sha256'] == sha256(selected)


if __name__ == '__main__':
    test_report()
    print('Portable gain arithmetic OK')
