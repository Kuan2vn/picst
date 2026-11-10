"""Summarize recorded learning signals, preserving the native run reports."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, banks, summary = [], [], {}
    for path in args.reports:
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None
            read = lambda name: [json.loads(line) for line in archive.read(name).decode().splitlines() if line.strip()]
            protocol = json.loads(archive.read('protocol.json'))
            study = protocol['gsd_study']
            arm = study['arm']
            if study.get('loss_mode') in ('blend', 'soft_js'):
                arm += '_' + study['loss_mode']
            history = read('warm/training_history.jsonl')
            feedback = read('warm/teacher_feedback.jsonl')
            candidates = read('warm/candidates.jsonl')
            curves = read('warm/evaluation_curve.jsonl')
        assert history and curves
        actual_updates = history[-1]['iteration']
        assert all(a['iteration'] < b['iteration'] for a, b in zip(history, history[1:]))
        arm_rows = []
        for low, high in [(0, 3000), (3000, 6000), (6000, 9000), (9000, 12000),
                          (12000, 15000), (15000, 18000), (18000, 20000)]:
            if low >= actual_updates:
                continue
            planned_high = high
            high = min(high, actual_updates)
            records = [r for r in history if low < r['iteration'] <= high]
            events = [r for r in feedback if low < r['iteration'] <= high]
            updates = [r for r in events if r['update']]
            total_pixels = sum(sum(r['pseudo_class_pixels']) for r in records)
            row = dict(arm=arm, from_update=low+1, to_update=high, logged_records=len(records),
                       loss_positive_mean=statistics.mean(r['loss_positive'] for r in records),
                       loss_negative_mean=statistics.mean(r['loss_negative'] for r in records),
                       car_pseudo_mass=sum(r['pseudo_class_pixels'][4] for r in records)/total_pixels,
                       tree_pseudo_mass=sum(r['pseudo_class_pixels'][3] for r in records)/total_pixels,
                       car_moving_class_prior_mean=statistics.mean(r['running_conf'][4] for r in records),
                       teacher_queries=len(events), teacher_updates=len(updates),
                       forced_updates=sum(r['forced'] for r in updates),
                       mean_update_interval=statistics.mean(r['interval'] for r in updates) if updates else None,
                       js_mean=statistics.mean(r['gsd_js_mean'] for r in records) if arm.startswith(('gsd','generic')) else None,
                       recipient_coverage=statistics.mean(r['gsd_actual_recipient_coverage'] for r in records) if arm.startswith(('gsd','generic')) else None,
                       eligible_fraction=statistics.mean(r['gsd_eligible_fraction'] for r in records) if arm.startswith(('gsd','generic')) else None,
                       complete_planned_window=(high == planned_high))
            rows.append(row); arm_rows.append(row)
        for record in candidates:
            banks.append(dict(arm=arm, iteration=record['iteration'],
                              threshold_kind='confidence' if record['initial'] else 'temporal_image_class_iou',
                              car_pseudo_fraction=record['pseudo_fraction'][4],
                              tree_pseudo_fraction=record['pseudo_fraction'][3],
                              car_threshold=record['thresholds'][4], tree_threshold=record['thresholds'][3]))
        endpoint, peak = curves[-1], max(curves, key=lambda p: p['metrics']['miou'])
        matrix = endpoint['metrics']['confusion_matrix']
        car_tp = matrix[4][4]
        car_fp = sum(row[4] for row in matrix)-car_tp
        summary[arm] = dict(peak=peak, endpoint=endpoint, windows=arm_rows,
                            car_false_positive_pixels=car_fp, car_true_positive_pixels=car_tp,
                            last_record=history[-1], candidates=candidates, evaluation_curve=curves,
                            actual_logged_updates=actual_updates,
                            records_are_sampled='step1 and every20 updates; window masses use recorded samples',
                            running_conf_definition='moving mean posterior/class prior, not conditional confidence')
        summary[arm]['candidate_threshold_definition'] = 'confidence at initialization; temporal image/class IoU at later refreshes'
    for name, records in [('learning_windows.csv', rows), ('candidate_bank_signals.csv', banks)]:
        with (args.output/name).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
    (args.output/'learning_signals.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'windows': rows, 'candidate_banks': banks}, indent=2))


if __name__ == '__main__':
    main()
