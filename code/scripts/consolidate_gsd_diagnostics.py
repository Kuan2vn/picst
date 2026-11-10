"""Consolidate the five completed frozen-checkpoint probes; never launch inference."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    stages = ['source', 'raw_peak', 'gsd_peak', 'gsd_final', 'gsd_peak_source_ref']
    summaries, metadata = {}, {}
    for stage in stages:
        summaries[stage] = json.loads((args.root/stage/'summary.json').read_text())
        metadata[stage] = json.loads((args.root/stage/'metadata.json').read_text())
    first = metadata['source']
    for stage, meta in metadata.items():
        # Self-check actual cohort/provenance before comparing aggregate numbers.
        assert meta['diagnostic_names'] == first['diagnostic_names'], stage
        assert meta['view_manifest_sha256'] == first['view_manifest_sha256'], stage
        assert meta['target_train_gt_opened'] == 0 and meta['target_eval_gt_opened'] == 128, stage
        assert summaries[stage]['valid_labeled_pixels'] == summaries['source']['valid_labeled_pixels'], stage
    assert metadata['gsd_peak']['checkpoint_sha256'] == metadata['gsd_peak_source_ref']['checkpoint_sha256']
    assert summaries['gsd_peak']['native_metrics_on_common_support'] == summaries['gsd_peak_source_ref']['native_metrics_on_common_support']
    assert summaries['gsd_peak']['per_image_q0p5']['confidence'] == summaries['gsd_peak_source_ref']['per_image_q0p5']['confidence']
    assert metadata['gsd_peak_source_ref']['reference_checkpoint_sha256'] == first['checkpoint_sha256']
    rows, conditional = [], []
    for stage, summary in summaries.items():
        matrix = summary['native_metrics_on_common_support']['confusion_matrix']
        car_fp = sum(matrix[i][4] for i in range(6) if i != 4)
        for selector in ['confidence', 'ratio_1p8', 'ratio_2']:
            image = summary['per_image_q0p5'][selector]
            points = summary['confidence'] if selector == 'confidence' else summary['views'][selector]['lambda_sweep']['1.0']
            pooled = next(p for p in points if p['coverage'] == .5)
            rows.append(dict(stage=stage, selector=selector, coverage=.5,
                             native_error_percent=100*summary['native_error'],
                             per_image_risk_percent=100*image['micro_risk'],
                             per_image_car_precision_percent=100*image['precision_by_predicted_class'][4],
                             per_image_tree_precision_percent=100*image['precision_by_predicted_class'][3],
                             pooled_risk_percent=100*pooled['micro_risk'],
                             pooled_car_precision_percent=100*(1-pooled['risk_by_predicted_class']['4']),
                             per_image_selected_gt_valid=sum(image['selected_gt_valid'])))
            if selector == 'confidence':
                continue
            view = summary['views'][selector]
            point = next(p for p in view['conditional_risk'] if p['class'] == 4 and p['confidence_bin'][0] == .95)
            conditional.append(dict(stage=stage, selector=selector,
                                    car_confidence_min=.95, pixels=point['count'],
                                    low_js_car_error_percent=100*point['low_js_risk'],
                                    high_js_car_error_percent=100*point['high_js_risk'],
                                    both_wrong_same_car=view['same_wrong_by_native_class'][4],
                                    native_car_false_positives=car_fp,
                                    shared_wrong_car_fraction=view['same_wrong_by_native_class'][4]/car_fp))
    args.output.mkdir(parents=True, exist_ok=True)
    for name, records in [('selector_comparison.csv', rows), ('car_conditional_risk.csv', conditional)]:
        with (args.output/name).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
    payload = dict(cohort_images=128, cohort_valid_pixels=summaries['source']['valid_labeled_pixels'],
                   views=first['views'], metadata=metadata, selectors=rows, car_conditional=conditional,
                   limitations=['Native predictions on a predeclared evaluation subset; not full training objective.',
                                'Per-image ranking matches batch1 grouping but omits weak augmentation, confidence filtering and TCR.',
                                'Pooled ranking is over all128 images; it is not batch6 ranking.',
                                'GT-ignore pixels are removed after per-image ranking, so reported labeled coverage can differ slightly.',
                                'No uncertainty intervals or additional seeds in this diagnostic.'])
    (args.output/'selector_comparison.json').write_text(json.dumps(payload, indent=2)+'\n', encoding='utf-8')
    print(f'Verified {len(stages)} probes on128 identical images; wrote {len(rows)} selector comparisons.')


if __name__ == '__main__':
    main()
