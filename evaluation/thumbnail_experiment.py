"""Portable reproduction of the supplied preliminary thumbnail SVM experiment.

This preserves its 240x300 JPEG thumbnails, fixed C=1/gamma=scale and five
shared splits. It is separate from the report's full-resolution grid search.
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data_preprocessing.prepare_dataset import RESAMPLE, rgb_image, write_csv
from hsv_lbp_svm.svm_baseline import extract_features, load_shared_splits


def run(dataset, audit, output, seeds):
    dataset, audit, output = dataset.resolve(), audit.resolve(), output.resolve()
    if not output.is_relative_to(dataset):
        raise ValueError('Thumbnail experiment outputs must remain under the dataset')
    if output.exists():
        raise ValueError('Choose a fresh thumbnail output directory')
    # The shared loader checks every manifest and source image before fitting.
    splits = {seed: load_shared_splits(dataset, audit, seed)[0] for seed in seeds}
    output.mkdir(parents=True)
    thumbnails = output / 'thumbnails'
    thumbnails.mkdir()
    inventory = {row['image_id']: row for data in splits.values()
                 for part in data.values() for row in part.rows}
    features = {}
    for i, (image_id, row) in enumerate(sorted(inventory.items())):
        image = rgb_image(dataset / row['path'])
        image.thumbnail((240, 300), RESAMPLE)
        path = thumbnails / (image_id + '.jpg')
        image.save(path, quality=85)
        features[image_id] = extract_features(cv2.imread(str(path))).astype('float32')
        if i % 250 == 0:
            print(f'Thumbnail features: {i}/{len(inventory)}', flush=True)
    predictions = []
    for seed, data in splits.items():
        x = {part: np.stack([features[r['image_id']] for r in ds.rows]) for part, ds in data.items()}
        y = {part: np.array([int(r['disease_id']) for r in ds.rows]) for part, ds in data.items()}
        scaler = StandardScaler().fit(x['train'])
        model = SVC(kernel='rbf', C=1, gamma='scale', class_weight='balanced', cache_size=400)
        model.fit(scaler.transform(x['train']), y['train'])
        # A fixed candidate, matching the preliminary experiment; never tune on test.
        predicted = model.predict(scaler.transform(x['test']))
        predictions.extend(dict(seed=seed, image_id=row['image_id'], background=row['background'],
                                true_label=int(row['disease_id']), pred_label=int(label))
                           for row, label in zip(data['test'].rows, predicted))
        print(f'Completed preliminary thumbnail seed {seed}', flush=True)
    write_csv(output / 'test_predictions.csv', predictions)
    (output / 'experiment_config.json').write_text(json.dumps(dict(
        scope='preliminary_thumbnail_reference', seeds=seeds, thumbnail_hw=[300, 240],
        jpeg_quality=85, C=1, gamma='scale', class_weight='balanced',
        note='Historical control; not the full-resolution baseline or a CNN result.'), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[11, 22, 33, 44, 55])
    args = parser.parse_args()
    run(args.dataset, args.audit, args.output, args.seeds)
