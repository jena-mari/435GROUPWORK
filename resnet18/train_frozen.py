"""Role 5 frozen ResNet-18 experiment, integrated with the shared CNN helpers.

Adapted from the group-supplied train_frozen.py. Run from the repository root.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from data_preprocessing.prepare_dataset import SEEDS, sha_file, write_csv, write_json
from data_preprocessing.verify_dataset import verify
from . import common as cc
from .validate_implementation import check_changes, small_loader, snapshot

METHOD = 'frozen_resnet18'
AUDIT = Path(__file__).resolve().parents[1] / 'Dhan-Shomadhan/preprocessing'


def run_seed(seed, args, device):
    cc.seed_everything(seed)
    root, output = args.dataset.resolve(), args.output.resolve()
    directory = output / f'seed_{seed}'
    data = cc.make_shared_data(root, args.audit, seed, batch_size=args.batch_size,
                               num_workers=args.workers)
    training, validation = data.loaders['train'], data.loaders['validation']
    if args.smoke:
        training = small_loader(data.datasets['train'], seed, 1, args.batch_size, True)
        validation = small_loader(data.datasets['validation'], seed + 1000, 1, args.batch_size, False)
        data.context.update(run_scope='implementation_check', check_image_ids={
            part: [r['image_id'] for r in cc.dataset_rows(loader.dataset)]
            for part, loader in [('train', training), ('validation', validation)]})
    checkpoint = directory / 'best.pt'
    if args.evaluate_only:
        config = json.loads((directory / 'run_config.json').read_text())
        if config['context'] != data.context or not config.get('training_complete'):
            raise ValueError('Evaluation requires a completed training run with the same context')
        if config.get('method') != METHOD or config.get('smoke'):
            raise ValueError('Choose a completed full-data frozen experiment')
        model = cc.build_model('frozen', pretrained=False).to(device)
        payload = cc.load_checkpoint(checkpoint, model, data.context, 'frozen')
        if sha_file(checkpoint) != config.get('checkpoint_sha256') or payload['epoch'] != config.get('best_epoch'):
            raise ValueError('Selected checkpoint differs from the completed run record')
        if payload['weights_source'] != 'IMAGENET1K_V1':
            raise ValueError('The frozen experiment requires ImageNet-pretrained weights')
    else:
        directory.mkdir(parents=True, exist_ok=False)
        model = cc.build_model('frozen', cache_dir=root / 'model_cache/torchvision',
                               weights_path=args.weights_path).to(device)
        optimizer = cc.make_optimizer(model, 'frozen')
        config = cc.run_configuration(model, optimizer, data, 'frozen', device)
        config.update(actual_batch_size=args.batch_size, actual_max_epochs=args.epochs,
            actual_patience=args.patience, workers=args.workers, smoke=args.smoke,
            training_images=len(training.dataset), validation_images=len(validation.dataset),
            dataset_root=str(root), audit_directory=str(args.audit.resolve()),
            driver_code_sha256=sha_file(Path(__file__)), checkpoint='best.pt',
            selection_rule='strictly highest validation macro-F1; ties count as no improvement',
            validation_only=args.validation_only, training_complete=False)
        write_json(directory / 'run_config.json', config)
        selector = cc.BestCheckpoint(checkpoint, data.context, 'frozen', args.patience)
        before = snapshot(model)  # Verify the complete frozen backbone, including BN buffers.
        history, started = [], time.monotonic()
        for epoch in range(1, args.epochs + 1):
            train = cc.train_one_epoch(model, training, optimizer, device, 'frozen')
            val = cc.evaluate(model, validation, device)
            history.append(dict(epoch=epoch, training_loss=train['loss'],
                training_accuracy=train['accuracy'], validation_loss=val['loss'],
                validation_accuracy=val['accuracy'], validation_macro_f1=val['macro_f1']))
            stop = selector.update(val['macro_f1'], epoch, model, optimizer, config)
            write_csv(directory / 'validation_history.csv', history)
            print(f"seed {seed} epoch {epoch}: train loss {train['loss']:.4f}, "
                  f"validation macro-F1 {val['macro_f1']:.4f}, best epoch {selector.best_epoch}", flush=True)
            if stop:
                break
        changes = check_changes(model, before, 'frozen')
        config.update(training_complete=True, epochs_completed=len(history),
            best_epoch=selector.best_epoch, best_validation_macro_f1=selector.best_score,
            elapsed_training_seconds=time.monotonic() - started, frozen_contract_check=changes,
            checkpoint_sha256=sha_file(checkpoint))
        write_json(directory / 'run_config.json', config)
        cc.load_checkpoint(checkpoint, model, data.context, 'frozen')
        rows = cc.export_predictions(model, validation, directory / 'validation_predictions.csv',
                                     data.context, 'frozen', device)
        write_json(directory / 'validation_metrics.json', cc.score_predictions(rows))
    if args.validation_only:
        print(f'Validation only: {directory}; test images evaluated: 0', flush=True)
        return None
    rows = cc.export_predictions(model, data.loaders['test'], directory / 'test_predictions.csv',
                                 data.context, 'frozen', device)
    scores = cc.score_predictions(rows)
    write_json(directory / 'metrics.json', scores)
    write_json(directory / 'test_evaluation.json', dict(checkpoint_sha256=sha_file(checkpoint),
        evaluation_device=str(device), context=data.context, images=len(rows)))
    print(f'Saved selected-checkpoint test predictions: {directory}', flush=True)
    return scores


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('Dhan-Shomadhan'))
    parser.add_argument('--audit', type=Path, default=AUDIT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seeds', nargs='+', type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--device', choices=('auto', 'cpu', 'mps', 'cuda'), default='auto')
    parser.add_argument('--weights-path', type=Path, help='Optional official ImageNet weights for offline loading')
    parser.add_argument('--validation-only', action='store_true', help='Train/select/export validation predictions; do not evaluate test')
    parser.add_argument('--evaluate-only', action='store_true', help='Export test predictions from completed validation-only runs')
    parser.add_argument('--smoke', action='store_true', help='10 train/10 validation images; diagnostic checkpoints; no test evaluation')
    args = parser.parse_args(argv)
    if len(args.seeds) != len(set(args.seeds)) or min(args.epochs, args.batch_size, args.patience) < 1 or args.workers < 0:
        parser.error('Use distinct seeds and positive epoch/batch/patience values; workers must be non-negative')
    if args.evaluate_only and (args.validation_only or args.smoke):
        parser.error('--evaluate-only cannot be combined with --validation-only or --smoke')
    if args.smoke:
        if args.epochs > 3:
            parser.error('For --smoke specify --epochs 1, 2 or 3')
        args.validation_only = True
    args.output = args.output or args.dataset / 'experiments' / ('resnet18_checks/frozen_smoke' if args.smoke else METHOD)
    if not args.output.resolve().is_relative_to(args.dataset.resolve()):
        parser.error('Save experiment outputs under the main dataset directory')
    for seed in args.seeds:
        directory = args.output / f'seed_{seed}'
        if args.evaluate_only:
            if not (directory / 'best.pt').is_file() or not (directory / 'run_config.json').is_file():
                parser.error(f'No completed checkpoint/configuration: {directory}')
            if any((directory / name).exists() for name in ('test_predictions.csv', 'test_predictions.metadata.json', 'metrics.json', 'test_evaluation.json')):
                parser.error(f'Test outputs already exist: {directory}')
        elif directory.exists():
            parser.error(f'Run directory exists; choose another --output: {directory}')
    verify(args.dataset, args.audit)
    device = cc.choose_device(args.device)
    results = {seed: run_seed(seed, args, device) for seed in args.seeds}
    if not args.validation_only:
        if set(results) == set(SEEDS):
            write_csv(args.output / 'summary.csv', cc.summarise_runs(results))
        else:
            print('Partial invocation: no five-run summary. Use the evaluation package once all five seeds are ready.')


if __name__ == '__main__':
    main()
