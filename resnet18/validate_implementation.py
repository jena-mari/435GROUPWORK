"""Small pretrained checks for Role 5/6 components; never evaluates test images.

This is an implementation check, not the final five-run CNN experiment.
"""
import argparse
import copy
import platform
import random
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader, Subset

from data_preprocessing.prepare_dataset import write_csv, write_json
from .common import (
    BestCheckpoint, METHODS, assert_model_contract, assert_optimizer_contract,
    build_model, choose_device, dataset_rows, evaluate, export_predictions,
    initialise_finetuned, load_checkpoint, make_optimizer, make_shared_data,
    read_checkpoint, run_configuration, score_predictions, seed_everything, train_one_epoch,
)


def small_loader(dataset, seed, per_stratum, batch_size, training):
    strata = {}
    for i, row in enumerate(dataset.rows):
        strata.setdefault((row['disease'], row['background']), []).append(i)
    rng = random.Random(seed)
    selected = []
    for key in sorted(strata):
        pool = strata[key][:]
        rng.shuffle(pool)
        selected.extend(pool[:per_stratum])
    return DataLoader(Subset(dataset, selected), batch_size=batch_size, shuffle=training,
                      num_workers=0, generator=torch.Generator().manual_seed(seed))


def snapshot(model):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def check_changes(model, before, stage):
    changed = [name for name, value in model.state_dict().items()
               if not torch.equal(before[name], value.detach().cpu())]
    permitted = ('fc.',) if stage == 'frozen' else ('fc.', 'layer4.')
    unexpected = [name for name in changed if not name.startswith(permitted)]
    running_changes = [name for name in changed if name.endswith(('running_mean', 'running_var', 'num_batches_tracked'))]
    if unexpected or running_changes or not any(name.startswith('fc.') for name in changed):
        raise ValueError(f'Incorrect parameter/buffer changes: {unexpected + running_changes}')
    if stage == 'finetuned' and not any(name.startswith('layer4.') for name in changed):
        raise ValueError('No layer4 parameter changed during the fine-tuning check')
    return dict(changed_parameter_or_buffer_names=changed, fixed_batchnorm_buffers=True,
                frozen_parameters_unchanged=True)


def run_check(dataset_root, audit, output, seed=11, epochs=1, per_stratum=1,
              batch_size=4, requested_device='auto', weights_path=None):
    root, audit, output = map(lambda p: Path(p).resolve(), (dataset_root, audit, output))
    if output.exists() or not output.is_relative_to(root):
        raise ValueError('Choose a new output directory under the main dataset directory')
    if epochs < 1 or epochs > 3 or per_stratum < 1 or per_stratum > 3 or batch_size < 1:
        raise ValueError('Use 1–3 check epochs, 1–3 images per stratum and a positive batch size')
    seed_everything(seed)
    device = choose_device(requested_device)
    data = make_shared_data(root, audit, seed, batch_size=batch_size)
    training = small_loader(data.datasets['train'], seed, per_stratum, batch_size, True)
    validation = small_loader(data.datasets['validation'], seed + 1000, per_stratum, batch_size, False)
    context = copy.deepcopy(data.context)
    context['run_scope'] = 'implementation_check'
    context['check_image_ids'] = {part: [r['image_id'] for r in dataset_rows(loader.dataset)]
                                  for part, loader in [('train', training), ('validation', validation)]}
    output.mkdir(parents=True)
    write_json(output / 'run_config.json', dict(scope='implementation_check', device=str(device),
        seed=seed, epochs=epochs, batch_size=batch_size, per_stratum=per_stratum,
        context=context, python=platform.python_version(), torch=str(torch.__version__),
        torchvision=torchvision.__version__, test_images_used=0))
    report = dict(scope='implementation_check', final_experiment=False, seed=seed,
        device=str(device), training_images=len(training.dataset),
        validation_images=len(validation.dataset), test_images_used=0, stages={})
    print(f'Implementation check on {device}: {len(training.dataset)} train, '
          f'{len(validation.dataset)} validation, 0 test images', flush=True)
    frozen_path = output / 'frozen' / 'best.pt'
    for stage in ('frozen', 'finetuned'):
        print(f'Checking {stage} mode', flush=True)
        if stage == 'frozen':
            model = build_model('frozen', cache_dir=root / 'model_cache' / 'torchvision',
                                weights_path=weights_path).to(device)
        else:
            model, source = initialise_finetuned(frozen_path, context, device)
            report['frozen_source_epoch'] = source['epoch']
        contract = assert_model_contract(model, stage)
        optimizer = make_optimizer(model, stage)
        assert_optimizer_contract(model, optimizer, stage)
        before = snapshot(model)
        directory = output / stage
        directory.mkdir(parents=True)
        selector = BestCheckpoint(directory / 'best.pt', context, stage)
        config = run_configuration(model, optimizer, data, stage, device)
        config.update(context=context, actual_batch_size=batch_size, actual_max_epochs=epochs,
                      run_scope='implementation_check')
        write_json(directory / 'run_config.json', config)
        history = []
        for epoch in range(1, epochs + 1):
            train = train_one_epoch(model, training, optimizer, device, stage)
            val = evaluate(model, validation, device)
            history.append(dict(epoch=epoch, training_loss=train['loss'], validation_loss=val['loss'],
                                training_macro_f1=train['macro_f1'], validation_macro_f1=val['macro_f1']))
            selector.update(val['macro_f1'], epoch, model, optimizer,
                            dict(check_only=True, batch_size=batch_size))
            print(f"  epoch {epoch}: train loss {train['loss']:.4f}, validation loss {val['loss']:.4f}", flush=True)
        changes = check_changes(model, before, stage)
        write_csv(directory / 'validation_history.csv', history)
        load_checkpoint(directory / 'best.pt', model, context, stage)
        predictions = export_predictions(model, validation, directory / 'validation_predictions.csv',
                                         context, stage, device)
        write_json(directory / 'validation_metrics.json', score_predictions(predictions))
        restored = build_model(stage, pretrained=False).to(device)
        load_checkpoint(directory / 'best.pt', restored, context, stage)
        model.eval(); restored.eval()
        images = next(iter(validation))['image'].to(device)
        with torch.inference_mode():
            torch.testing.assert_close(model(images), restored(images), rtol=1e-5, atol=1e-6)
        wrong = copy.deepcopy(context); wrong['seed'] = 22 if seed != 22 else 11
        try:
            read_checkpoint(directory / 'best.pt', wrong, stage)
        except ValueError:
            rejected_wrong_seed = True
        else:
            raise ValueError('Wrong-seed checkpoint was accepted')
        report['stages'][stage] = dict(method=METHODS[stage], contract=contract, **changes,
            checkpoint_round_trip=True, rejected_wrong_seed=rejected_wrong_seed,
            validation_predictions=len(predictions), best_epoch=selector.best_epoch)
        del restored, optimizer, model, before
        if device.type == 'mps':
            torch.mps.empty_cache()
    report['passed'] = True
    write_json(output / 'implementation_check.json', report)
    print(f'PASS: implementation checks saved under {output}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('Dhan-Shomadhan'))
    parser.add_argument('--audit', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--per-stratum', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    parser.add_argument('--weights-path', type=Path)
    args = parser.parse_args()
    audit = args.audit or Path(__file__).resolve().parents[1] / 'Dhan-Shomadhan/preprocessing'
    output = args.output or args.dataset / 'experiments/resnet18_checks' / f'seed_{args.seed}'
    run_check(args.dataset, audit, output, args.seed, args.epochs, args.per_stratum,
              args.batch_size, args.device, args.weights_path)


if __name__ == '__main__':
    main()
