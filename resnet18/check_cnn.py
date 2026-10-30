"""Shared CNN diagnostics adapted from the group's SharedCNNComponents package.

Commands use the same policy, weights and checkpoint helpers as the experiments.
No command evaluates test images or changes the saved splits.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from data_preprocessing.prepare_dataset import (
    BACKGROUNDS, CLASSES, SEEDS, rgb_image, sha_file, write_json,
)
from data_preprocessing.verify_dataset import verify
from . import common as cc
from .validate_implementation import run_check

AUDIT = Path(__file__).resolve().parents[1] / 'Dhan-Shomadhan/preprocessing'


def download_weights(root):
    """Use the existing loader so online/offline weight validation stays identical."""
    cache = Path(root) / 'model_cache/torchvision'
    model = cc.build_model('frozen', cache_dir=cache)
    del model
    path = cache / Path(cc.WEIGHTS.url).name
    print(f'Verified official ImageNet weights: {path}', flush=True)
    return path


def _font(size):
    try:
        return ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', size)
    except OSError:
        return ImageFont.load_default(size=size)


def _display_tensor(tensor, policy):
    """Undo only fixed ImageNet normalisation for an inspection figure."""
    mean = torch.tensor(policy.mean).view(3, 1, 1)
    std = torch.tensor(policy.std).view(3, 1, 1)
    pixels = ((tensor.detach().cpu() * std + mean).clamp(0, 1) * 255).round().byte()
    return Image.fromarray(pixels.permute(1, 2, 0).numpy())


def preview(root, audit, seed=11, output=None, per_background=2):
    """Show original training images and the actual evaluation/augmentation outputs."""
    root, audit = Path(root).resolve(), Path(audit).resolve()
    output = Path(output or root / 'experiments/resnet18_checks' / f'transform_preview_seed_{seed}').resolve()
    if output.exists() or not output.is_relative_to(root):
        raise ValueError('Choose a fresh preview directory under the dataset')
    if not 1 <= per_background <= 5:
        raise ValueError('Use 1–5 examples per background')
    data = cc.make_shared_data(root, audit, seed)
    rng, selected = random.Random(seed), []
    for background in BACKGROUNDS:
        strata = {}
        for row in data.datasets['train'].rows:
            if row['background'] == background:
                strata.setdefault(row['disease'], []).append(row)
        diseases = sorted(strata)
        rng.shuffle(diseases)
        for disease in diseases[:per_background]:
            selected.append(rng.choice(sorted(strata[disease], key=lambda row: row['image_id'])))
    if len(selected) != per_background * len(BACKGROUNDS):
        raise ValueError('Training partition does not contain the requested preview strata')
    policy = cc.ImagePolicy()
    transform_eval, transform_train = cc.image_transform(False, policy), cc.image_transform(True, policy)
    tile, gap, left, top, row_height = policy.size, 16, 190, 92, policy.size + 45
    titles = ['Original image', 'Evaluation input', 'Training example 1', 'Training example 2', 'Training example 3']
    canvas = Image.new('RGB', (left + len(titles) * (tile + gap), top + len(selected) * row_height + 55), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 12), f'ResNet-18 input policy | seed {seed} | training images only', font=_font(24), fill='black')
    for column, title in enumerate(titles):
        draw.text((left + column * (tile + gap), 57), title, font=_font(17), fill='black')
    # Preview generation must not alter a caller's CPU augmentation stream.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        for index, row in enumerate(selected):
            y = top + index * row_height
            original = rgb_image(root / row['path'])
            original_tile = cc.Letterbox(tile, (255, 255, 255))(original)
            images = [original_tile, _display_tensor(transform_eval(original), policy)] + [
                _display_tensor(transform_train(original), policy) for _ in range(3)]
            draw.multiline_text((16, y + 38),
                f"{row['disease'].replace('_', ' ').title()}\n{row['background']} background\n\n{row['image_id']}",
                font=_font(15), fill='black', spacing=6)
            for column, image in enumerate(images):
                x = left + column * (tile + gap)
                canvas.paste(image, (x, y))
                draw.rectangle((x, y, x + tile - 1, y + tile - 1), outline='#bcbcbc')
    draw.text((16, canvas.height - 40),
        '224 x 224 padding; ImageNet normalisation; training-only horizontal flips and rotations up to 10 degrees. No colour jitter.',
        font=_font(16), fill='black')
    output.mkdir(parents=True)
    path = output / 'transform_preview.png'
    canvas.save(path)
    metadata = dict(scope='input_policy_preview', seed=seed, source_partition='train',
        test_images_evaluated=0, context=data.context, policy=policy.record(),
        code_sha256=sha_file(Path(__file__)), common_code_sha256=sha_file(Path(cc.__file__)),
        display='Fixed normalisation is inverted for viewing. The original panel is resized for display only.',
        images=[{key: row[key] for key in ('image_id', 'path', 'disease', 'background')} for row in selected])
    write_json(output / 'preview_metadata.json', metadata)
    print(f'Saved input-policy preview: {path}', flush=True)
    return path, metadata


def check(root, audit, output, seed=11, epochs=1, batch_size=4, device='auto', workers=0):
    """Check every saved split, then exercise both model stages on train/validation."""
    root, audit = Path(root).resolve(), Path(audit).resolve()
    checked_images, checked_manifests = verify(root, audit)
    split_records = {}
    for current_seed in SEEDS:
        data = cc.make_shared_data(root, audit, current_seed, batch_size=batch_size, num_workers=workers)
        split_records[str(current_seed)] = dict(
            counts={part: len(dataset) for part, dataset in data.datasets.items()}, context=data.context)
    report = run_check(root, audit, output, seed=seed, epochs=epochs,
                       batch_size=batch_size, requested_device=device)
    report.update(source_images_verified=checked_images, manifests_verified=checked_manifests, shared_splits=split_records,
                  checker_code_sha256=sha_file(Path(__file__)), common_code_sha256=sha_file(Path(cc.__file__)))
    write_json(Path(output) / 'shared_component_check.json', report)
    return report


def main(argv=None):
    def shared(parser, top):
        default = (lambda value: value) if top else (lambda _value: argparse.SUPPRESS)
        parser.add_argument('--dataset', type=Path, default=default(Path('Dhan-Shomadhan')))
        parser.add_argument('--audit', type=Path, default=default(AUDIT))
        parser.add_argument('--seed', type=int, choices=SEEDS, default=default(11))
        parser.add_argument('--workers', type=int, default=default(0))
        parser.add_argument('--device', choices=('auto', 'cpu', 'mps', 'cuda'), default=default('auto'))
        parser.add_argument('--cpu', action='store_true', default=default(False), help='Alias for --device cpu')
    parser = argparse.ArgumentParser(description=__doc__)
    shared(parser, True)
    commands = parser.add_subparsers(dest='command', required=True)
    shared(commands.add_parser('download-weights'), False)
    for name in ('check', 'smoke', 'preview'):
        command = commands.add_parser(name)
        shared(command, False)
        command.add_argument('--output', type=Path)
        if name == 'preview':
            command.add_argument('--per-background', type=int, default=2)
        else:
            command.add_argument('--epochs', type=int, choices=(1, 2, 3), default=1)
            command.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args(argv)
    if args.workers < 0 or getattr(args, 'batch_size', 1) < 1:
        parser.error('Use non-negative workers and a positive batch size')
    if args.cpu:
        if args.device not in ('auto', 'cpu'):
            parser.error('--cpu conflicts with the requested accelerator')
        args.device = 'cpu'
    if args.command == 'download-weights':
        download_weights(args.dataset)
    elif args.command == 'preview':
        preview(args.dataset, args.audit, args.seed, args.output, args.per_background)
    elif args.command == 'check':
        output = args.output or args.dataset / f'experiments/resnet18_checks/shared_check/seed_{args.seed}'
        check(args.dataset, args.audit, output, args.seed, args.epochs, args.batch_size, args.device, args.workers)
    else:
        from .train_frozen import main as frozen_main
        output = args.output or args.dataset / 'experiments/resnet18_checks/shared_smoke'
        frozen_main(['--dataset', str(args.dataset), '--audit', str(args.audit), '--seeds', str(args.seed),
            '--output', str(output), '--epochs', str(args.epochs), '--batch-size', str(args.batch_size),
            '--workers', str(args.workers), '--device', args.device, '--smoke'])


if __name__ == '__main__':
    main()
