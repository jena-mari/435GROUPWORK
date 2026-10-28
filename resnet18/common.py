"""Shared ResNet-18 inputs, model contracts, checkpointing and predictions.

Roles 5 and 6 own their experiment loops. This module fixes the parts that must
agree between them and rejects incompatible manifests and checkpoints.
"""
from __future__ import annotations

import json
import math
import os
import platform
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torchvision
import PIL
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import InterpolationMode

from data_preprocessing.prepare_dataset import (
    BACKGROUNDS, CLASSES, PARTITIONS, SEEDS, RiceDataset, sha_file,
    validate_split, write_csv, write_json,
)

STAGES = ('frozen', 'finetuned')
METHODS = {'frozen': 'frozen_resnet18', 'finetuned': 'finetuned_resnet18'}
WEIGHTS = ResNet18_Weights.IMAGENET1K_V1
STAGE_DEFAULTS = {
    'frozen': dict(batch_size=32, max_epochs=30, patience=5),
    'finetuned': dict(batch_size=32, max_epochs=20, patience=5),
}


@dataclass(frozen=True)
class ImagePolicy:
    size: int = 224
    horizontal_flip_probability: float = .5
    rotation_degrees: float = 10.
    mean: tuple = (.485, .456, .406)
    std: tuple = (.229, .224, .225)
    fill: tuple = (124, 116, 104)

    def record(self):
        return json.loads(json.dumps(dict(asdict(self),
            resize='bilinear aspect-ratio-preserving letterbox',
            training_order='letterbox, horizontal flip, expanded rotation, letterbox, tensor, normalise',
            evaluation_order='letterbox, tensor, normalise', color_jitter=False)))


@dataclass(frozen=True)
class Letterbox:
    size: int
    fill: tuple

    def __call__(self, image):
        image = ImageOps.contain(image.convert('RGB'), (self.size, self.size),
                                method=Image.Resampling.BILINEAR)
        canvas = Image.new('RGB', (self.size, self.size), self.fill)
        canvas.paste(image, ((self.size - image.width) // 2, (self.size - image.height) // 2))
        return canvas


def image_transform(training=False, policy=ImagePolicy()):
    if policy.size <= 0 or not 0 <= policy.horizontal_flip_probability <= 1 or policy.rotation_degrees < 0:
        raise ValueError('Invalid image policy')
    operations = [Letterbox(policy.size, policy.fill)]
    if training:
        operations += [transforms.RandomHorizontalFlip(policy.horizontal_flip_probability),
            transforms.RandomRotation(policy.rotation_degrees, interpolation=InterpolationMode.BILINEAR,
                                      expand=True, fill=policy.fill),
            Letterbox(policy.size, policy.fill)]
    return transforms.Compose(operations + [transforms.ToTensor(), transforms.Normalize(policy.mean, policy.std)])


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(_worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def choose_device(requested='auto'):
    if requested == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if requested == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS is unavailable; select cpu or check the PyTorch/macOS installation')
    if requested == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable on this machine')
    return torch.device(requested)


@dataclass
class SharedData:
    datasets: dict
    loaders: dict
    context: dict


def make_shared_data(dataset_root, audit, seed, batch_size=32, num_workers=0, policy=ImagePolicy()):
    root, audit = Path(dataset_root).resolve(), Path(audit).resolve()
    if seed not in SEEDS or batch_size <= 0 or num_workers < 0:
        raise ValueError('Invalid split seed, batch size or worker count')
    version = (audit / (audit / 'CURRENT_SPLIT_VERSION.txt').read_text().strip()).resolve()
    if not version.is_relative_to(audit):
        raise ValueError('Split version must remain inside the shared audit directory')
    provenance = json.loads((version / 'provenance.json').read_text())
    datasets, loaders, hashes, all_rows, assignment = {}, {}, {}, [], {}
    for index, partition in enumerate(PARTITIONS):
        relative = f'seed_{seed}/{partition}.csv'
        path = version / relative
        hashes[partition] = sha_file(path)
        if hashes[partition] != provenance['manifest_sha256'][relative]:
            raise ValueError(f'Manifest checksum changed: {relative}')
        dataset = RiceDataset(root, path, transform=image_transform(partition == 'train', policy))
        for row in dataset.rows:
            if int(row['seed']) != seed or row['partition'] != partition:
                raise ValueError('Manifest seed or partition mismatch')
            gid = row['group_id']
            if gid in assignment and assignment[gid] != partition:
                raise ValueError('Related-image group crosses partitions')
            assignment[gid] = partition
        datasets[partition] = dataset
        all_rows.extend(dataset.rows)
        loaders[partition] = DataLoader(dataset, batch_size=batch_size,
            shuffle=partition == 'train', num_workers=num_workers, drop_last=False,
            worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed + index))
    validate_split(all_rows, assignment)
    if len(all_rows) != provenance['eligible_images']:
        raise ValueError('Manifests do not cover the eligible inventory')
    context = dict(seed=seed, manifest_version=version.relative_to(audit).as_posix(),
                   manifest_sha256=hashes, policy=policy.record(), class_mapping=CLASSES,
                   background_mapping=BACKGROUNDS, run_scope='experiment')
    return SharedData(datasets, loaders, context)


def build_model(stage='frozen', *, pretrained=True, cache_dir=None, weights_path=None):
    if stage not in STAGES:
        raise ValueError(f'Choose a stage from {STAGES}')
    if stage == 'finetuned' and pretrained:
        raise ValueError('Use initialise_finetuned() with the same-split frozen checkpoint')
    model = resnet18(weights=None)
    if pretrained:
        if weights_path is not None:
            expected = Path(WEIGHTS.url).stem.rsplit('-', 1)[-1]
            if not sha_file(Path(weights_path)).startswith(expected):
                raise ValueError('Offline weights do not match the official ImageNet ResNet-18 checksum')
            state = torch.load(weights_path, map_location='cpu', weights_only=True)
        else:
            if cache_dir is None:
                raise ValueError('Specify cache_dir under the dataset for pretrained weights')
            cache_file = Path(cache_dir) / Path(WEIGHTS.url).name
            expected = cache_file.stem.rsplit('-', 1)[-1]
            if cache_file.exists() and not sha_file(cache_file).startswith(expected):
                raise ValueError('Cached ImageNet weights failed checksum verification; replace the damaged cache file')
            state = torch.hub.load_state_dict_from_url(WEIGHTS.url, model_dir=str(cache_dir),
                        map_location='cpu', check_hash=True, progress=False, weights_only=True)
        model.load_state_dict(state, strict=True)
    model.fc = nn.Linear(512, len(CLASSES))
    model.rice_weights_source = 'IMAGENET1K_V1' if pretrained else 'NONE'
    configure_trainable(model, stage)
    return model


def configure_trainable(model, stage):
    if stage not in STAGES:
        raise ValueError(f'Choose a stage from {STAGES}')
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith('fc.') or (stage == 'finetuned' and name.startswith('layer4.'))
    prepare_training_mode(model, stage)
    assert_model_contract(model, stage)


def prepare_training_mode(model, stage):
    """Call after any model.train(): gradients and BatchNorm state are separate."""
    if stage not in STAGES:
        raise ValueError(f'Choose a stage from {STAGES}')
    model.train()
    for name, child in model.named_children():
        child.train(name == 'fc' or (stage == 'finetuned' and name == 'layer4'))
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def assert_model_contract(model, stage):
    if stage not in STAGES or not isinstance(model.fc, nn.Linear) or model.fc.in_features != 512 or model.fc.out_features != 5 or model.fc.bias is None:
        raise ValueError('Expected a ResNet-18 with a biased 512-to-5 linear head')
    if any(len(getattr(model, f'layer{i}')) != 2 for i in range(1, 5)):
        raise ValueError('Expected two residual blocks in each ResNet-18 stage')
    for name, parameter in model.named_parameters():
        expected = name.startswith('fc.') or (stage == 'finetuned' and name.startswith('layer4.'))
        if parameter.requires_grad != expected:
            raise ValueError(f'Unexpected trainable/frozen parameter: {name}')
    for name, child in model.named_children():
        if name != 'fc' and not (stage == 'finetuned' and name == 'layer4') and child.training:
            raise ValueError(f'Frozen backbone stage is in training mode: {name}')
    for name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            if module.training or not module.track_running_stats or module.running_mean is None or module.running_var is None:
                raise ValueError(f'BatchNorm running statistics are not fixed: {name}')
    return dict(trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                trainable_names=[n for n, p in model.named_parameters() if p.requires_grad])


def make_optimizer(model, stage):
    assert_model_contract(model, stage)
    groups = [dict(params=list(model.fc.parameters()), lr=1e-3)] if stage == 'frozen' else [
        dict(params=list(model.layer4.parameters()), lr=1e-5),
        dict(params=list(model.fc.parameters()), lr=1e-4)]
    return torch.optim.Adam(groups, betas=(.9, .999), eps=1e-8, weight_decay=0.)


def assert_optimizer_contract(model, optimizer, stage):
    if not isinstance(optimizer, torch.optim.Adam):
        raise ValueError('The agreed optimiser is Adam')
    expected = [(set(map(id, model.fc.parameters())), 1e-3)] if stage == 'frozen' else [
        (set(map(id, model.layer4.parameters())), 1e-5), (set(map(id, model.fc.parameters())), 1e-4)]
    actual = optimizer.param_groups
    if len(actual) != len(expected):
        raise ValueError('Incorrect optimiser parameter groups')
    for group in actual:
        ids = [id(parameter) for parameter in group['params']]
        if len(ids) != len(set(ids)) or not any(set(ids) == params and group['lr'] == lr for params, lr in expected):
            raise ValueError('Optimiser parameters or learning rates disagree with the stage')
        if group['weight_decay'] != 0 or group['betas'] != (.9, .999) or group['eps'] != 1e-8:
            raise ValueError('Record a deliberate change before changing the common Adam settings')


def run_configuration(model, optimizer, data, stage, device):
    """JSON-ready settings; add the actual epoch/batch overrides used by a run."""
    contract = assert_model_contract(model, stage)
    assert_optimizer_contract(model, optimizer, stage)
    if stage == 'finetuned' and not getattr(model, 'rice_frozen_checkpoint_sha256', None):
        raise ValueError('Initialise fine-tuning from the same-split frozen checkpoint first')
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    return dict(method=METHODS[stage], stage=stage, context=_plain(data.context),
        planned_defaults=_plain(STAGE_DEFAULTS[stage]), device=str(device),
        python=platform.python_version(), torch=str(torch.__version__),
        torchvision=torchvision.__version__, numpy=np.__version__, pillow=PIL.__version__,
        weights_source=getattr(model, 'rice_weights_source', 'UNKNOWN'),
        frozen_source_sha256=getattr(model, 'rice_frozen_checkpoint_sha256', None),
        frozen_source_epoch=getattr(model, 'rice_frozen_epoch', None),
        common_code_sha256=sha_file(Path(__file__)), **contract,
        optimizer='Adam', optimizer_groups=[dict(lr=group['lr'], betas=list(group['betas']),
            eps=group['eps'], weight_decay=group['weight_decay'],
            parameter_names=[names[id(p)] for p in group['params']]) for group in optimizer.param_groups])


def dataset_rows(dataset):
    if isinstance(dataset, Subset):
        rows = dataset_rows(dataset.dataset)
        return [rows[index] for index in dataset.indices]
    if not hasattr(dataset, 'rows'):
        raise ValueError('Use RiceDataset or a Subset of it to retain image metadata')
    return dataset.rows


def _logits(model, images):
    logits = model(images)
    if logits.shape != (len(images), 5) or not torch.isfinite(logits).all():
        raise ValueError('Model must return finite B-by-5 logits, without a final softmax')
    return logits


def _epoch_result(loss_sum, truth, predicted):
    if not truth:
        raise ValueError('Empty data loader')
    return dict(loss=loss_sum / len(truth), images=len(truth),
                accuracy=float(accuracy_score(truth, predicted)),
                macro_f1=float(f1_score(truth, predicted, labels=list(range(5)),
                                        average='macro', zero_division=0)))


def train_one_epoch(model, loader, optimizer, device, stage):
    if any(row['partition'] != 'train' for row in dataset_rows(loader.dataset)):
        raise ValueError('Training requires the training partition')
    prepare_training_mode(model, stage)
    assert_model_contract(model, stage)
    assert_optimizer_contract(model, optimizer, stage)
    total, truth, predicted = 0., [], []
    for batch in loader:
        images, labels = batch['image'].to(device), batch['disease_id'].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = _logits(model, images)
        loss = nn.functional.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            raise ValueError('Non-finite training loss')
        loss.backward()
        optimizer.step()
        total += float(loss.detach().cpu()) * len(labels)
        truth.extend(labels.detach().cpu().tolist())
        predicted.extend(logits.detach().argmax(1).cpu().tolist())
    return _epoch_result(total, truth, predicted)


def evaluate(model, loader, device):
    model.eval()
    total, truth, predicted = 0., [], []
    with torch.inference_mode():
        for batch in loader:
            images, labels = batch['image'].to(device), batch['disease_id'].to(device)
            logits = _logits(model, images)
            total += float(nn.functional.cross_entropy(logits, labels).cpu()) * len(labels)
            truth.extend(labels.cpu().tolist())
            predicted.extend(logits.argmax(1).cpu().tolist())
    return _epoch_result(total, truth, predicted)


def _plain(value):
    return json.loads(json.dumps(value))


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu(item) for item in value)
    return value


def read_checkpoint(path, context, required_stage):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('format_version') != 1 or payload.get('stage') != required_stage:
        raise ValueError('Checkpoint stage or format mismatch')
    if payload.get('context') != _plain(context):
        raise ValueError('Checkpoint seed, manifests, preprocessing policy or run scope mismatch')
    return payload


def load_checkpoint(path, model, context, required_stage, optimizer=None):
    payload = read_checkpoint(path, context, required_stage)
    model.load_state_dict(payload['model_state'], strict=True)
    model.rice_weights_source = payload['weights_source']
    model.rice_checkpoint_context = _plain(context)
    model.rice_checkpoint_stage = required_stage
    model.rice_frozen_checkpoint_sha256 = payload.get('frozen_source_sha256')
    model.rice_frozen_epoch = payload.get('frozen_source_epoch')
    if optimizer is not None:
        optimizer.load_state_dict(payload['optimizer_state'])
    return payload


class BestCheckpoint:
    """Independent validation-macro-F1 selection; ties count as no improvement."""
    def __init__(self, path, context, stage, patience=5):
        self.path, self.context, self.stage = Path(path), _plain(context), stage
        if self.path.exists() or stage not in STAGES or patience < 1:
            raise ValueError('Use a new checkpoint path, a valid stage and positive patience')
        self.patience, self.best_score, self.bad_epochs, self.best_epoch = patience, -math.inf, 0, None

    def update(self, score, epoch, model, optimizer, config=None):
        score = float(score)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Validation macro-F1 must be finite and between zero and one')
        if self.stage == 'finetuned' and not getattr(model, 'rice_frozen_checkpoint_sha256', None):
            raise ValueError('Fine-tuned checkpoints require same-split frozen-checkpoint initialisation')
        assert_model_contract(model, self.stage)
        assert_optimizer_contract(model, optimizer, self.stage)
        if score > self.best_score:
            payload = dict(format_version=1, stage=self.stage, context=self.context,
                epoch=int(epoch), validation_macro_f1=score, model_state=_cpu(model.state_dict()),
                optimizer_state=_cpu(optimizer.state_dict()), config=_plain(config or {}),
                weights_source=getattr(model, 'rice_weights_source', 'UNKNOWN'),
                frozen_source_sha256=getattr(model, 'rice_frozen_checkpoint_sha256', None),
                frozen_source_epoch=getattr(model, 'rice_frozen_epoch', None),
                torch_version=str(torch.__version__))
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(self.path.suffix + '.tmp')
            torch.save(payload, temporary)
            os.replace(temporary, self.path)
            self.best_score, self.best_epoch, self.bad_epochs = score, epoch, 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def initialise_finetuned(frozen_checkpoint, context, device):
    model = build_model('finetuned', pretrained=False)
    payload = load_checkpoint(frozen_checkpoint, model, context, required_stage='frozen')
    model.rice_frozen_checkpoint_sha256 = sha_file(Path(frozen_checkpoint))
    model.rice_frozen_epoch = payload['epoch']
    configure_trainable(model, 'finetuned')
    return model.to(device), payload


def score_predictions(rows):
    scores = {}
    for scenario in ('field', 'white', 'all'):
        selected = rows if scenario == 'all' else [r for r in rows if r['background_id'] == BACKGROUNDS[scenario]]
        if not selected:
            raise ValueError(f'No predictions for background scenario {scenario}')
        truth = [r['disease_id'] for r in selected]
        predicted = [r['predicted_disease_id'] for r in selected]
        scores[scenario] = dict(images=len(selected), accuracy=float(accuracy_score(truth, predicted)),
            macro_f1=float(f1_score(truth, predicted, labels=list(range(5)), average='macro', zero_division=0)),
            confusion_matrix=confusion_matrix(truth, predicted, labels=list(range(5))).tolist())
    return scores


def summarise_runs(scores_by_seed):
    if set(scores_by_seed) != set(SEEDS):
        raise ValueError('Five-run summaries require seeds 11, 22, 33, 44 and 55')
    result = []
    for scenario in ('field', 'white', 'all'):
        row = dict(scenario=scenario, runs=5)
        for metric in ('accuracy', 'macro_f1'):
            values = [scores_by_seed[seed][scenario][metric] for seed in SEEDS]
            if not np.isfinite(values).all():
                raise ValueError('Non-finite run metrics')
            row[metric + '_mean'] = float(np.mean(values))
            row[metric + '_std'] = float(np.std(values, ddof=1))
        result.append(row)
    return result


def export_predictions(model, loader, path, context, stage, device):
    if stage not in STAGES:
        raise ValueError('Invalid model stage')
    path = Path(path)
    if path.exists() or path.with_suffix('.metadata.json').exists():
        raise ValueError('Prediction output exists; choose a new path')
    rows = dataset_rows(loader.dataset)
    if any(int(row['seed']) != context['seed'] for row in rows):
        raise ValueError('Prediction loader seed disagrees with the run context')
    partitions = {r['partition'] for r in rows}
    if len(partitions) != 1 or partitions == {'train'}:
        raise ValueError('Export predictions for a single validation or test partition')
    if 'test' in partitions and context['run_scope'] != 'experiment':
        raise ValueError('Implementation-check runs must not export test predictions')
    if getattr(model, 'rice_checkpoint_context', None) != _plain(context) or getattr(model, 'rice_checkpoint_stage', None) != stage:
        raise ValueError('Load the selected checkpoint for this context and stage before exporting predictions')
    expected = {row['image_id']: row for row in rows}
    if len(expected) != len(rows):
        raise ValueError('Repeated image IDs in prediction dataset')
    collected = {}
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            probabilities = _logits(model, batch['image'].to(device)).softmax(1).cpu()
            for i, image_id in enumerate(batch['image_id']):
                if image_id not in expected or image_id in collected:
                    raise ValueError('Unexpected or repeated prediction image ID')
                source = expected[image_id]
                label, background = int(batch['disease_id'][i]), int(batch['background_id'][i])
                if (label, background) != (int(source['disease_id']), int(source['background_id'])):
                    raise ValueError('Prediction metadata disagrees with the manifest')
                probability = probabilities[i].tolist()
                collected[image_id] = dict(image_id=image_id, seed=context['seed'], method=METHODS[stage],
                    disease_id=label, predicted_disease_id=int(probabilities[i].argmax()), background_id=background,
                    **{f'prob_{index}': float(value) for index, value in enumerate(probability)})
    if set(collected) != set(expected):
        raise ValueError('Missing image predictions')
    predictions = [collected[row['image_id']] for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(path, predictions)
    write_json(path.with_suffix('.metadata.json'), dict(context=_plain(context), stage=stage,
        partition=next(iter(partitions)), images=len(predictions), weights_source=getattr(model, 'rice_weights_source', 'UNKNOWN')))
    return predictions
