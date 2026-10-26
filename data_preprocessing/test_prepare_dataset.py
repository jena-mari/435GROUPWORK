"""Tests for leakage-sensitive preparation behaviour; run with unittest."""
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from prepare_dataset import (BACKGROUNDS, CLASSES, PARTITIONS, RiceDataset,
                             grouped_split, label_path, resolve_inventory,
                             validate_split, write_csv)


def fixture():
    return [dict(image_id=f'{d}_{b}_{i}', path=f'{d}/{b}/{i}.jpg',
                 disease=d, background=b, readable='1',
                 sha256=f'{d}_{b}_{i}', group_id=f'{d}_{b}_{i}')
            for d in CLASSES for b in BACKGROUNDS for i in range(10)]


class PreparationTests(unittest.TestCase):
    def test_real_folder_aliases_and_trailing_spaces(self):
        self.assertEqual(label_path(Path('Field Background  /Browon Spot/bsf_0.jpg')),
                         ('brown_spot', 'field'))
        self.assertEqual(label_path(Path('White Background /Shath Blight/sbw_0.jpg')),
                         ('sheath_blight', 'white'))
        self.assertEqual(label_path(Path('White Background /Unknown/0.jpg')), ('', 'white'))

    def test_grouped_reproducibility_and_coverage(self):
        rows = fixture()
        # One repeated leaf spans two backgrounds. It must stay intact.
        rows[10]['group_id'] = rows[0]['group_id']
        a, _ = grouped_split(rows, 11, trials=12)
        b, _ = grouped_split(rows, 11, trials=12)
        c, _ = grouped_split(rows, 22, trials=12)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        validate_split(rows, a)
        self.assertEqual(set(a.values()), set(PARTITIONS))
        # Retain exactly one assignment for each eligible image.
        self.assertEqual(sum(sum(a[r['group_id']] == p for r in rows)
                             for p in PARTITIONS), len(rows))

    def test_infeasible_groups_fail(self):
        rows = fixture()
        for r in rows[:10]:
            r['group_id'] = 'same_leaf'
        with self.assertRaisesRegex(ValueError, 'Fewer than 3'):
            grouped_split(rows, 11, trials=2)

    def test_pending_pairs_block_build(self):
        rows = fixture()[:3]
        pairs = [dict(left_id=rows[0]['image_id'], right_id=rows[1]['image_id'],
                      decision='pending', reason='')]
        with self.assertRaisesRegex(ValueError, 'Pending'):
            resolve_inventory(rows, [], pairs)

    def test_exact_conflicts_require_resolution(self):
        rows = [fixture()[0], fixture()[10]]
        rows[1]['sha256'] = rows[0]['sha256']
        with self.assertRaisesRegex(ValueError, 'label conflict'):
            resolve_inventory(rows, [], [])
        decision = dict(image_id=rows[1]['image_id'], action='exclude',
                        reason='manual source-label resolution')
        eligible, _ = resolve_inventory(rows, [decision], [])
        self.assertEqual(len(eligible), 1)

    def test_exact_dedup_preserves_transitive_relatedness(self):
        rows = fixture()[:3]
        rows[1]['sha256'] = rows[0]['sha256']
        pairs = [dict(left_id=rows[1]['image_id'], right_id=rows[2]['image_id'],
                      decision='related', reason='same visible leaf damage')]
        eligible, log = resolve_inventory(rows, [], pairs)
        self.assertEqual(len(eligible), 2)
        self.assertEqual(len({r['group_id'] for r in eligible}), 1)
        self.assertTrue(any('exact SHA-256' in r['reason'] for r in log))

    def test_leaking_exact_hash_is_detected(self):
        rows = fixture()
        assignment, _ = grouped_split(rows, 11, trials=2)
        first = rows[0]
        second = next(r for r in rows if assignment[r['group_id']] != assignment[first['group_id']])
        second['sha256'] = first['sha256']
        with self.assertRaisesRegex(ValueError, 'leakage'):
            validate_split(rows, assignment)

    def test_loader_rgb_labels_transform_and_hash(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            Image.new('L', (12, 9), 100).save(root / 'leaf.png')
            from prepare_dataset import sha_file
            r = dict(path='leaf.png', image_id='one', disease='brown_spot', disease_id=0,
                     background='field', background_id=0, sha256=sha_file(root/'leaf.png'))
            write_csv(root/'train.csv', [r])
            ds = RiceDataset(root, root/'train.csv', verify_hashes=True)
            self.assertEqual(ds[0]['image'].mode, 'RGB')
            self.assertEqual(ds[0]['background_id'], 0)
            ds = RiceDataset(root, root/'train.csv', transform=lambda im: im.size)
            self.assertEqual(ds[0]['image'], (12, 9))
            Image.new('RGB', (12, 9), 'green').save(root/'leaf.png')
            with self.assertRaisesRegex(ValueError, 'changed'):
                RiceDataset(root, root/'train.csv', verify_hashes=True)


if __name__ == '__main__':
    unittest.main()
