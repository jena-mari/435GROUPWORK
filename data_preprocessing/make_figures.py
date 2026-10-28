"""Rebuild audit figures from the saved inventory, reviews and thumbnails.

Run from the repository root:
    python data_preprocessing/make_figures.py
This does not change image labels, review decisions or train/validation/test splits.
"""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from prepare_dataset import CLASSES, read_csv, sample_figure


def generate(audit):
    figures = audit / 'figures'
    figures.mkdir(exist_ok=True)
    rows = read_csv(audit / 'inventory.csv')
    sample_figure(rows, audit, 'samples_raw.png')
    version = audit / (audit / 'CURRENT_SPLIT_VERSION.txt').read_text().strip()
    sample_figure(read_csv(version / 'eligible_inventory.csv'), audit, 'samples_eligible.png')

    pairs = read_csv(audit / 'pair_review.csv')
    # Use original row order to break distance ties and preserve review row IDs.
    strongest = sorted(enumerate(pairs, 1),
                       key=lambda item: int(item[1]['dhash_distance']) + int(item[1]['phash_distance']))[:32]
    for start in range(0, len(strongest), 8):
        canvas = Image.new('RGB', (1440, 1280), 'white')
        draw = ImageDraw.Draw(canvas)
        for k, (number, pair) in enumerate(strongest[start:start + 8]):
            x, y = k % 4 * 360, k // 4 * 640
            draw.text((x + 4, y + 2),
                      f"Pair {number}: d{pair['dhash_distance']} p{pair['phash_distance']}", fill='black')
            for j, side in enumerate(('left', 'right')):
                with Image.open(audit / 'thumbnails' / f"{pair[side + '_id']}.jpg") as im:
                    im.thumbnail((174, 590))
                    canvas.paste(im, (x + j * 180, y + 25))
                draw.text((x + j * 180, y + 610), Path(pair[side + '_path']).name, fill='black')
        canvas.save(figures / f'pairs_strong_{start // 8:02d}.jpg', quality=95)

    for disease in CLASSES:
        pool = [row for row in rows if row['disease'] == disease and row['readable'] == '1']
        for start in range(0, len(pool), 48):
            canvas = Image.new('RGB', (1440, 1640), 'white')
            draw = ImageDraw.Draw(canvas)
            for k, row in enumerate(pool[start:start + 48]):
                x, y = k % 8 * 180, k // 8 * 273
                with Image.open(audit / 'thumbnails' / f"{row['image_id']}.jpg") as im:
                    im.thumbnail((176, 245))
                    canvas.paste(im, (x + (180 - im.width) // 2, y))
                draw.text((x + 2, y + 247), Path(row['path']).name, fill='black')
            canvas.save(figures / f'contact_{disease}_{start // 48:02d}.jpg', quality=95)
    print(f'Audit figures saved under {figures}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, default=Path('Dhan-Shomadhan/preprocessing'))
    args = parser.parse_args()
    generate(args.audit.resolve())
