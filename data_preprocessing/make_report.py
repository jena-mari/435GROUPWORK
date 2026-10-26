"""Create the dataset inspection report from an existing audit and split version."""
import argparse
import html
import json
from collections import Counter
from pathlib import Path

from prepare_dataset import CLASSES, SEEDS, read_csv


def table(headers, rows):
    return '<table><thead><tr>' + ''.join(f'<th>{html.escape(str(x))}</th>' for x in headers) + \
           '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{html.escape(str(x))}</td>'
           for x in row) + '</tr>' for row in rows) + '</tbody></table>'


def generate(output, version):
    summary = json.loads((output / 'audit_summary.json').read_text())
    provenance = json.loads((version / 'provenance.json').read_text())
    rows = read_csv(version / 'eligible_inventory.csv')
    raw = read_csv(output / 'inventory.csv')
    counts = Counter((r['disease'], r['background']) for r in rows)
    groups = Counter(r['group_id'] for r in rows)
    largest, largest_size = groups.most_common(1)[0]
    split_counts = read_csv(version / 'split_summary.csv')
    splits_table = []
    for seed in SEEDS:
        counts_by_part = {r['partition']: int(r['images']) for r in split_counts if int(r['seed']) == seed}
        splits_table.append([seed, counts_by_part['train'], counts_by_part['validation'], counts_by_part['test']])
    n = len(rows)
    all_counts = [[d.replace('_', ' ').title(), counts[d, 'field'], counts[d, 'white'],
                   counts[d, 'field'] + counts[d, 'white']] for d in CLASSES]
    field = sum(r['background'] == 'field' for r in rows)
    white = n - field
    all_counts.append(['Total', field, white, n])
    description = f'''III. Problem Description

The task is to identify one of five rice diseases from an RGB photograph: Brown Spot, Leaf Scald, Rice Blast, Rice Tungro or Sheath Blight. The input may show a leaf in the field or against a white background. Each image receives one disease label, so the task is treated as five-class image classification. The experiment assumes that one of these five diseases is present; it does not evaluate healthy leaves, other diseases or the location of individual lesions.

The audited copy of Dhan-Shomadhan [1] contains {n:,} images: {field} field images and {white} white-background images. Brown Spot has the fewest images (139), while Sheath Blight has the most (283). Figure 1 shows one example of each disease under both backgrounds. In the field examples, overlapping leaves, soil, water and other vegetation make the affected leaf harder to isolate. The white-background examples show the leaf more clearly, although shadows, lighting gradients and leaf placement still vary. Symptoms also differ in size and appearance. Resizing can obscure small lesions, and cropping can remove damage near the leaf tip or edge. These differences suggest that classification needs to account for local appearance as well as overall colour.

The planned comparison uses a handcrafted baseline, a frozen pretrained ResNet-18 and a partly fine-tuned ResNet-18. This will test whether transferred visual features help distinguish the classes and whether adapting the final residual stage adds a benefit. The larger white-background subset could dominate an overall score, so each method will be evaluated on field, white-background and combined test images. Macro-F1 will be used alongside per-class results to account for unequal class counts. Related and unresolved similarity-candidate images are kept in the same partition to reduce leakage between training and evaluation. Supplied disease labels are retained; the audit checks their consistency rather than independently confirming diagnoses.
'''
    approach = f'''IV.1. Shared Dataset Preparation

Images were listed in a fixed sorted order. Their original paths were retained, while explicit aliases mapped spelling variants and trailing spaces to one disease/background label scheme. The supplied CSV was checked against filenames and folder labels; {summary['metadata_issues']} disagreements were found. All {n:,} files passed stream verification and full decoding. Image dimensions were 1952 x 4160 for {summary['dimensions'].get('1952x4160', 0)} images, 4160 x 1952 for {summary['dimensions'].get('4160x1952', 0)}, and 1536 x 3264 for {summary['dimensions'].get('1536x3264', 0)}. The loader applies EXIF orientation handling and converts images to RGB. Original files were left unchanged.

Full-file SHA-256 checks found {summary['exact_duplicate_groups']} byte-identical duplicate groups. Possible related images were then screened using 64-bit difference hashes (dHash) and perceptual hashes (pHash). A pair was flagged if its dHash Hamming distance was at most 6 or its pHash distance was at most 8. This produced {summary['near_duplicate_candidates']:,} candidate pairs. These thresholds were used for screening before model evaluation; they were not calibrated to identify the same leaf. A sample of the strongest pairs and selected class contact sheets contained both repeated views and similar leaf outlines on white backgrounds. Four pairs were recorded as visually supported related views. The remaining {provenance['unresolved_pairs_grouped']:,} candidates were kept together under the conservative grouping policy.

Images linked by candidate relationships were assigned to the same group, including links through other images. The retained dataset contains {len(groups)} groups. Keeping an unresolved pair together reduces the chance that a repeated view appears in both training and testing. An unrelated pair may also be joined, which reduces flexibility in the split but retains both images. The comparison uses this precaution because leakage could inflate the scores of all three methods. No images were removed on the basis of perceptual similarity.

This policy has a substantial cost. Its largest connected group contains {largest_size} white-background images ({largest_size / n:.1%} of the dataset) and is assigned to training in every run. Those images never contribute to the test scores, limiting coverage of the dataset. The group is not evidence that all its images show the same leaf. Full pairwise review remains unfinished, and plant/session identifiers were not supplied. Both missed relationships and unnecessary grouping remain possible.

Five repeated random holdouts were saved using seeds 11, 22, 33, 44 and 55, with target proportions of 70% training, 15% validation and 15% testing. Whole groups were allocated using the best of 200 seeded greedy starts, minimising squared deviations from the ten disease/background targets, weighted by target counts. Table 2 reports the achieved counts. Every partition contains all five diseases under both backgrounds. Within a run, every eligible image appears once and no image group or exact file hash crosses partitions. Images can move between partitions across runs; this is repeated holdout evaluation rather than five-fold cross-validation.

The baseline and both ResNet-18 variants use the same saved manifests. The loader returns an image, disease ID, background ID and image ID. Background is kept for evaluation and is not a classifier input. Model-specific transforms are applied after partitioning, with random augmentation restricted to training images. Scaling, PCA, feature selection and any other fitted preprocessing must use only the corresponding training partition. Background removal, lesion cropping, denoising and colour thresholding were not applied as shared preprocessing steps; their effect would need a separate experiment.
'''
    algorithm = '''Algorithm 1. Auditing and conservative generation of five shared holdouts

Input: dataset root; seeds {11,22,33,44,55}; target ratios {0.70,0.15,0.15}.
1. Enumerate source images in sorted order; map explicit folder aliases to fixed IDs.
2. Verify and decode each image. Record source path/labels, dimensions, SHA-256,
   dHash64, pHash64 and descriptive quality proxies. Check supplied metadata.
3. Flag unreadable/unknown labels and conflicting exact-copy annotations.
   Resolve conflicts explicitly. Retain one copy per unambiguous exact-copy set.
4. Flag candidate pairs if dHash distance <=6 OR pHash distance <=8.
5. Record any visual pair decisions and additional manual grouping information.
6. Under the selected conservative policy, retain both images of every unresolved
   candidate pair and join their groups. Relatedness is not assumed to be proven.
   Compute connected components, including transitive links and manual groups.
7. For each seed, allocate whole groups to training, validation and testing using
   200 seeded greedy starts and retain the valid allocation nearest stratum targets.
8. Verify exact coverage, group/hash separation and all ten strata per partition.
9. Save three manifests, achieved counts, fixed mappings, review decisions and
   provenance for each run under the main dataset directory.
10. Load the same saved manifests for all three methods. Fit any data-derived
    preprocessing on training images only; augment training images only.
'''
    (output / 'report_sections.txt').write_text(description + '\n' + approach + '\n' + algorithm +
        '\n[1] M. F. Hossain, S. Abujar, S. R. H. Noori, and S. A. Hossain, '
        '"Dhan-Shomadhan: A Dataset of Rice Leaf Disease Classification for Bangladeshi Local Rice," '
        'Mendeley Data, V1, 2021, doi: 10.17632/znsxdctwtt.1.\n')
    challenges = [
        ['Clutter and overlapping leaves', 'Field examples bsf_0, lsf_0 and rbf_0 include soil, water and intersecting leaves.',
         'Preserve context initially; compare features/models on the same raw inputs.'],
        ['Illumination and colour', 'Examples show green/yellow/brown variation; white paper has shadows and uneven illumination.',
         'Inspect colour augmentation for plausibility; apply it to training only.'],
        ['Narrow leaves and small symptoms', 'Portrait leaves occupy a small fraction of many images; several symptoms are near tips or edges.',
         'Check downsampling and cropping before fixing transforms; avoid destructive preprocessing by default.'],
        ['Class and background imbalance', f'{field} field versus {white} white images; 139 Brown Spot versus 283 Sheath Blight.',
         'Stratify by disease/background; report macro-F1, per-class support and three background scenarios.'],
        ['Repeated imagery', 'Candidate review includes repeated field views such as rtf_17/18 and rbf_21/22.',
         'Keep candidate-connected groups intact within each run.'],
        ['Annotation uncertainty', 'Folder aliases differ in spelling. Supplied CSV agrees with folders after explicit mapping.',
         'Keep original labels and mapping; agreement does not establish correct pathology.'],
        ['Quality proxies', 'Inventory records luminance mean/SD and resized Laplacian variance for every image.',
         'Use for inspection; no automatic blur/brightness-based exclusions.']]
    label_table = [[d.replace('_', ' ').title(), i] for d, i in CLASSES.items()]
    parts = ['<!doctype html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        '<title>Dhan-Shomadhan dataset inspection</title><style>',
        'body{font:16px/1.6 system-ui,sans-serif;color:#20332d;background:#f4f6f2;margin:0}',
        'main{max-width:1060px;margin:35px auto;background:white;padding:42px;border-radius:16px}',
        'h1,h2{line-height:1.2;color:#164835}h1{font-size:36px}h2{margin-top:2em}',
        '.tag{font-size:13px;letter-spacing:.06em;text-transform:uppercase;color:#527963}',
        '.note{padding:18px 22px;background:#fff4df;border-left:5px solid #b87a20}',
        'table{border-collapse:collapse;width:100%;margin:20px 0;font-size:15px}',
        'th,td{text-align:left;padding:10px 14px;border-bottom:1px solid #d9e3dc}',
        'th{background:#edf4ee}img{max-width:100%;height:auto}.caption{color:#53675d;font-size:14px}',
        'a{color:#096c4a}pre{white-space:pre-wrap;background:#f1f5f1;padding:20px}',
        'details{margin:18px 0}summary{cursor:pointer;font-weight:bold}',
        '@media(max-width:700px){main{margin:0;padding:20px}table{font-size:12px}th,td{padding:7px}}',
        '@media print{body{background:white}main{margin:0;padding:0}details{display:block}}',
        '</style></head><body><main><div class="tag">CSCI435 · dataset preparation</div>',
        '<h1>Dhan-Shomadhan<br>inspection and shared splits</h1>',
        f'<p>{n:,} images · 5 disease classes · 2 backgrounds · 5 reproducible holdouts</p>',
        '<p class="note"><strong>Selected policy: conservative grouping.</strong> '
        'Unresolved similarity candidates remain in the dataset and are kept together. '
        f'The largest connected group has {largest_size} images and is in training in every run. '
        'This reduces candidate-related leakage risk, but limits test coverage. '
        'Visual review covered a sample of the candidate pairs.</p>',
        '<h2>1. Verified dataset counts</h2>', table(['Disease', 'Field', 'White', 'Total'], all_counts),
        f'<p>All {len(raw):,} files decode; {summary["exact_duplicate_groups"]} exact-copy groups; '
        f'{summary["metadata_issues"]} metadata/folder disagreements. These checks establish file '
        'and annotation consistency, not independent biological label verification.</p>',
        '<h2>2. Sample-image figure</h2><img src="samples_raw.png" alt="One image per disease and background">',
        '<p class="caption">Figure 1. First readable image in sorted path order for each disease/background '
        'stratum. Original aspect ratios are preserved; no lesion crops or background removal are applied. '
        'These are dataset examples, not an independent confirmation of disease.</p>',
        '<h2>3. Observed image challenges</h2>', table(['Challenge', 'Inspection evidence', 'Implication'], challenges),
        '<h2>4. Audit and near-duplicate policy</h2>',
        f'<p>The screen produced {summary["near_duplicate_candidates"]:,} pairs. Four visually supported '
        f'related pairs were recorded; {provenance["unresolved_pairs_grouped"]:,} unresolved pairs were '
        f'conservatively joined. The retained {n:,} images form {len(groups)} connected groups. '
        'No perceptually similar images were deleted. <a href="review.html">Inspect candidate pairs</a>; '
        '<a href="pair_review.csv">original review decisions</a>; '
        f'<a href="{version.relative_to(output).as_posix()}/effective_pair_decisions.csv">effective grouping decisions</a>.</p>',
        '<p>The candidate rule is dHash64 Hamming distance ≤6 OR pHash64 distance ≤8. '
        'It was fixed before model evaluation and was not calibrated as a same-leaf detector. '
        'Crops, rotations and changes of background may evade this rule. Candidate connectivity '
        'can also merge unrelated leaves through chains of similar silhouettes.</p>',
        '<h2>5. Five shared holdouts</h2>', table(['Seed', 'Training', 'Validation', 'Testing'], splits_table),
        '<p>Targets are 70/15/15. Every partition includes all ten disease/background combinations; '
        'no image, group or exact file hash crosses partitions within a run. The generator uses '
        '200 seeded greedy starts per run. Saved counts are the achieved allocation; the heuristic '
        'does not guarantee the global optimum. These are repeated random holdouts, not five-fold CV.</p>',
        f'<p><a href="{version.relative_to(output).as_posix()}/split_summary.csv">Detailed split summary</a> · '
        f'<a href="{version.relative_to(output).as_posix()}/provenance.json">Provenance and checksums</a></p>',
        '<h2>6. Loading images</h2><p>The shared loader returns an RGB image, disease ID, '
        'background ID and image ID. Background is evaluation metadata. Supply model-specific '
        'transforms at loading time, with augmentation only for training. Fit any scalers, PCA, '
        'feature selection or other learned preprocessing on each training partition only. '
        'The original images remain unchanged.</p>', table(['Disease', 'Integer ID'], label_table),
        '<h2>7. Report sections</h2><p><a href="report_sections.txt">Open the plain-text draft</a>. '
        'The Problem Description is drafted for the one-page section; placement of tables '
        'and the figure must be checked in the final report layout.</p>']
    for name, text in [('Problem Description', description), ('Dataset Preparation and justification', approach),
                        ('Revised algorithm', algorithm)]:
        parts.append(f'<details><summary>{name}</summary><pre>{html.escape(text)}</pre></details>')
    parts += ['<h2>8. Reproduction and limits</h2><p>Use Python 3.12, NumPy 2.3.5 and Pillow 12.3.0. '
        'Run audit, then build with --conservative-pending. Preserve original paths, all source images '
        'and the recorded reviews. All three methods must consume the same manifest version. '
        'Preprocessing tests cover allocation, duplicate resolution, rerun source selection and loading. '
        'PyTorch batch loading and model training have not yet been tested.</p>',
        '<p>Visual inspection covered the 10-image figure, the 32 strongest candidate pairs, and '
        'selected contact sheets from every disease. It was not an exhaustive examination of every '
        'pair at native resolution. No field deployment performance or independent plant-level '
        'generalisation follows from this audit.</p>',
        '<p>[1] M. F. Hossain, S. Abujar, S. R. H. Noori, and S. A. Hossain, '
        '“Dhan-Shomadhan: A Dataset of Rice Leaf Disease Classification for Bangladeshi Local Rice,” '
        'Mendeley Data, V1, 2021. <a href="https://doi.org/10.17632/znsxdctwtt.1">doi:10.17632/znsxdctwtt.1</a>.</p>',
        '</main></body></html>']
    (output / 'dataset_report.html').write_text(''.join(parts), encoding='utf-8')
    print(output / 'dataset_report.html')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('audit', type=Path)
    p.add_argument('version', type=Path)
    args = p.parse_args()
    generate(args.audit.resolve(), args.version.resolve())
