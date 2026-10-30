# Converting COCO (CVAT) video folders to LaSOT format

This guide explains how to turn your COCO-annotated video folders into the
LaSOT layout SUTrack trains on, using `tools/coco2lasot.py`, and how to launch
training with the result. It is written for an **offline machine**: no network
access is required for any step.

## Expected input layout

A root folder that contains one subfolder per video. Each video subfolder is a
standard CVAT COCO 1.1 export:

```
my_dataset/                     <-- --src
├── video_001/                  <-- one subfolder per video
│   ├── annotations/
│   │   └── instances_default.json
│   └── images/
│       ├── 000001.png
│       ├── 000002.png
│       └── ...
├── video_002/
│   └── ...
```

`instances_default.json` is a COCO annotations file: `images` entries with
`id`, `file_name`, `width`, `height`, and `annotations` entries with
`image_id`, `bbox` (`[x, y, w, h]`), `category_id`. One box per frame is used;
frames with several boxes take the first non-crowd box. Frames with no
annotation, annotations whose image file is missing, unreadable images, and
empty/degenerate boxes are skipped and counted in the conversion report.

## Convert

```bash
python tools/coco2lasot.py --src /path/to/my_dataset --out data/lasot_custom
```

Run from the repo root (use your training environment's python). Output:

```
data/lasot_custom/
├── <class>/<class>-<video>/img/00000001.jpg   # frames renumbered 1..N, contiguous
│                      /groundtruth.txt        # x,y,w,h per line, one per frame
│                      /full_occlusion.txt     # all zeros (COCO has no occlusion flag)
│                      /out_of_view.txt        # all zeros
│                      /nlp.txt                # "the <class> object."
├── lasot_custom_train_split.txt               # all converted sequence names
├── sequence_sources.json                      # sequence name -> source folder (collision safety)
└── conversion_report.json                     # per-video stats: kept/skipped frames, warnings
```

Notes on what the converter does:

- **Class folder name** comes from the majority annotation category of each
  video (overridable with `--class-name`; fallback `custom` when the JSON has
  no usable category). The class name becomes the LaSOT "class" level.
- **Sequence names** are sanitized to `lowercase_with_underscores` — the
  training loader splits `class-video` on the *first* hyphen, so any other
  hyphens in your folder names are replaced with underscores.
- **Short videos** (< 20 usable frames) are skipped: the training sampler
  cannot draw a training pair from them. The threshold matches the sampler
  minimum and can be changed with `--min-frames` (not recommended).
- **Boxes** are clamped to the image bounds; frames whose box is empty after
  clamping are skipped. Pass `--no-clamp` to keep raw coordinates instead.

### Image modes

Default `transcode` re-encodes every frame as a real JPEG (quality 95) — the
loader hard-codes `.jpg` names, and real JPEGs keep the dataset self-contained
and portable. Alternatives:

| Mode | What it does | When to use |
|------|--------------|-------------|
| `transcode` (default) | decode + re-encode as JPEG q95 via cv2 (Pillow fallback) | default; needs `cv2` **or** `PIL` in the env |
| `copy` | copy raw file bytes under `00000001.jpg` names | no re-encode; works with stdlib only; cv2-based readers still decode PNG content by magic bytes |
| `link` | relative symlinks to the source frames | cheapest on disk; the source tree must stay in place |

`--workers N` converts N videos in parallel.

### Re-runs, collisions and resume

- Converted videos are skipped on re-run (reported as `exists`) unless
  `--overwrite` is passed — add more video folders to `--src` and re-run
  freely. **The split file is rebuilt from what is complete on disk after
  every run**, so previously converted sequences are never dropped from it.
- Sanitization maps e.g. `drone-1` and `drone_1` to the same sequence name;
  the registry `sequence_sources.json` remembers which source folder produced
  each sequence, and a colliding source automatically gets a `_2`, `_3`…
  suffix instead of clobbering the existing one. Keep `sequence_sources.json`
  next to the dataset.
- A sequence folder without `groundtruth.txt` (leftover of an interrupted
  run) is rebuilt, never treated as done.
- Re-running when everything already exists exits `0` (nothing new, nothing
  failed). Exit `1` means some video failed or nothing usable was produced.

## Train

1. Copy the generated split file where the loader looks for it (it is
   gitignored, so it must be copied on each machine):

   ```bash
   cp data/lasot_custom/lasot_custom_train_split.txt lib/train/data_specs/
   ```

2. Put the pretrained encoder weights where the config expects them
   (`pretrained/itpn/fast_itpn_base_clipl_e1600.pt` — this file **is
   required**; the backbone build `torch.load`s it, see `MODEL.ENCODER.PRETRAIN_TYPE`.
   Download per the repo README / MODEL_ZOO.md).

   For a true finetune from the released SUTrack-B384 tracker (recommended
   over backbone-only init on a small dataset), also download that checkpoint
   and set `TRAIN.PRETRAINED_PATH:` in the yaml to its path. Loading is
   non-strict, so the CLIP weights inside the official checkpoint are simply
   skipped now that language is disabled.

3. Launch training (`--save_dir` is required — checkpoints and logs go under it):

   ```bash
   python lib/train/run_training.py --script sutrack --config sutrack_b384_custom --save_dir .
   ```

   `experiments/sutrack/sutrack_b384_custom.yaml` is a copy of
   `sutrack_b384.yaml` with the dataset switched to `LASOT_CUSTOM`
   (single dataset, ratio 1, 30 000 samples/epoch, 60 epochs, LR drop at 48,
   test at 60 — a finetune-scale schedule). The dataset root is
   `data/lasot_custom` relative to the repo (`lasot_custom_dir` in
   `lib/train/admin/local.py`), so the repo can be moved between machines
   without editing paths.

   With `--save_dir .`, checkpoints land in `./checkpoints/train/sutrack/sutrack_b384_custom/`
   and logs in `./logs/` (both gitignored). Resume after an interruption by
   re-running the same command — the trainer picks up the latest
   `SUTRACK_ep*.pth.tar` automatically.

### CLIP and task recognition are disabled in this config

`sutrack_b384_custom.yaml` ships with both extras turned off, since a custom
single-domain dataset needs neither:

| Key (in the yaml) | What it controls | Off value |
|---|---|---|
| `DATA.MULTI_MODAL_LANGUAGE` | builds the CLIP text encoder and feeds a text token to the encoder at train time (`lib/models/sutrack/sutrack.py:125`) | `False` — CLIP is never constructed, so CLIP weights are never loaded (important offline) |
| `MODEL.TEXT_ENCODER.TYPE` | which CLIP model to load — inactive once the line above is `False` | (left as-is, unused) |
| `TRAIN.TASK_CE_WEIGHT` | weight of the task-classification loss (`lib/train/train_script.py:72`) | `0.0` — the task head gets zero gradient and its output is unused |
| `TRAIN.TYPE` | optimizer param grouping; `text_frozen` only froze CLIP params, pointless without CLIP | `normal` |
| `TEST.MULTI_MODAL_LANGUAGE.DEFAULT` | language at inference (`lib/test/tracker/sutrack.py:67`) | `False` |

Two things intentionally stay as they are:

- **`MODEL.TASK_DECODER` / `MODEL.TASK_NUM` / `MODEL.TASK_INDEX`** — the small
  task head is always built (no yaml gate for it) and the training actor reads
  `MODEL.TASK_INDEX[<dataset>]` to build labels, so removing these keys
  crashes training. With `TASK_CE_WEIGHT: 0` the head is dead weight (<0.1M
  params, one MLP on one token) and keeps the model compatible with official
  SUTrack checkpoints if you ever finetune from one.
- **`DATA.MULTI_MODAL_VISION: True`** — not CLIP; this is the 6-channel
  (duplicated-RGB) input the b384 encoder architecture expects. Leave it on.

### GPU memory note

`TRAIN.BATCH_SIZE: 32` assumes a large training GPU. On an 8GB card
(e.g. RTX 5060) b384 training OOMs above batch 1–2; lower `BATCH_SIZE`
accordingly (and consider `MODEL.ENCODER.USE_CHECKPOINT: True` if you need
more). Verified: batch 1 forward+backward fits in ~7GB.

4. (Optional) verify the pipeline before a long run:

   ```bash
   /path/to/env/python -m pytest tools/tests/test_coco2lasot.py -v
   ```

   The suite covers the converter, the repo wiring, and an end-to-end run
   against a mock dataset built from real COCO 2017 images in `data/`.

## Troubleshooting

- **`failed: 0, ok: 0, skipped: ...`** — inspect `conversion_report.json`;
   the usual cause is videos under 20 usable frames or annotations whose
   `file_name` does not match files in `images/`.
- **Training hangs at startup with no progress** — a sequence in the split
   file has fewer than 20 frames or its flag files are missing; the sampler
   retries forever instead of failing. Re-check the report.
- **`FileNotFoundError: .../lasot_custom_train_split.txt`** — step 1 of
  *Train* was skipped on this machine.
