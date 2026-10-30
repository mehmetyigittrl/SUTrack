#!/usr/bin/env python3
"""Convert CVAT-style COCO video folders into LaSOT layout for SUTrack training.

Input  -- a root folder of video subfolders, each shaped like a CVAT "COCO 1.0"
          export (one box per frame is the expected case):

    <src>/<video_name>/annotations/instances_default.json
    <src>/<video_name>/images/*.png

Output -- a LaSOT-layout root the training loader can read
(lib/train/dataset/lasot.py expects <class>/<class>-<video>/img/%08d.jpg plus
groundtruth.txt / full_occlusion.txt / out_of_view.txt):

    <out>/<class>/<class>-<video>/img/00000001.jpg ...
    <out>/<class>/<class>-<video>/groundtruth.txt      # x,y,w,h per line
    <out>/<class>/<class>-<video>/full_occlusion.txt   # 0,0,... (COCO has no occlusion labels)
    <out>/<class>/<class>-<video>/out_of_view.txt      # 0,0,...
    <out>/<class>/<class>-<video>/nlp.txt              # placeholder, read only at eval time
    <out>/lasot_custom_train_split.txt                 # feed to lib/train/data_specs/
    <out>/conversion_report.json

Rules:
  * frames without an annotation, annotations whose image file is missing on
    disk, crowd annotations and invalid/empty boxes are skipped; surviving
    frames are renumbered contiguously from 1.
  * sequence names must not contain '-' (the loader splits names on it), so
    '-' is replaced with '_' everywhere; class comes from the majority COCO
    category name (fallback: 'custom'), overridable with --class-name.
  * sanitization can map distinct source folders to one name ('a-b' and
    'a_b'); the registry <out>/sequence_sources.json remembers which source
    produced each sequence, and a colliding source gets a '_2', '_3'...
    suffix instead of clobbering it. Sequence dirs without groundtruth.txt
    (interrupted runs) are rebuilt, never treated as done.
  * the split file is rebuilt from what is complete on disk after every run,
    so adding videos and re-running keeps all previously converted sequences.
  * sequences with fewer than --min-frames usable frames are skipped: the
    training sampler silently rejects them (>= 20 frames, > 6 visible for
    b384) and can hang the data loader.
  * frames must be named .jpg (hard-coded in the loader), so the default
    --image-mode transcode re-encodes to real JPEG (needs cv2 or Pillow).
    --image-mode copy keeps original bytes under .jpg names (fine for the
    cv2-based training reader), --image-mode link symlinks instead of copying.

The script only needs the Python standard library for copy/link modes.

Example:
    python tools/coco2lasot.py --src /path/to/videos --out data/lasot_custom
    cp data/lasot_custom/lasot_custom_train_split.txt lib/train/data_specs/
    python lib/train/run_training.py --script sutrack --config sutrack_b384_custom --save_dir .
"""
import argparse
import json
import math
import os
import re
import shutil
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SPLIT_NAME = "lasot_custom_train_split.txt"
DEFAULT_REPORT_NAME = "conversion_report.json"
DEFAULT_REGISTRY_NAME = "sequence_sources.json"


@dataclass
class ConvertOptions:
    image_mode: str = "transcode"          # copy | link | transcode
    jpeg_quality: int = 95
    min_frames: int = 20                   # sampler gate for sutrack_b384 (>= 20 frames)
    class_name: str = None                 # override class; default: majority category
    fallback_class: str = "custom"
    overwrite: bool = False
    annotations_subdir: str = "annotations"
    annotations_name: str = "instances_default.json"
    images_subdir: str = "images"
    split_name: str = DEFAULT_SPLIT_NAME
    report_name: str = DEFAULT_REPORT_NAME
    clamp: bool = True                     # clamp boxes to image bounds
    transcode_backend: str = "auto"        # auto | cv2 | PIL
    workers: int = 1


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def sanitize_name(name):
    """Make a filesystem/split-safe LaSOT token: [a-z0-9_], no hyphens.

    The training loader splits sequence names on '-' (class-vid), so hyphens
    anywhere in class or video names break path construction.
    """
    s = re.sub(r"[^a-z0-9_]+", "_", str(name).lower())
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _clamp_box(bbox, width, height):
    """Clamp [x, y, w, h] to the image; return None-free floats (caller checks emptiness)."""
    x, y, w, h = bbox
    if width and height and _is_num(width) and _is_num(height):
        x0 = min(max(x, 0.0), float(width))
        y0 = min(max(y, 0.0), float(height))
        x1 = min(max(x + w, 0.0), float(width))
        y1 = min(max(y + h, 0.0), float(height))
        return [x0, y0, x1 - x0, y1 - y0]
    return [float(x), float(y), float(w), float(h)]


def _safe_image_path(images_dir, file_name):
    """Resolve an images entry file_name, refusing paths escaping images_dir."""
    p = Path(os.path.normpath(str(images_dir / str(file_name))))
    base = Path(os.path.normpath(str(images_dir)))
    if base not in p.parents:
        return None
    return p


class _Transcoder:
    """Lazy image re-encoder; cv2 preferred, Pillow fallback."""

    def __init__(self, backend="auto", jpeg_quality=95):
        self.backend = backend
        self.quality = int(jpeg_quality)
        self._cv2 = None
        self._pil = None

    def _resolve(self):
        if self._cv2 is not None or self._pil is not None:
            return
        if self.backend == "cv2":
            import cv2
            self._cv2 = cv2
        elif self.backend == "PIL":
            from PIL import Image
            self._pil = Image
        else:  # auto
            try:
                import cv2
                self._cv2 = cv2
            except ImportError:
                from PIL import Image
                self._pil = Image

    def write_jpeg(self, src, dst):
        """Re-encode src to a real JPEG at dst. Returns False if unreadable."""
        self._resolve()
        try:
            if self._cv2 is not None:
                img = self._cv2.imread(str(src))
                if img is None:
                    return False
                params = [self._cv2.IMWRITE_JPEG_QUALITY, self.quality]
                ok, buf = self._cv2.imencode(".jpg", img, params)
                if not ok:
                    return False
                dst.write_bytes(buf.tobytes())
            else:
                with self._pil.open(str(src)) as im:
                    im.convert("RGB").save(str(dst), "JPEG", quality=self.quality)
        except Exception:
            return False
        return True


# ---------------------------------------------------------------------------
# per-video conversion
# ---------------------------------------------------------------------------
def _pick_class_name(data, anns_by_img, opts):
    if opts.class_name:
        return sanitize_name(opts.class_name) or sanitize_name(opts.fallback_class) or "custom"
    categories = {c.get("id"): c for c in data.get("categories") or []}
    counts = Counter()
    for anns in anns_by_img.values():
        for a in anns:
            counts[a.get("category_id")] += 1
    if counts:
        winner = counts.most_common(1)[0][0]
        name = sanitize_name((categories.get(winner) or {}).get("name") or "")
        if name:
            return name
    return sanitize_name(opts.fallback_class) or "custom"


def _parse_bbox(bbox):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    if not all(_is_num(v) for v in bbox):
        return None
    return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]


def convert_folder(src_dir, out_root, opts=ConvertOptions(), registry=None, taken=None, lock=None):
    """Convert one CVAT-style video subfolder into one LaSOT sequence.

    registry maps seq_name -> source folder name (persisted by convert_dataset
    as sequence_sources.json) so re-runs recognize their own output and name
    collisions between distinct sources ('a-b' vs 'a_b' sanitize identically)
    get a '_2', '_3'... suffix instead of overwriting each other.
    taken/lock make allocation safe with --workers > 1.
    """
    src_dir, out_root = Path(src_dir), Path(out_root)
    if registry is None:
        registry = {}
    if taken is None:
        taken = set()
    if lock is None:
        lock = threading.Lock()
    res = {
        "source": src_dir.name,
        "status": "failed",
        "class": None,
        "sequence": None,
        "frames_total": 0,
        "frames_used": 0,
        "skipped_missing_img": 0,
        "skipped_no_ann": 0,
        "skipped_invalid_box": 0,
        "skipped_unreadable": 0,
        "multi_ann_frames": 0,
        "warnings": [],
    }

    json_path = src_dir / opts.annotations_subdir / opts.annotations_name
    if not json_path.is_file():
        res["warnings"].append("missing %s" % json_path.relative_to(src_dir))
        return res
    try:
        with open(json_path) as f:
            data = json.load(f)
    except Exception as e:
        res["warnings"].append("bad json: %s" % e)
        return res

    images = sorted(data.get("images") or [], key=lambda im: im.get("id", 0))
    res["frames_total"] = len(images)

    # non-crowd annotations grouped per image id, deterministic order (by ann id)
    anns_by_img = {}
    for a in data.get("annotations") or []:
        if a.get("iscrowd", 0) != 0:
            continue
        anns_by_img.setdefault(a.get("image_id"), []).append(a)
    for anns in anns_by_img.values():
        anns.sort(key=lambda a: (a.get("id") is None, a.get("id")))

    cls = _pick_class_name(data, anns_by_img, opts)
    vid = sanitize_name(src_dir.name) or "video"
    base_seq = "%s-%s" % (cls, vid)
    with lock:
        seq_name, suffix = base_seq, 2
        while True:
            owner = registry.get(seq_name)
            if seq_name not in taken and (owner is None or owner == src_dir.name):
                taken.add(seq_name)
                break
            seq_name = "%s_%d" % (base_seq, suffix)
            suffix += 1
    res["class"], res["sequence"] = cls, seq_name

    images_dir = src_dir / opts.images_subdir
    candidates = []  # (src_path, [x, y, w, h])
    for im in images:
        anns = anns_by_img.get(im.get("id")) or []
        if not anns:
            res["skipped_no_ann"] += 1
            continue
        if len(anns) > 1:
            res["multi_ann_frames"] += 1
        bbox = _parse_bbox(anns[0].get("bbox"))
        if bbox is not None and opts.clamp:
            bbox = _clamp_box(bbox, im.get("width"), im.get("height"))
        if bbox is None or bbox[2] <= 0 or bbox[3] <= 0:
            res["skipped_invalid_box"] += 1
            continue
        src_file = _safe_image_path(images_dir, im.get("file_name", ""))
        if src_file is None or not src_file.is_file():
            res["skipped_missing_img"] += 1
            continue
        candidates.append((src_file, bbox))

    if len(candidates) < opts.min_frames:
        res["status"] = "skipped_short"
        res["frames_used"] = len(candidates)
        return res

    seq_dir = out_root / cls / seq_name
    if seq_dir.exists():
        # a dir without groundtruth.txt is an interrupted run's leftover:
        # always rebuild it, never report it 'exists'
        complete = (seq_dir / "groundtruth.txt").is_file()
        if complete and not opts.overwrite:
            res["status"] = "exists"
            res["frames_used"] = len(candidates)
            if registry.get(seq_name) is None:
                # dir predates the registry; assume it is this same video so
                # future runs stay stable
                registry[seq_name] = src_dir.name
            return res
        shutil.rmtree(seq_dir)

    (seq_dir / "img").mkdir(parents=True, exist_ok=True)
    transcoder = _Transcoder(opts.transcode_backend, opts.jpeg_quality)

    rows = []
    for src_file, bbox in candidates:
        dst = seq_dir / "img" / ("%08d.jpg" % (len(rows) + 1))
        try:
            if opts.image_mode == "copy":
                shutil.copyfile(src_file, dst)
            elif opts.image_mode == "link":
                os.symlink(os.path.relpath(src_file, dst.parent), dst)
            else:  # transcode
                if not transcoder.write_jpeg(src_file, dst):
                    res["skipped_unreadable"] += 1
                    continue
        except OSError as e:
            res["warnings"].append("image %s: %s" % (src_file.name, e))
            continue
        rows.append(bbox)

    if len(rows) < opts.min_frames:  # unreadables may have shrunk it
        shutil.rmtree(seq_dir, ignore_errors=True)
        res["status"] = "skipped_short"
        res["frames_used"] = len(rows)
        return res

    with open(seq_dir / "groundtruth.txt", "w") as f:
        for x, y, w, h in rows:
            f.write("%.2f,%.2f,%.2f,%.2f\n" % (x, y, w, h))
    (seq_dir / "full_occlusion.txt").write_text(",".join(["0"] * len(rows)) + "\n")
    (seq_dir / "out_of_view.txt").write_text(",".join(["0"] * len(rows)) + "\n")
    (seq_dir / "nlp.txt").write_text("the %s object.\n" % cls)  # placeholder, comma-free

    res["status"] = "ok"
    res["frames_used"] = len(rows)
    with lock:
        registry[seq_name] = src_dir.name
    return res


# ---------------------------------------------------------------------------
# whole-dataset conversion
# ---------------------------------------------------------------------------
def _scan_sequences(out_root):
    """Sequence names that exist complete on disk, sorted and deduplicated.

    The split file is rebuilt from this scan (not from this run's results) so
    incremental runs -- where earlier videos report 'exists' -- cannot shrink
    the training set.
    """
    names = set()
    for cls_dir in out_root.iterdir():
        if not cls_dir.is_dir() or cls_dir.name.startswith("."):
            continue
        for seq_dir in cls_dir.iterdir():
            if (seq_dir.is_dir() and "-" in seq_dir.name
                    and (seq_dir / "groundtruth.txt").is_file()):
                names.add(seq_dir.name)
    return sorted(names)


def convert_dataset(src_root, out_root, opts=ConvertOptions()):
    """Convert every video subfolder under src_root into a LaSOT tree."""
    src_root, out_root = Path(src_root), Path(out_root)
    if not src_root.is_dir():
        raise NotADirectoryError("source root not found: %s" % src_root)
    out_root.mkdir(parents=True, exist_ok=True)

    registry_path = out_root / DEFAULT_REGISTRY_NAME
    registry = {}
    if registry_path.is_file():
        try:
            registry = json.loads(registry_path.read_text())
        except Exception:
            registry = {}
    taken, lock = set(), threading.Lock()

    subdirs = sorted(p for p in src_root.iterdir()
                     if p.is_dir() and not p.name.startswith("."))
    if opts.workers > 1:
        with ThreadPoolExecutor(max_workers=opts.workers) as ex:
            results = list(ex.map(
                lambda d: convert_folder(d, out_root, opts, registry, taken, lock), subdirs))
    else:
        results = [convert_folder(d, out_root, opts, registry, taken, lock)
                   for d in subdirs]

    summary = {
        "ok": sum(r["status"] == "ok" for r in results),
        "skipped": sum(r["status"] == "skipped_short" for r in results),
        "failed": sum(r["status"] == "failed" for r in results),
        "exists": sum(r["status"] == "exists" for r in results),
        "results": results,
    }

    seq_names = _scan_sequences(out_root)
    if seq_names:
        (out_root / opts.split_name).write_text("\n".join(seq_names) + "\n")
    registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True))
    with open(out_root / opts.report_name, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description="Convert CVAT-style COCO video folders to LaSOT layout for SUTrack.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--src", required=True, help="root folder of video subfolders")
    p.add_argument("--out", required=True, help="output LaSOT root (e.g. data/lasot_custom)")
    p.add_argument("--image-mode", choices=["copy", "link", "transcode"], default="transcode",
                   help="transcode: real JPEG via cv2/Pillow; copy: raw bytes under .jpg names; "
                        "link: symlink to source")
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--min-frames", type=int, default=20,
                   help="skip sequences with fewer usable frames (sampler needs >= 20)")
    p.add_argument("--class-name", default=None, help="force the LaSOT class folder name")
    p.add_argument("--fallback-class", default="custom")
    p.add_argument("--annotations-subdir", default="annotations")
    p.add_argument("--annotations-name", default="instances_default.json")
    p.add_argument("--images-subdir", default="images")
    p.add_argument("--split-name", default=DEFAULT_SPLIT_NAME)
    p.add_argument("--report-name", default=DEFAULT_REPORT_NAME)
    p.add_argument("--no-clamp", dest="clamp", action="store_false",
                   help="do not clamp boxes to image bounds")
    p.add_argument("--transcode-backend", choices=["auto", "cv2", "PIL"], default="auto")
    p.add_argument("--workers", type=int, default=1, help="parallel video conversions")
    p.add_argument("--overwrite", action="store_true", help="replace existing sequence folders")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    opts = ConvertOptions(
        image_mode=args.image_mode, jpeg_quality=args.jpeg_quality,
        min_frames=args.min_frames, class_name=args.class_name,
        fallback_class=args.fallback_class, overwrite=args.overwrite,
        annotations_subdir=args.annotations_subdir,
        annotations_name=args.annotations_name, images_subdir=args.images_subdir,
        split_name=args.split_name, report_name=args.report_name,
        clamp=args.clamp, transcode_backend=args.transcode_backend,
        workers=args.workers)

    try:
        summary = convert_dataset(args.src, args.out, opts)
    except (NotADirectoryError, ImportError) as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return 1

    for r in summary["results"]:
        if r["status"] == "ok":
            print("ok       %-30s -> %s (%d frames)" % (r["source"], r["sequence"], r["frames_used"]))
        elif r["status"] == "exists":
            print("exists   %-30s %s (use --overwrite)" % (r["source"], r["sequence"]))
        elif r["status"] == "skipped_short":
            print("short    %-30s %s (only %d < %d usable frames)" %
                  (r["source"], r["sequence"], r["frames_used"], opts.min_frames))
        else:
            print("FAILED   %-30s %s" % (r["source"], "; ".join(r["warnings"]) or "unknown"))
    print("\nsequences ok=%d skipped=%d failed=%d exists=%d" %
          (summary["ok"], summary["skipped"], summary["failed"], summary["exists"]))
    print("split file : %s" % (Path(args.out) / opts.split_name))
    print("report     : %s" % (Path(args.out) / opts.report_name))

    # failure = something broke, or the run produced/kept nothing usable;
    # a pure re-run (everything 'exists') is a success
    if summary["failed"] > 0 or (summary["ok"] == 0 and summary["exists"] == 0):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
