"""Tests for tools/coco2lasot.py (COCO/CVAT-video -> LaSOT converter) and repo wiring.

Run inside the SUTrack python env, e.g.:
    python -m pytest tools/tests/test_coco2lasot.py -v
or:
    python -m unittest tools.tests.test_coco2lasot -v
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools import coco2lasot  # noqa: E402

VAL2017_DIR = REPO_ROOT / "data" / "val2017"
VAL2017_JSON = REPO_ROOT / "data" / "annotations" / "instances_val2017.json"


# --------------------------------------------------------------------------
# fixture helpers
# --------------------------------------------------------------------------
def png_bytes(w, h, color=(120, 40, 200)):
    """Real PNG bytes (needs cv2/numpy, available in the SUTrack env)."""
    import cv2
    import numpy as np
    img = np.full((h, w, 3), color, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def frame_bytes(i, w=320, h=240):
    """Deterministic distinct content per frame index (real PNG)."""
    return png_bytes(w, h, color=(i % 256, (i * 37) % 256, (i * 73) % 256))


def box_i(i):
    return [float(10 + i), 20.0, 30.0, 40.0]


def write_coco_video(src_root, vid_name, frames, categories=None,
                     raw_bytes=False, w=320, h=240):
    """Write one CVAT-style video subfolder.

    frames: list of dicts:
        {'id': int, 'anns': [ {'bbox': [...], 'category_id': int, 'iscrowd': 0, 'id': int} ... ] or None,
         'on_disk': bool (default True), 'file_name': optional override}
    Returns (vid_path, images_entries, annotations_entries).
    """
    if categories is None:
        categories = [{"id": 1, "name": "uav", "supercategory": "vehicle"}]
    vid_dir = src_root / vid_name
    (vid_dir / "annotations").mkdir(parents=True, exist_ok=True)
    (vid_dir / "images").mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    ann_id = 1000
    for k, fr in enumerate(frames, start=1):
        fname = fr.get("file_name", "frame_%06d.png" % k)
        images.append({"id": fr["id"], "file_name": fname, "width": w, "height": h})
        if fr.get("on_disk", True):
            data = (b"FAKEIMG%d" % fr["id"]) if raw_bytes else frame_bytes(fr["id"], w, h)
            (vid_dir / "images" / fname).write_bytes(data)
        for ann in fr.get("anns") or []:
            ann = dict(ann)
            ann.setdefault("id", ann_id)
            ann.setdefault("iscrowd", 0)
            ann.setdefault("category_id", 1)
            ann["image_id"] = fr["id"]
            annotations.append(ann)
            ann_id += 1
    (vid_dir / "annotations" / "instances_default.json").write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": categories}))
    return vid_dir, images, annotations


def read_gt(seq_dir):
    lines = (seq_dir / "groundtruth.txt").read_text().strip().splitlines()
    return [[float(v) for v in ln.split(",")] for ln in lines]


def make_minimal_lasot_seq(root, class_name, vid_name, n=21):
    """Hand-build a valid LaSOT sequence for wiring tests."""
    seq = root / class_name / ("%s-%s" % (class_name, vid_name))
    (seq / "img").mkdir(parents=True, exist_ok=True)
    for i in range(1, n + 1):
        (seq / "img" / ("%08d.jpg" % i)).write_bytes(frame_bytes(i + 500))
    with open(seq / "groundtruth.txt", "w") as f:
        for i in range(n):
            f.write("10.00,20.00,30.00,40.00\n")
    (seq / "full_occlusion.txt").write_text(",".join(["0"] * n) + "\n")
    (seq / "out_of_view.txt").write_text(",".join(["0"] * n) + "\n")
    (seq / "nlp.txt").write_text("a %s.\n" % class_name)
    return seq


# --------------------------------------------------------------------------
# pure helpers
# --------------------------------------------------------------------------
class TestSanitize(unittest.TestCase):
    def test_lowercases_and_replaces_non_alnum(self):
        self.assertEqual(coco2lasot.sanitize_name("Remote Control"), "remote_control")

    def test_hyphen_becomes_underscore(self):
        # the training loader splits sequence names on '-', hyphens are illegal
        self.assertEqual(coco2lasot.sanitize_name("my-video"), "my_video")

    def test_collapses_repeated_underscores(self):
        self.assertEqual(coco2lasot.sanitize_name("a__b"), "a_b")

    def test_strips_and_can_be_empty(self):
        self.assertEqual(coco2lasot.sanitize_name("!!"), "")


# --------------------------------------------------------------------------
# core conversion on synthetic CVAT folders (copy mode = stdlib only)
# --------------------------------------------------------------------------
class TestConvertSynthetic(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="c2l_syn_"))
        self.src = self.tmp / "src"
        self.out = self.tmp / "out"
        self.src.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def opts(self, **kw):
        kw.setdefault("image_mode", "copy")
        return coco2lasot.ConvertOptions(**kw)

    # ---- happy path -------------------------------------------------
    def test_happy_path_video(self):
        frames = [{"id": i, "anns": [{"bbox": box_i(i)}]} for i in range(1, 26)]
        write_coco_video(self.src, "vid1", frames, raw_bytes=True)
        summary = coco2lasot.convert_dataset(self.src, self.out, self.opts())

        self.assertEqual(summary["ok"], 1)
        self.assertEqual(summary["skipped"], 0)
        seq = self.out / "uav" / "uav-vid1"
        self.assertTrue((seq / "groundtruth.txt").exists())
        gt = read_gt(seq)
        self.assertEqual(len(gt), 25)
        self.assertEqual(gt[0], [11.0, 20.0, 30.0, 40.0])  # box_i(1)
        self.assertEqual(gt[24], [35.0, 20.0, 30.0, 40.0])  # box_i(25)
        # contiguous renumbered frames
        for i in range(1, 26):
            f = seq / "img" / ("%08d.jpg" % i)
            self.assertTrue(f.exists(), f)
            self.assertEqual(f.read_bytes(), b"FAKEIMG%d" % i)  # copy keeps bytes
        self.assertEqual(sorted(os.listdir(seq / "img")), ["%08d.jpg" % i for i in range(1, 26)])

    def test_occlusion_outofview_nlp_files(self):
        frames = [{"id": i, "anns": [{"bbox": box_i(i)}]} for i in range(1, 26)]
        write_coco_video(self.src, "vid1", frames, raw_bytes=True)
        coco2lasot.convert_dataset(self.src, self.out, self.opts())
        seq = self.out / "uav" / "uav-vid1"
        self.assertEqual((seq / "full_occlusion.txt").read_text().strip(), ",".join(["0"] * 25))
        self.assertEqual((seq / "out_of_view.txt").read_text().strip(), ",".join(["0"] * 25))
        nlp = (seq / "nlp.txt").read_text()
        self.assertEqual(len(nlp.strip().splitlines()), 1)
        self.assertNotIn(",", nlp)

    # ---- skip rules --------------------------------------------------
    def test_skips_missing_ann_missing_file_crowd_multi(self):
        frames = []
        for i in range(1, 26):
            fr = {"id": i, "anns": [{"bbox": box_i(i)}]}
            if i in (3, 7):                       # frame with no annotation
                fr["anns"] = None
            if i == 5:                            # annotation without image on disk
                fr["on_disk"] = False
            if i == 9:                            # two boxes on one frame -> first by ann id
                fr["anns"] = [{"bbox": box_i(i), "id": 91}, {"bbox": [1.0, 2.0, 3.0, 4.0], "id": 92}]
            if i == 11:                           # only a crowd ann -> dropped
                fr["anns"] = [{"bbox": box_i(i), "iscrowd": 1}]
            frames.append(fr)
        write_coco_video(self.src, "vid2", frames, raw_bytes=True)
        summary = coco2lasot.convert_dataset(self.src, self.out, self.opts())

        self.assertEqual(summary["ok"], 1)
        res = summary["results"][0]
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["frames_used"], 21)
        self.assertEqual(res["skipped_no_ann"], 3)      # frames 3, 7 and crowd-only 11
        self.assertEqual(res["skipped_missing_img"], 1)
        self.assertEqual(res["skipped_invalid_box"], 0)
        self.assertEqual(res["multi_ann_frames"], 1)
        seq = self.out / "uav" / "uav-vid2"
        self.assertEqual(len(read_gt(seq)), 21)
        # kept frames are 1,2,4,6,8,9,10,12..25 -> index 3 is frame 6, index 5 is frame 9 (first ann kept)
        gt = read_gt(seq)
        self.assertEqual(gt[3], box_i(6))
        self.assertEqual(gt[4], box_i(8))
        self.assertEqual(gt[5], box_i(9))

    def test_short_video_skipped_and_threshold_configurable(self):
        frames = [{"id": i, "anns": [{"bbox": box_i(i)}]} for i in range(1, 11)]
        write_coco_video(self.src, "vid3", frames, raw_bytes=True)
        summary = coco2lasot.convert_dataset(self.src, self.out, self.opts())
        self.assertEqual(summary["skipped"], 1)
        self.assertFalse((self.out / "uav" / "uav-vid3").exists())

        summary2 = coco2lasot.convert_dataset(self.src, self.out, self.opts(min_frames=5, overwrite=True))
        self.assertEqual(summary2["ok"], 1)
        self.assertEqual(len(read_gt(self.out / "uav" / "uav-vid3")), 10)

    # ---- naming ------------------------------------------------------
    def test_hyphen_in_subfolder_name_sanitized(self):
        frames = [{"id": i, "anns": [{"bbox": box_i(i)}]} for i in range(1, 26)]
        write_coco_video(self.src, "test-vid", frames, raw_bytes=True)
        coco2lasot.convert_dataset(self.src, self.out, self.opts())
        seq = self.out / "uav" / "uav-test_vid"
        self.assertTrue((seq / "groundtruth.txt").exists())
        split = (self.out / "lasot_custom_train_split.txt").read_text().split()
        self.assertIn("uav-test_vid", split)

    def test_unknown_category_falls_back_to_custom(self):
        cats = [{"id": 1, "name": "uav", "supercategory": "vehicle"}]
        frames = [{"id": i, "anns": [{"bbox": box_i(i), "category_id": 7}]} for i in range(1, 26)]
        write_coco_video(self.src, "nocat", frames, categories=cats, raw_bytes=True)
        coco2lasot.convert_dataset(self.src, self.out, self.opts())
        self.assertTrue((self.out / "custom" / "custom-nocat" / "groundtruth.txt").exists())

    def test_majority_category_wins(self):
        cats = [{"id": 1, "name": "uav", "supercategory": "v"},
                {"id": 2, "name": "bird", "supercategory": "v"}]
        frames = []
        for i in range(1, 23):
            cat = 2 if i in (2, 5, 9) else 1
            frames.append({"id": i, "anns": [{"bbox": box_i(i), "category_id": cat}]})
        write_coco_video(self.src, "mixed", frames, categories=cats, raw_bytes=True)
        coco2lasot.convert_dataset(self.src, self.out, self.opts())
        self.assertTrue((self.out / "uav" / "uav-mixed").exists())

    def test_class_name_override(self):
        frames = [{"id": i, "anns": [{"bbox": box_i(i)}]} for i in range(1, 26)]
        write_coco_video(self.src, "vid1", frames, raw_bytes=True)
        coco2lasot.convert_dataset(self.src, self.out, self.opts(class_name="My Target"))
        self.assertTrue((self.out / "my_target" / "my_target-vid1").exists())

    # ---- bbox clamping -------------------------------------------------
    def test_bbox_clamped_and_invalid_dropped(self):
        frames = []
        for i in range(1, 26):
            bbox = box_i(i)
            if i == 4:
                bbox = [310.0, 230.0, 30.0, 30.0]   # exceeds 320x240 -> clamped
            if i == 5:
                bbox = [400.0, 500.0, 20.0, 20.0]   # fully outside -> dropped
            if i == 6:
                bbox = [50.0, 50.0, 0.0, 20.0]      # zero width -> dropped
            frames.append({"id": i, "anns": [{"bbox": bbox}]})
        write_coco_video(self.src, "clamped", frames, raw_bytes=True)
        summary = coco2lasot.convert_dataset(self.src, self.out, self.opts())

        res = summary["results"][0]
        self.assertEqual(res["skipped_invalid_box"], 2)
        gt = read_gt(self.out / "uav" / "uav-clamped")
        self.assertEqual(len(gt), 23)
        self.assertEqual(gt[3], [310.0, 230.0, 10.0, 10.0])

    def test_no_clamp_option(self):
        frames = []
        for i in range(1, 26):
            bbox = [310.0, 230.0, 30.0, 30.0] if i == 4 else box_i(i)
            frames.append({"id": i, "anns": [{"bbox": bbox}]})
        write_coco_video(self.src, "clamped", frames, raw_bytes=True)
        coco2lasot.convert_dataset(self.src, self.out, self.opts(clamp=False))
        gt = read_gt(self.out / "uav" / "uav-clamped")
        self.assertEqual(gt[3], [310.0, 230.0, 30.0, 30.0])

    # ---- split / report / overwrite ------------------------------------
    def test_split_file_written_sorted(self):
        for name in ("vid1", "vid2"):
            frames = [{"id": i, "anns": [{"bbox": box_i(i)}]} for i in range(1, 26)]
            write_coco_video(self.src, name, frames, raw_bytes=True)
        coco2lasot.convert_dataset(self.src, self.out, self.opts())
        split = (self.out / "lasot_custom_train_split.txt").read_text().split()
        self.assertEqual(split, ["uav-vid1", "uav-vid2"])

    def test_report_json_written(self):
        frames = [{"id": i, "anns": [{"bbox": box_i(i)}]} for i in range(1, 26)]
        write_coco_video(self.src, "vid1", frames, raw_bytes=True)
        coco2lasot.convert_dataset(self.src, self.out, self.opts())
        report = json.loads((self.out / "conversion_report.json").read_text())
        self.assertEqual(report["ok"], 1)
        self.assertEqual(len(report["results"]), 1)

    def test_rerun_without_overwrite_reports_exists(self):
        frames = [{"id": i, "anns": [{"bbox": box_i(i)}]} for i in range(1, 26)]
        write_coco_video(self.src, "vid1", frames, raw_bytes=True)
        coco2lasot.convert_dataset(self.src, self.out, self.opts())
        s2 = coco2lasot.convert_dataset(self.src, self.out, self.opts())
        self.assertEqual(s2["results"][0]["status"], "exists")
        self.assertEqual(s2["ok"], 0)

    def test_broken_and_empty_subfolders_do_not_crash(self):
        (self.src / "empty_vid" / "images").mkdir(parents=True)   # no annotations folder
        (self.src / "note.txt").write_text("stray file")
        frames = [{"id": i, "anns": [{"bbox": box_i(i)}]} for i in range(1, 26)]
        write_coco_video(self.src, "vid1", frames, raw_bytes=True)
        summary = coco2lasot.convert_dataset(self.src, self.out, self.opts())
        self.assertEqual(summary["ok"], 1)
        self.assertEqual(summary["failed"], 1)
        res = [r for r in summary["results"] if r["source"] == "empty_vid"]
        self.assertEqual(res[0]["status"], "failed")

    def test_bad_json_recorded_as_failed(self):
        bad = self.src / "badjson"
        (bad / "annotations").mkdir(parents=True)
        (bad / "images").mkdir(parents=True)
        (bad / "annotations" / "instances_default.json").write_text("{not json")
        frames = [{"id": i, "anns": [{"bbox": box_i(i)}]} for i in range(1, 26)]
        write_coco_video(self.src, "vid1", frames, raw_bytes=True)
        summary = coco2lasot.convert_dataset(self.src, self.out, self.opts())
        self.assertEqual(summary["ok"], 1)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["results"][0]["source"], "badjson")


# --------------------------------------------------------------------------
# image transcoding (needs cv2 / PIL from the SUTrack env)
# --------------------------------------------------------------------------
class TestTranscode(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="c2l_tr_"))
        self.src = self.tmp / "src"
        self.out = self.tmp / "out"
        self.src.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def make_video(self, mode, **kw):
        frames = [{"id": i, "anns": [{"bbox": box_i(i)}]} for i in range(1, 26)]
        write_coco_video(self.src, "vid1", frames)  # real PNG inputs
        return coco2lasot.convert_dataset(
            self.src, self.out, coco2lasot.ConvertOptions(image_mode=mode, **kw))

    def test_transcode_produces_real_jpeg(self):
        import cv2
        self.make_video("transcode")
        f = self.out / "uav" / "uav-vid1" / "img" / "00000001.jpg"
        data = f.read_bytes()
        self.assertTrue(data[:2] == b"\xff\xd8", "missing JPEG magic")
        img = cv2.imread(str(f))
        self.assertIsNotNone(img)
        self.assertEqual(img.shape, (240, 320, 3))

    def test_transcode_via_pil_backend(self):
        self.make_video("transcode", transcode_backend="PIL")
        f = self.out / "uav" / "uav-vid1" / "img" / "00000001.jpg"
        self.assertEqual(f.read_bytes()[:2], b"\xff\xd8")

    def test_link_mode_symlinks(self):
        self.make_video("link")
        f = self.out / "uav" / "uav-vid1" / "img" / "00000001.jpg"
        self.assertTrue(f.is_symlink())
        # resolves to the source PNG (content-sniffing readers handle it)
        self.assertTrue(f.exists())


# --------------------------------------------------------------------------
# repo wiring: LasotCustom, names2datasets, config, local.py, yaml
# --------------------------------------------------------------------------
class TestWiring(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="c2l_wire_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_lasot_custom_class_loads_like_lasot(self):
        from lib.train.dataset import LasotCustom  # via package __init__
        from lib.train.data import opencv_loader

        root = self.tmp / "lasot_root"
        make_minimal_lasot_seq(root, "uav", "vid1", n=21)
        split = self.tmp / "split.txt"
        split.write_text("uav-vid1\n")

        ds = LasotCustom(root=root, split="train", split_file=split, image_loader=opencv_loader)
        self.assertEqual(ds.get_num_sequences(), 1)
        self.assertEqual(ds.get_name(), "lasot")  # inherits -> TASK_INDEX['LASOT'] keeps working
        info = ds.get_sequence_info(0)
        self.assertEqual(tuple(info["bbox"].shape), (21, 4))
        self.assertEqual(int(info["visible"].sum()), 21)
        self.assertTrue(ds._get_frame_path(ds._get_sequence_path(0), 0).endswith("img/00000001.jpg"))
        frames, annos, meta = ds.get_frames(0, [0, 20])
        self.assertEqual(frames[0].shape, (240, 320, 3))
        self.assertEqual(meta["object_class_name"], "uav")

    def test_lasot_custom_default_root_from_env(self):
        from lib.train.admin.local import EnvironmentSettings
        env = EnvironmentSettings()
        self.assertEqual(env.lasot_custom_dir, str(REPO_ROOT / "data" / "lasot_custom"))

    def test_names2datasets_wires_lasot_custom(self):
        import types
        from lib.train.base_functions import names2datasets
        from lib.train.dataset import LasotCustom

        # tmp LaSOT root + split file at the default data_specs location
        root = self.tmp / "lasot_root"
        make_minimal_lasot_seq(root, "uav", "vid1", n=21)
        split_default = REPO_ROOT / "lib" / "train" / "data_specs" / "lasot_custom_train_split.txt"
        had_split = split_default.exists()
        old_split = split_default.read_text() if had_split else None
        split_default.write_text("uav-vid1\n")
        try:
            settings = types.SimpleNamespace(
                env=types.SimpleNamespace(lasot_custom_dir=str(root)), use_lmdb=False,
                multi_modal_vision=False, multi_modal_language=False,
                use_nlp={"LASOT_CUSTOM": False})
            ds_list = names2datasets(["LASOT_CUSTOM"], settings, image_loader=lambda p: None)
            self.assertIsInstance(ds_list[0], LasotCustom)
            self.assertEqual(ds_list[0].root, str(root))
            self.assertEqual(ds_list[0].get_num_sequences(), 1)
        finally:
            if had_split:
                split_default.write_text(old_split)
            else:
                split_default.unlink()

    def test_config_has_use_nlp_default_and_yaml_merges(self):
        import importlib
        mod = importlib.import_module("lib.config.sutrack.config")
        self.assertIs(mod.cfg.DATA.USE_NLP.LASOT_CUSTOM, False)

        yaml_path = REPO_ROOT / "experiments" / "sutrack" / "sutrack_b384_custom.yaml"
        self.assertTrue(yaml_path.exists(), "sutrack_b384_custom.yaml missing")
        # _update_config raises ValueError for any key absent from config.py
        mod.update_config_from_file(str(yaml_path))
        self.assertEqual(mod.cfg.DATA.TRAIN.DATASETS_NAME, ["LASOT_CUSTOM"])
        self.assertEqual(mod.cfg.DATA.TRAIN.DATASETS_RATIO, [1])

    def test_split_spec_file_ignored_by_git(self):
        # the generated split file must not show up as untracked noise
        r = subprocess.run(["git", "check-ignore", "-q",
                            "lib/train/data_specs/lasot_custom_train_split.txt"],
                           cwd=str(REPO_ROOT))
        self.assertEqual(r.returncode, 0, "lasot_custom_train_split.txt is not gitignored")


# --------------------------------------------------------------------------
# end-to-end: real coco2017 -> mock CVAT dataset -> CLI -> repo loader
# --------------------------------------------------------------------------
@unittest.skipUnless(VAL2017_JSON.exists() and VAL2017_DIR.exists(),
                     "coco2017 val2017 not present under ./data")
class TestEndToEndCoco2017Real(unittest.TestCase):
    N_MAIN = 27   # frames in main video; 4 get dropped -> 23 used (>= 20)

    @classmethod
    def setUpClass(cls):
        import cv2
        cls.tmp = Path(tempfile.mkdtemp(prefix="c2l_e2e_"))
        cls.src = cls.tmp / "mock_coco"
        cls.out = cls.tmp / "lasot_out"
        cls.src.mkdir(parents=True)

        with open(VAL2017_JSON) as f:
            coco = json.load(f)
        imgs = {im["id"]: im for im in coco["images"]}
        cats = {c["id"]: c for c in coco["categories"]}

        # collect annotations grouped per image, non-crowd, reasonably sized
        per_img = {}
        for a in coco["annotations"]:
            if a["iscrowd"] != 0:
                continue
            x, y, w, h = a["bbox"]
            if w < 40 or h < 40:
                continue
            per_img.setdefault(a["image_id"], []).append(a)

        picked = [imgs[i] for i in sorted(per_img)][:cls.N_MAIN]
        assert len(picked) == cls.N_MAIN
        cls.main_cat = cats[per_img[picked[0]["id"]][0]["category_id"]]["name"]
        cls.cls_name = coco2lasot.sanitize_name(cls.main_cat)
        cls.main_img_size = (picked[0]["height"], picked[0]["width"])
        cls.cats_payload = [{"id": 1, "name": cls.main_cat, "supercategory": "custom"}]

        def write_video(vid_name, n, anns_for, drop_ids=()):
            vid_dir = cls.src / vid_name
            (vid_dir / "annotations").mkdir(parents=True)
            (vid_dir / "images").mkdir(parents=True)
            images, annotations = [], []
            for k in range(1, n + 1):
                im = picked[(k - 1) % cls.N_MAIN]
                fname = "image_%06d.png" % k
                images.append({"id": k, "file_name": fname,
                               "width": im["width"], "height": im["height"]})
                if k not in drop_ids:
                    img = cv2.imread(str(VAL2017_DIR / im["file_name"]))
                    ok, buf = cv2.imencode(".png", img)
                    assert ok
                    (vid_dir / "images" / fname).write_bytes(buf.tobytes())
                for ann in anns_for(k):
                    ann = dict(ann)
                    ann["image_id"] = k
                    annotations.append(ann)
            (vid_dir / "annotations" / "instances_default.json").write_text(
                json.dumps({"images": images, "annotations": annotations,
                            "categories": cls.cats_payload}))

        def main_anns(k):
            im = picked[(k - 1) % cls.N_MAIN]
            base = per_img[im["id"]][0]["bbox"]
            bbox = [base[0] + (k % 5) - 2, base[1] + (k % 3) - 1, base[2], base[3]]
            if k == 23:                       # fully outside -> dropped frame
                bbox = [9000.0, 9000.0, 20.0, 20.0]
            if k in (20, 21, 22, 23):
                if k == 22:
                    return [{"bbox": bbox, "category_id": 1, "iscrowd": 1, "id": 9000 + k}]
                return [{"bbox": bbox, "category_id": 1, "iscrowd": 0, "id": 9000 + k}]
            return [{"bbox": bbox, "category_id": 1, "iscrowd": 0, "id": 9000 + k}]

        # main video: 26 frames; #20 no ann, #21 file missing, #22 crowd-only, #23 outside
        write_video("main_video", cls.N_MAIN,
                    lambda k: ([] if k == 20 else main_anns(k)),
                    drop_ids={21})
        # short video -> skipped by min_frames
        write_video("short_video", 6,
                    lambda k: [{"bbox": [50.0, 60.0, 80.0, 90.0], "category_id": 1,
                                "iscrowd": 0, "id": 9500 + k}])
        # hyphen-named video -> sanitized sequence name
        write_video("hyphen-vid", 22,
                    lambda k: [{"bbox": [40.0, 50.0, 90.0, 110.0], "category_id": 1,
                                "iscrowd": 0, "id": 9700 + k}])

        # run the converter CLI end to end
        r = subprocess.run(
            [sys.executable, str(REPO_ROOT / "tools" / "coco2lasot.py"),
             "--src", str(cls.src), "--out", str(cls.out), "--image-mode", "transcode"],
            capture_output=True, text=True)
        cls.cli_out = r

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_cli_exit_zero(self):
        self.assertEqual(self.cli_out.returncode, 0, self.cli_out.stderr[-2000:])

    def test_output_structure(self):
        seq = self.out / self.cls_name / ("%s-main_video" % self.cls_name)
        gt = read_gt(seq)
        self.assertEqual(len(gt), 23)          # 27 - no-ann - missing-file - crowd - outside
        self.assertEqual((seq / "full_occlusion.txt").read_text().strip(), ",".join(["0"] * 23))
        for i in range(1, 24):
            self.assertTrue((seq / "img" / ("%08d.jpg" % i)).exists())
        self.assertEqual((seq / "img" / "00000001.jpg").read_bytes()[:2], b"\xff\xd8")

    def test_short_video_not_in_output(self):
        split = (self.out / "lasot_custom_train_split.txt").read_text().split()
        self.assertEqual(split, sorted(split))
        self.assertNotIn("%s-short_video" % self.cls_name, split)
        self.assertIn("%s-hyphen_vid" % self.cls_name, split)

    def test_repo_loader_reads_converted_dataset(self):
        from lib.train.dataset import LasotCustom
        from lib.train.data import opencv_loader

        ds = LasotCustom(root=self.out, split="train",
                         split_file=self.out / "lasot_custom_train_split.txt",
                         image_loader=opencv_loader)
        self.assertEqual(ds.get_num_sequences(), 2)  # main + hyphen, short skipped
        idx = ds.sequence_list.index("%s-main_video" % self.cls_name)
        info = ds.get_sequence_info(idx)
        self.assertEqual(tuple(info["bbox"].shape), (23, 4))
        self.assertEqual(int(info["visible"].sum()), 23)
        frames, annos, meta = ds.get_frames(idx, [0, 22])
        self.assertEqual(frames[0].shape[:2], self.main_img_size)
        self.assertEqual(frames[0].ndim, 3)

    def test_satisfies_sampler_minimum(self):
        # sampler gate: len(visible) >= 20 and visible.sum() > 2*(search+template) = 6
        from lib.train.dataset import LasotCustom
        from lib.train.data import opencv_loader

        ds = LasotCustom(root=self.out, split="train",
                         split_file=self.out / "lasot_custom_train_split.txt",
                         image_loader=opencv_loader)
        for i in range(ds.get_num_sequences()):
            info = ds.get_sequence_info(i)
            self.assertGreaterEqual(len(info["visible"]), 20)
            self.assertGreater(int(info["visible"].sum()), 6)


# --------------------------------------------------------------------------
# regressions for the adversarial-review findings
# --------------------------------------------------------------------------
class TestIncrementalAndCollisions(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="c2l_rev_"))
        self.src = self.tmp / "src"
        self.out = self.tmp / "out"
        self.src.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def opts(self, **kw):
        kw.setdefault("image_mode", "copy")
        return coco2lasot.ConvertOptions(**kw)

    def _vid(self, name, n=25):
        frames = [{"id": i, "anns": [{"bbox": box_i(i)}]} for i in range(1, n + 1)]
        return write_coco_video(self.src, name, frames, raw_bytes=True)

    def _split(self):
        return (self.out / "lasot_custom_train_split.txt").read_text().split()

    def test_incremental_rerun_keeps_all_sequences_in_split(self):
        # review finding: the split was rebuilt from only this run's 'ok'
        # rows, so adding one video and re-running silently dropped the rest
        self._vid("vid1")
        self._vid("vid2")
        s = coco2lasot.convert_dataset(self.src, self.out, self.opts())
        self.assertEqual(s["ok"], 2)
        self._vid("vid3")
        s2 = coco2lasot.convert_dataset(self.src, self.out, self.opts())
        self.assertEqual(s2["ok"], 1)
        self.assertEqual(s2["exists"], 2)
        self.assertEqual(self._split(), ["uav-vid1", "uav-vid2", "uav-vid3"])

    def test_name_collision_within_run_gets_unique_sequences(self):
        # 'drone-1' and 'drone_1' both sanitize to 'drone_1'
        self._vid("drone-1")
        self._vid("drone_1")
        s = coco2lasot.convert_dataset(self.src, self.out, self.opts())
        self.assertEqual((s["ok"], s["exists"]), (2, 0))
        self.assertTrue((self.out / "uav" / "uav-drone_1" / "groundtruth.txt").is_file())
        self.assertTrue((self.out / "uav" / "uav-drone_1_2" / "groundtruth.txt").is_file())
        self.assertEqual(self._split(), ["uav-drone_1", "uav-drone_1_2"])

    def test_name_collision_with_overwrite_preserves_both_videos(self):
        # with --overwrite the second video used to rmtree the first one's
        # output and duplicate the split entry
        self._vid("drone-1")
        self._vid("drone_1")
        s = coco2lasot.convert_dataset(self.src, self.out, self.opts(overwrite=True))
        self.assertEqual(s["ok"], 2)
        self.assertEqual(len(read_gt(self.out / "uav" / "uav-drone_1")), 25)
        self.assertEqual(len(read_gt(self.out / "uav" / "uav-drone_1_2")), 25)
        self.assertEqual(self._split(), ["uav-drone_1", "uav-drone_1_2"])

    def test_cross_run_collision_gets_suffix(self):
        self._vid("a-b")
        coco2lasot.convert_dataset(self.src, self.out, self.opts())
        shutil.rmtree(self.src)
        self.src.mkdir()
        self._vid("a_b")  # same sanitized name, different source folder
        s = coco2lasot.convert_dataset(self.src, self.out, self.opts())
        self.assertEqual(s["ok"], 1)
        self.assertEqual(s["exists"], 0)
        self.assertEqual(self._split(), ["uav-a_b", "uav-a_b_2"])

    def test_all_exists_rerun_exits_zero(self):
        self._vid("vid1")
        argv = ["--src", str(self.src), "--out", str(self.out), "--image-mode", "copy"]
        self.assertEqual(coco2lasot.main(argv), 0)
        self.assertEqual(coco2lasot.main(argv), 0)  # nothing new, nothing failed

    def test_half_written_sequence_dir_is_rebuilt(self):
        # a dir left by an interrupted run (img/ but no groundtruth.txt) must
        # not be reported 'exists'
        self._vid("vid1")
        stale = self.out / "uav" / "uav-vid1" / "img"
        stale.mkdir(parents=True)
        (stale / "00000001.jpg").write_bytes(b"junk")
        s = coco2lasot.convert_dataset(self.src, self.out, self.opts())
        self.assertEqual(s["ok"], 1)
        self.assertEqual(len(read_gt(self.out / "uav" / "uav-vid1")), 25)

    def test_sequence_sources_registry_written(self):
        self._vid("drone-1")
        self._vid("drone_1")
        coco2lasot.convert_dataset(self.src, self.out, self.opts())
        reg = json.loads((self.out / "sequence_sources.json").read_text())
        self.assertEqual(reg, {"uav-drone_1": "drone-1", "uav-drone_1_2": "drone_1"})


class TestFinetuneLoader(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="c2l_ft_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _net(self):
        import torch
        return torch.nn.Sequential(torch.nn.Linear(4, 4))

    def test_loads_state_dict_with_module_prefix_and_extra_keys(self):
        import torch
        from lib.train.train_script import load_pretrained_tracker
        net = self._net()
        target_w = torch.randn(4, 4)
        sd = {"module.0.weight": target_w.clone(),
              "module.0.bias": torch.zeros(4),
              "text_encoder.proj.weight": torch.randn(2, 2)}  # CLIP weights, absent when disabled
        ckpt = self.tmp / "ck.pth"
        torch.save(sd, ckpt)
        missing, unexpected = load_pretrained_tracker(net, ckpt)
        self.assertTrue(torch.equal(net[0].weight, target_w))
        self.assertEqual(missing, [])
        self.assertIn("text_encoder.proj.weight", unexpected)

    def test_loads_checkpoint_wrapped_in_model_key(self):
        import torch
        from lib.train.train_script import load_pretrained_tracker
        net = self._net()
        target_b = torch.randn(4)
        ckpt = self.tmp / "ck2.pth"
        torch.save({"model": {"0.weight": torch.zeros(4, 4), "0.bias": target_b.clone()}}, ckpt)
        missing, unexpected = load_pretrained_tracker(net, ckpt)
        self.assertTrue(torch.equal(net[0].bias, target_b))
        self.assertEqual((missing, unexpected), ([], []))

    def test_local_py_save_dirs_are_repo_relative(self):
        # stale absolute /home/cx paths crash makedirs on any other machine
        from lib.train.admin.environment import env_settings
        env = env_settings()
        self.assertEqual(str(Path(env.workspace_dir)), str(REPO_ROOT / "checkpoints"))
        self.assertEqual(str(Path(env.tensorboard_dir)), str(REPO_ROOT / "tensorboard"))
        self.assertEqual(str(Path(env.pretrained_networks)), str(REPO_ROOT / "pretrained_networks"))


class TestTestPathWiring(unittest.TestCase):
    """The tracking/test.py entry and test-side env settings must work on any machine."""

    def test_tracking_test_py_runs_in_script_mode(self):
        # README invokes `python tracking/test.py sutrack <cfg> --dataset ...`;
        # the sys.path bootstrap used to resolve one level ABOVE the repo root.
        out = subprocess.run(
            [sys.executable, "tracking/test.py", "--help"],
            capture_output=True, text=True, cwd=str(REPO_ROOT))
        self.assertEqual(out.returncode, 0, msg=out.stderr[-2000:])

    def test_eval_local_paths_are_repo_relative(self):
        # stale absolute /home/cx paths broke parameters() (FileNotFoundError on
        # the experiments yaml) and would send test results to a missing tree
        from lib.test.evaluation.local import local_env_settings
        s = local_env_settings()
        self.assertEqual(str(Path(s.prj_dir)), str(REPO_ROOT))
        self.assertEqual(str(Path(s.save_dir)), str(REPO_ROOT))
        self.assertTrue(str(Path(s.results_path)).startswith(str(REPO_ROOT)))
        for attr in ("network_path", "result_plot_path", "segmentation_path",
                     "lasot_path", "lasot_lmdb_path", "lasot_extension_subset_path",
                     "got10k_path", "got10k_lmdb_path", "otb_path", "otblang_path",
                     "uav_path", "nfs_path", "tc128_path", "tnl2k_path",
                     "trackingnet_path", "vot_path"):
            self.assertTrue(str(getattr(s, attr)).startswith(str(REPO_ROOT)),
                            "%s=%r not under repo root" % (attr, getattr(s, attr)))

    def test_parameters_resolves_custom_yaml_and_checkpoint(self):
        # parameters(yaml_name) reads experiments/sutrack/<name>.yaml via
        # env_settings().prj_dir and builds the checkpoint path from save_dir
        import contextlib
        import io
        from lib.test.parameter.sutrack import parameters
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):  # parameters() prints the cfg
            p = parameters("sutrack_b384_custom")
        self.assertTrue(str(p.checkpoint).startswith(str(REPO_ROOT)))
        self.assertTrue(p.checkpoint.endswith(".pth.tar"))


if __name__ == "__main__":
    unittest.main()
