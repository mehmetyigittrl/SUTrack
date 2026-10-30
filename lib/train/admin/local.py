import os

# Repo root (lib/train/admin/local.py -> three levels up); used for paths that
# should stay valid when the repo is moved between machines.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..'))


class EnvironmentSettings:
    def __init__(self):
        # Repo-relative (the stale absolute /home/cx paths crashed makedirs on
        # any other machine); all three are gitignored.
        self.workspace_dir = os.path.join(_REPO_ROOT, 'checkpoints')    # Base directory for saving network checkpoints.
        self.tensorboard_dir = os.path.join(_REPO_ROOT, 'tensorboard')    # Directory for tensorboard files.
        self.pretrained_networks = os.path.join(_REPO_ROOT, 'pretrained_networks')
        self.lasot_dir = os.path.join(_REPO_ROOT, 'data', 'lasot')
        # Custom LaSOT-layout dataset produced by tools/coco2lasot.py
        self.lasot_custom_dir = os.path.join(_REPO_ROOT, 'data', 'lasot_custom')
        self.vasttrack_dir = '/home/cx/cx1/github-repo/SUTrack/data/vasttrack'
        self.got10k_dir = '/home/cx/cx1/github-repo/SUTrack/data/got10k/train'
        self.lasot_lmdb_dir = '/home/cx/cx1/github-repo/SUTrack/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/home/cx/cx1/github-repo/SUTrack/data/got10k_lmdb'
        self.trackingnet_dir = '/home/cx/cx1/github-repo/SUTrack/data/trackingnet'
        self.trackingnet_lmdb_dir = '/home/cx/cx1/github-repo/SUTrack/data/trackingnet_lmdb'
        self.coco_dir = '/home/cx/cx1/github-repo/SUTrack/data/coco'
        self.coco_lmdb_dir = '/home/cx/cx1/github-repo/SUTrack/data/coco_lmdb'
        self.imagenet1k_dir = '/home/cx/cx1/github-repo/SUTrack/data/imagenet1k'
        self.imagenet22k_dir = '/home/cx/cx1/github-repo/SUTrack/data/imagenet22k'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/home/cx/cx1/github-repo/SUTrack/data/vid'
        self.imagenet_lmdb_dir = '/home/cx/cx1/github-repo/SUTrack/data/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
        self.depthtrack_dir = '/home/cx/cx1/github-repo/SUTrack/data/depthtrack/train'
        self.lasher_dir = '/home/cx/cx1/github-repo/SUTrack/data/lasher/trainingset'
        self.visevent_dir = '/home/cx/cx1/github-repo/SUTrack/data/visevent/train'
        self.refcoco_dir = '/home/cx/cx1/github-repo/SUTrack/data/refcoco'
        self.tnl2k_dir = '/home/cx/cx1/github-repo/SUTrack/data/tnl2k/train'
        self.otb99_dir = '/home/cx/cx1/github-repo/SUTrack/data/otb_lang'
