import os

from lib.test.evaluation.environment import EnvSettings

# Repo root (lib/test/evaluation/local.py -> three levels up); used for paths
# that should stay valid when the repo is moved between machines.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..'))


def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.prj_dir = _REPO_ROOT
    settings.save_dir = _REPO_ROOT
    settings.results_path = os.path.join(_REPO_ROOT, 'test', 'tracking_results')    # Where to store tracking results
    settings.segmentation_path = os.path.join(_REPO_ROOT, 'test', 'segmentation_results')
    settings.network_path = os.path.join(_REPO_ROOT, 'test', 'networks')    # Where tracking networks are stored.
    settings.result_plot_path = os.path.join(_REPO_ROOT, 'test', 'result_plots')

    settings.lasot_path = os.path.join(_REPO_ROOT, 'data', 'lasot')
    settings.lasot_lmdb_path = os.path.join(_REPO_ROOT, 'data', 'lasot_lmdb')
    settings.lasot_extension_subset_path = os.path.join(_REPO_ROOT, 'data', 'lasot_extension_subset')
    settings.lasotlang_path = os.path.join(_REPO_ROOT, 'data', 'lasot')
    settings.got10k_path = os.path.join(_REPO_ROOT, 'data', 'got10k')
    settings.got10k_lmdb_path = os.path.join(_REPO_ROOT, 'data', 'got10k_lmdb')
    settings.trackingnet_path = os.path.join(_REPO_ROOT, 'data', 'trackingnet')
    settings.tnl2k_path = os.path.join(_REPO_ROOT, 'data', 'tnl2k', 'test')
    settings.otb_path = os.path.join(_REPO_ROOT, 'data', 'OTB2015')
    settings.otblang_path = os.path.join(_REPO_ROOT, 'data', 'otb_lang')
    settings.uav_path = os.path.join(_REPO_ROOT, 'data', 'UAV123')
    settings.nfs_path = os.path.join(_REPO_ROOT, 'data', 'nfs')
    settings.tc128_path = os.path.join(_REPO_ROOT, 'data', 'TC128')
    settings.vot_path = os.path.join(_REPO_ROOT, 'data', 'VOT2019')

    settings.tpl_path = ''
    settings.davis_dir = ''
    settings.youtubevos_dir = ''
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.tn_packed_results_path = ''

    return settings
