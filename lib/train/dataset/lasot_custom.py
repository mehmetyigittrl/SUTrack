import os

import pandas

from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings
from .lasot import Lasot

DEFAULT_SPLIT_FILE = 'lasot_custom_train_split.txt'


class LasotCustom(Lasot):
    """ LaSOT-layout dataset built from custom videos by tools/coco2lasot.py.

    Identical to Lasot (same folder contract, same task name 'lasot' so
    MODEL.TASK_INDEX keeps mapping it to the LASOT task token) except that
    the train split is read from data_specs/lasot_custom_train_split.txt and
    the default root is env_settings().lasot_custom_dir.
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, vid_ids=None, split=None,
                 data_fraction=None, multi_modal_vision=False, multi_modal_language=False,
                 use_nlp=False, split_file=None):
        """
        args:
            root - path to the converted LaSOT-layout dataset
                   (default: env_settings().lasot_custom_dir).
            split_file - optional explicit split file path (default:
                   lib/train/data_specs/lasot_custom_train_split.txt).
            remaining args: see Lasot.
        """
        self.custom_split_file = split_file
        root = env_settings().lasot_custom_dir if root is None else root
        super().__init__(root, image_loader, vid_ids, split, data_fraction,
                         multi_modal_vision, multi_modal_language, use_nlp)

    def _build_sequence_list(self, vid_ids=None, split=None):
        if split != 'train':
            raise ValueError('LasotCustom only supports split="train".')
        if vid_ids is not None:
            raise ValueError('Cannot set both split and vid_ids.')
        if self.custom_split_file is not None:
            file_path = self.custom_split_file
        else:
            ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
            file_path = os.path.join(ltr_path, 'data_specs', DEFAULT_SPLIT_FILE)
        # sequence_list = pandas.read_csv(file_path, header=None, squeeze=True).values.tolist()
        sequence_list = pandas.read_csv(file_path, header=None).squeeze("columns").values.tolist()

        return sequence_list
