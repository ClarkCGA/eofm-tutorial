import os
from torch.utils.data import Dataset
import torch
from .utils import *
from .augmentation import *
from .tools import setup_logger, progress_reporter


class ImageDataSSL(Dataset):
    """
    On-the-fly Dataset for self-supervised diffusion training (image-only, no labels).
    Dynamically loads .tif files during iteration.
    """

    def __init__(self, data_path, log_dir, catalog, data_size, buffer, buffer_comp,
                 usage, img_path_cols, apply_normalization=True, normal_strategy="min_max",
                 stat_procedure="lab", global_stats=None, trans=None, parallel=False, verbose=True, **kwargs):

        self.data_path = data_path
        self.log_dir = log_dir
        self.data_size = data_size
        self.buffer = buffer
        self.composite_buffer = buffer_comp
        self.usage = usage
        self.img_cols = img_path_cols if isinstance(img_path_cols, list) else [img_path_cols]
        self.apply_normalization = apply_normalization
        self.normal_strategy = normal_strategy
        self.stat_procedure = stat_procedure
        self.global_stats = global_stats
        self.trans = trans
        self.parallel = parallel
        self.kwargs = kwargs
        self.downfactor = self.kwargs.get("downfactor", 32)
        self.nodata = self.kwargs.get("nodata", None)
        self.clip_val = self.kwargs.get("clip_val", None)
        self.chip_size = self.data_size + self.buffer * 2
        self.verbose = verbose

        self.logger = setup_logger(self.log_dir, f"{self.usage}_ssl_dataset", use_date=False)
        progress_reporter(f'Started dataset creation for {self.usage}', verbose=self.verbose, logger=self.logger)

        self.catalog = catalog.loc[catalog['usage'] == self.usage].copy()
        self.img_paths = self.catalog[self.img_cols].values.tolist()

        progress_reporter(f'Completed dataset creation for {self.usage}', verbose=self.verbose, logger=self.logger)

    def __getitem__(self, index):
        img_entry = self.img_paths[index]
        dir_imgs = [os.path.join(self.data_path, p) for p in img_entry]

        # Load and preprocess the image on-the-fly
        window = get_buffered_window(dir_imgs[0], dir_imgs[0], self.buffer)
        img = process_img(
            dir_imgs, self.usage, apply_normalization=self.apply_normalization,
            normal_strategy=self.normal_strategy, stat_procedure=self.stat_procedure,
            global_stats=self.global_stats, window=window,
            nodata_val_ls=self.nodata, clip_val=self.clip_val
        )

        img = torch.from_numpy(img.transpose((2, 0, 1)).copy()).float()
        return img

    def __len__(self):
        return len(self.img_paths)
