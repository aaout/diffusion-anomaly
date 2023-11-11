import torch
import torch.nn
import numpy as np
import os
import os.path
import sys
import nibabel
from scipy import ndimage


class BRATSDataset(torch.utils.data.Dataset):
    def __init__(self, directory, test_flag=False):
        """
        directory is expected to contain some folder structure:
                if some subfolder contains only files, all of these
                files are assumed to have a name like
                brats_train_001_XXX_123_w.nii.gz
                where XXX is one of t1, t1ce, t2, flair, seg
                we assume these five files belong to the same image
                seg is supposed to contain the segmentation
        """
        super().__init__()
        self.directory = os.path.expanduser(directory)

        self.test_flag = test_flag
        if test_flag:
            self.seqtypes = ["t1", "t1ce", "t2", "flair"]
        else:
            self.seqtypes = ["t1", "t1ce", "t2", "flair", "seg"]

        self.seqtypes_set = set(self.seqtypes)
        self.database = []
        for root, dirs, files in os.walk(self.directory):
            # if there are no subdirs, we have data
            if not dirs:
                files.sort()
                datapoint = dict()
                # extract all files as channels
                for f in files:
                    seqtype = f.split("_")[3]
                    datapoint[seqtype] = os.path.join(root, f)
                assert (
                    set(datapoint.keys()) == self.seqtypes_set
                ), f"datapoint is incomplete, keys are {datapoint.keys()}"
                self.database.append(datapoint)

    def __getitem__(self, x):
        """1つのindexに対し, 入力画像, ラベル画像, 弱ラベル(normal or abnormal), ファイル名を返す

        Args:
            x: index

        Returns:
            image: input test data [1, 4, 256, 256]
            out_dict: label dict {'y': tensor([1])} 1->diseased, 0->healthy
            weak_label: weak label 1->diseased, 0->healthy
            label: label img data [1, 1, 240, 240]
            number: file name tuple: ('BraTS20_Training_349_t1_099.nii.gz',)
        """

        out = []
        filedict = self.database[x]
        for seqtype in self.seqtypes:
            number = filedict["t1"].split("/")[-1]
            nib_img = nibabel.load(filedict[seqtype])
            out.append(torch.tensor(nib_img.get_fdata()))
        out = torch.stack(out)
        out_dict = {}
        if self.test_flag:
            input_test_data_path = filedict["t1"]
            seg_test_data_path = input_test_data_path.replace("t1", "seg")
            seg_test_data_path = seg_test_data_path.replace("test", "test_labels")
            image = torch.zeros(4, 256, 256)
            image[:, 8:-8, 8:-8] = out
            seg = nibabel.load(seg_test_data_path)
            seg = torch.tensor(seg.get_fdata())
            seg_image = torch.zeros(256, 256)
            seg_image[8:-8, 8:-8] = seg
            label = seg_image

            if seg.max() > 0:
                weak_label = 1
            else:
                weak_label = 0
            out_dict["y"] = weak_label
        else:
            image = torch.zeros(4, 256, 256)
            image[:, 8:-8, 8:-8] = out[:-1, ...]  # pad to a size of (256,256)
            label = out[-1, ...][None, ...]
            if label.max() > 0:
                weak_label = 1
            else:
                weak_label = 0
            out_dict["y"] = weak_label

        return (image, out_dict, weak_label, label, number)

    def __len__(self):
        return len(self.database)
