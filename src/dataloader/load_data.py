# -*- coding: utf-8 -*-
# Time    : 2023/10/30 16:03
# Author  : fanc
# File    : load_data.py

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import json
import SimpleITK as sitk
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

def split_data(data_dir, infos_name, filter_volume=0.0, rate=0.8, seed=1900):
    infos_path = os.path.join(data_dir, infos_name)
    if infos_name.lower().endswith(".csv"):
        infos = build_infos_from_metadata_csv(data_dir, infos_name)
    else:
        with open(infos_path, 'r', encoding='utf-8') as f:
            infos = json.load(f)

    valid_infos = []
    for x in infos:
        if float(x.get("volume", 1.0)) >= float(filter_volume):
            valid_infos.append(x)
    infos = valid_infos

    labels = np.array([int(i["label"]) for i in infos])
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=rate, random_state=seed)
    train_idx, test_idx = next(splitter.split(np.zeros(len(labels)), labels))
    train_infos = [infos[i] for i in train_idx]
    test_infos = [infos[i] for i in test_idx]
    return train_infos, test_infos

def _normalize_pid(pid):
    pid = str(pid).upper().strip().replace("_", "-").replace(" ", "-")
    if pid.startswith("LUNG1-"):
        suffix = pid.split("-")[-1]
        if suffix.isdigit():
            return f"LUNG1-{int(suffix):03d}"
    return pid

def _resolve_metadata_path(raw_path, data_dir, source_prefix):
    raw = str(raw_path).strip()
    raw = raw.replace(
        "/kaggle/input/nsclc-radiomics/",
        "/kaggle/input/datasets/umutkrdrms/nsclc-radiomics/"
    )
    if source_prefix and raw.startswith(source_prefix):
        raw = raw.replace(source_prefix, data_dir, 1)
    if raw.startswith("/kaggle/input/datasets/umutkrdrms/nsclc-radiomics/LUNG1-"):
        raw = raw.replace(
            "/kaggle/input/datasets/umutkrdrms/nsclc-radiomics/",
            "/kaggle/input/datasets/umutkrdrms/nsclc-radiomics/NSCLC-Radiomics/",
            1,
        )
    if os.path.exists(raw):
        return raw
    if os.path.exists(os.path.join(data_dir, raw)):
        return os.path.join(data_dir, raw)
    return raw

def _load_image_or_series(path_like):
    if os.path.isfile(path_like):
        if str(path_like).lower().endswith(".dcm"):
            # If a single DICOM file is provided, load the full series from its folder.
            folder = os.path.dirname(path_like)
            reader = sitk.ImageSeriesReader()
            series_ids = reader.GetGDCMSeriesIDs(folder)
            if series_ids:
                file_names = reader.GetGDCMSeriesFileNames(folder, series_ids[0])
                reader.SetFileNames(file_names)
                return reader.Execute()
        return sitk.ReadImage(path_like)
    if os.path.isdir(path_like):
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(path_like)
        if not series_ids:
            raise FileNotFoundError(f"No DICOM series found under: {path_like}")
        file_names = reader.GetGDCMSeriesFileNames(path_like, series_ids[0])
        reader.SetFileNames(file_names)
        return reader.Execute()
    raise FileNotFoundError(f"Path does not exist: {path_like}")

def build_infos_from_metadata_csv(data_dir, metadata_name):
    cfg_path = '/kaggle/working/KidneyStoneSC/configs/dataset.json'
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    metadata_path = os.path.join(data_dir, metadata_name)
    clinical_path = os.path.join(data_dir, cfg.get('clinical_dir', ''))
    df_meta = pd.read_csv(metadata_path)
    df_cli = pd.read_csv(clinical_path) if clinical_path.lower().endswith(".csv") else pd.read_excel(clinical_path)
    pid_col = cfg.get('pid_col', 'PatientID')
    if pid_col not in df_cli.columns:
        raise ValueError(f"Clinical file missing pid column '{pid_col}'")
    label_col = cfg.get("label_col", "deadstatus.event")
    if label_col not in df_cli.columns:
        raise ValueError(f"Clinical file missing label column '{label_col}'")

    df_lab = df_cli[[pid_col, label_col]].dropna()
    if df_lab.empty:
        raise ValueError(f"No non-null labels found for '{label_col}'")

    if label_col == "deadstatus.event":
        label_map = { _normalize_pid(r[pid_col]): int(r[label_col]) for _, r in df_lab.iterrows() }
        label_vocab = None
    else:
        label_values = df_lab[label_col].astype(str).fillna("Unknown")
        label_vocab = sorted(label_values.unique().tolist())
        label_to_idx = {v: i for i, v in enumerate(label_vocab)}
        label_map = { _normalize_pid(r[pid_col]): int(label_to_idx[str(r[label_col])]) for _, r in df_lab.iterrows() }
    source_prefix = cfg.get('metadata_source_prefix', '/kaggle/input/nsclc-radiomics/NSCLC-Radiomics')
    pid_key = cfg.get('metadata_pid_col', 'patient_id')
    ct_key = cfg.get('metadata_ct_col', 'ct_path')
    seg_key = cfg.get('metadata_seg_col', 'seg_path')
    infos = []
    for _, row in df_meta.iterrows():
        pid = _normalize_pid(row[pid_key])
        if pid not in label_map:
            continue
        label_name = None
        if label_vocab is not None:
            label_name = label_vocab[int(label_map[pid])]
        infos.append(
            {
                'pid': pid,
                'sid': pid,
                'label': int(label_map[pid]),
                'label_name': label_name,
                'volume': 1.0,
                'ct_path': _resolve_metadata_path(row[ct_key], data_dir, source_prefix),
                'seg_path': _resolve_metadata_path(row[seg_key], data_dir, source_prefix),
            }
        )
    return infos

# class MyDataset(Dataset):
#     def __init__(self, data_dir, infos, input_size, phase='train', task=[0, 1]):
#         '''
#         task: 0 :seg,  1 :cla
#         '''
#         task = [int(i) for i in re.findall('\d', str(task))]
#         img_dir = os.path.join(data_dir, 'imgs_nii')
#         mask_dir = os.path.join(data_dir, 'mask_nii')
#         self.seg = False
#         self.cla = False
#         if 0 in task:
#             self.seg = True
#         if 1 in task:
#             self.cla = True
#
#         self.input_size = tuple([int(i) for i in re.findall('\d+', str(input_size))])
#         self.img_dir = img_dir
#         if self.cla:
#             self.labels = [i['label'] for i in infos]
#         if self.seg:
#             self.mask_dir = mask_dir
#         # self.labels = [[1, 0] if int(i['label']) == 1 else [0, 1] for i in infos]
#
#         self.ids = [i['id'] for i in infos]
#         self.phase = phase
#         self.labels = torch.tensor(self.labels, dtype=torch.float)
#
#     def __len__(self):
#         return len(self.ids)
#
#     def __getitem__(self, i):
#         img = sitk.ReadImage(os.path.join(self.img_dir, f"{self.ids[i]}.nii.gz"))
#         if self.seg:
#             mask = sitk.ReadImage(os.path.join(self.mask_dir, f"{self.ids[i]}-mask.nii.gz"))
#         else:
#             mask = None
#         if self.phase == 'train':
#             img, mask = self.train_preprocess(img, mask)
#         else:
#             img, mask = self.val_preprocess(img, mask)
#         if self.cla:
#             label = self.labels[i]
#         else:
#             label = 1
#
#         img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
#         if self.seg:
#             mask = torch.tensor(mask, dtype=torch.uint8).unsqueeze(0)
#         if self.cla:
#             label = self.labels[i].unsqueeze(0)
#
#         return img, mask, label
#
#     def train_preprocess(self, img, mask):
#         img, mask = self.resample(itkimage=img, itkmask=mask)
#         # mask = self.resample(mask)
#         # print(img.shape, mask.shape)
#         if self.seg:
#             assert img.shape == mask.shape, "img and mask shape not match"
#             # img, mask = self.crop(img, mask)
#         img = self.normalize(img)
#         img, mask = self.resize(img, mask)
#
#         return img, mask
#     def val_preprocess(self, img, mask):
#         img, mask = self.resample(img, mask)
#         # mask = self.resample(mask)
#         if self.seg:
#             assert img.shape == mask.shape, "img and mask shape not match"
#         # img, mask = self.crop(img, mask)
#         img = self.normalize(img)
#         img, mask = self.resize(img, mask)
#
#         return img, mask
#
#     def crop(self, img, mask):
#         crop_img = img
#         crop_mask = mask
#         # amos kidney mask
#         crop_mask[crop_mask == 2] = 1
#         crop_mask[crop_mask != 1] = 0
#         target = np.where(crop_mask == 1)
#         [d, h, w] = crop_img.shape
#         [max_d, max_h, max_w] = np.max(np.array(target), axis=1)
#         [min_d, min_h, min_w] = np.min(np.array(target), axis=1)
#         [target_d, target_h, target_w] = np.array([max_d, max_h, max_w]) - np.array([min_d, min_h, min_w])
#         z_min = int((min_d - target_d / 2) * random.random())
#         y_min = int((min_h - target_h / 2) * random.random())
#         x_min = int((min_w - target_w / 2) * random.random())
#
#         z_max = int(d - ((d - (max_d + target_d / 2)) * random.random()))
#         y_max = int(h - ((h - (max_h + target_h / 2)) * random.random()))
#         x_max = int(w - ((w - (max_w + target_w / 2)) * random.random()))
#
#         z_min = np.max([0, z_min])
#         y_min = np.max([0, y_min])
#         x_min = np.max([0, x_min])
#
#         z_max = np.min([d, z_max])
#         y_max = np.min([h, y_max])
#         x_max = np.min([w, x_max])
#
#         z_min = int(z_min)
#         y_min = int(y_min)
#         x_min = int(x_min)
#
#         z_max = int(z_max)
#         y_max = int(y_max)
#         x_max = int(x_max)
#         crop_img = crop_img[z_min: z_max, y_min: y_max, x_min: x_max]
#         crop_mask = crop_mask[z_min: z_max, y_min: y_max, x_min: x_max]
#
#         return crop_img, crop_mask
#
#     def resample(self, itkimage, itkmask, new_spacing=[1, 1, 1]):
#         # spacing = itkimage.GetSpacing()
#         img = sitk.GetArrayFromImage(itkimage)
#         if self.seg:
#             mask = sitk.GetArrayFromImage(itkmask)
#         else:
#             mask = None
#         # # MASK 膨胀腐蚀操作
#         # kernel = ball(5)  # 3D球形核
#         # # 应用3D膨胀
#         # dilated_mask = dilation(mask, kernel)
#         # mask = closing(dilated_mask, kernel)
#         # resize_factor = spacing / np.array(new_spacing)
#         # resample_img = zoom(img, resize_factor, order=0)
#         # resample_mask = zoom(mask, resize_factor, order=0, mode='nearest')
#         return np.array(img, dtype=np.float32), np.array(mask, dtype=np.float32)
#
#     def normalize(self, img):
#
#         # CT值范围选取
#         min = 0
#         max = 2000
#         img[img < min] = min
#         img[img > max] = max
#
#         # std = np.std(img)
#         # avg = np.average(img)
#         # return (img - avg + std) / (std * 2)
#         return (img - min) / (max - min)
#
#     def resize(self, img, mask):
#         # img = np.transpose(img, (2, 1, 0))
#         # mask = np.transpose(mask, (2, 1, 0))
#         rate = np.array(self.input_size) / np.array(img.shape)
#         try:
#             img = zoom(img, rate.tolist(), order=0)
#             if self.seg:
#                 mask = zoom(mask, rate.tolist(), order=0, mode='nearest')
#         except Exception as e:
#             print(e)
#             img = resize(img, self.input_size)
#             if self.seg:
#                 mask = resize(mask, self.input_size, order=0)
#         # # MASK 膨胀腐蚀操作
#         # kernel = ball(5)  # 3D球形核
#         # # 应用3D膨胀
#         # dilated_mask = dilation(mask, kernel)
#         # mask = closing(dilated_mask, kernel)
#
#         # 高斯滤波去噪
#         # img = gaussian_filter(img, sigma=1)
#         # 中值滤波去噪
#         # from scipy.ndimage import median_filter
#         # img = median_filter(img, size=3)
#
#         return img, mask

class MyDataset(Dataset):
    def __init__(self, data_dir, infos, phase='train', clinical_preprocessor=None):
        with open('/kaggle/working/KidneyStoneSC/configs/dataset.json', 'r', encoding='utf-8') as f:
            config = json.load(f)

        # img_dir = os.path.join(data_dir, 'cropped_img')
        # mask_dir = os.path.join(data_dir, 'cropped_mask')
        img_dir = config['img_dir']
        mask_dir = config['mask_dir']
        data_dir = config['data_dir']
        self.pid_col = config.get('pid_col', 'pid')
        clinical_dir = config.get('clinical_dir', '')
        self.clinical_features = config.get('clinical_features', [])
        self.use_clinical = bool(clinical_dir)
        self.clinical_preprocessor = clinical_preprocessor
        self.clinical_dim = 0
        self.clinical_map = {}
        if self.use_clinical:
            clinical_path = os.path.join(data_dir, clinical_dir)
            if clinical_dir.lower().endswith('.csv'):
                clinical_df = pd.read_csv(clinical_path)
            else:
                clinical_df = pd.read_excel(clinical_path)
            if self.pid_col not in clinical_df.columns:
                raise ValueError(f"pid_col '{self.pid_col}' not found in clinical file")
            if not self.clinical_features:
                reserved_cols = {
                    self.pid_col, 'label', 'deadstatus.event', 'Survival.time',
                    'PatientID', 'pid', 'id'
                }
                self.clinical_features = [c for c in clinical_df.columns if c not in reserved_cols]
            self.clinical_map = self._prepare_clinical_map(clinical_df)
            self.clinical_dim = len(next(iter(self.clinical_map.values()))) if self.clinical_map else 0
        self.img_dir = os.path.join(data_dir, img_dir)
        self.labels = [i['label'] for i in infos]
        self.mask_dir = os.path.join(data_dir, mask_dir)
        self.ids = [i['sid'] for i in infos]
        self.pids = [i['pid'] for i in infos]
        self.ct_paths = [i.get('ct_path') for i in infos]
        self.seg_paths = [i.get('seg_path') for i in infos]
        self.phase = phase
        self.labels = torch.tensor(self.labels, dtype=torch.float)
        self.target_size = tuple(config.get("target_size", [96, 96, 96]))

    def _prepare_clinical_map(self, clinical_df):
        df = clinical_df.copy()
        for c in self.clinical_features:
            if c not in df.columns:
                df[c] = np.nan
        feat_df = df[[self.pid_col] + self.clinical_features].copy()
        numeric_cols = [c for c in self.clinical_features if pd.api.types.is_numeric_dtype(feat_df[c])]
        categorical_cols = [c for c in self.clinical_features if c not in numeric_cols]

        if self.clinical_preprocessor is None:
            numeric_medians = {c: float(feat_df[c].median()) if not feat_df[c].dropna().empty else 0.0 for c in numeric_cols}
            numeric_means = {}
            numeric_stds = {}
            for c in numeric_cols:
                filled = feat_df[c].fillna(numeric_medians[c]).astype(float)
                numeric_means[c] = float(filled.mean())
                std = float(filled.std())
                numeric_stds[c] = std if std > 1e-6 else 1.0
            cat_vocab = {}
            for c in categorical_cols:
                values = feat_df[c].fillna('Unknown').astype(str)
                cat_vocab[c] = sorted(values.unique().tolist())
            self.clinical_preprocessor = {
                'numeric_cols': numeric_cols,
                'categorical_cols': categorical_cols,
                'numeric_medians': numeric_medians,
                'numeric_means': numeric_means,
                'numeric_stds': numeric_stds,
                'cat_vocab': cat_vocab,
            }

        clinical_map = {}
        for _, row in feat_df.iterrows():
            pid = str(row[self.pid_col])
            vec = []
            for c in self.clinical_preprocessor['numeric_cols']:
                value = row[c] if c in row.index else np.nan
                if pd.isna(value):
                    value = self.clinical_preprocessor['numeric_medians'][c]
                value = float(value)
                value = (value - self.clinical_preprocessor['numeric_means'][c]) / self.clinical_preprocessor['numeric_stds'][c]
                vec.append(value)
            for c in self.clinical_preprocessor['categorical_cols']:
                value = row[c] if c in row.index else 'Unknown'
                value = 'Unknown' if pd.isna(value) else str(value)
                vocab = self.clinical_preprocessor['cat_vocab'][c]
                vec.extend([1.0 if value == v else 0.0 for v in vocab])
            clinical_map[pid] = np.array(vec, dtype=np.float32)
        return clinical_map

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        if self.ct_paths[i]:
            img = _load_image_or_series(self.ct_paths[i])
        else:
            img = sitk.ReadImage(os.path.join(self.img_dir, f"{self.ids[i]}.nii.gz"))
        if self.seg_paths[i]:
            mask = _load_image_or_series(self.seg_paths[i])
        else:
            mask = sitk.ReadImage(os.path.join(self.mask_dir, f"{self.ids[i]}.nii.gz"))
        if self.phase == 'train':
            img, mask = self.train_preprocess(img, mask)
        else:
            img, mask = self.val_preprocess(img, mask)
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask, dtype=torch.uint8).unsqueeze(0)
        # Resize large DICOM volumes to a fixed tensor size for stable GPU memory usage.
        if self.target_size:
            img = F.interpolate(
                img.unsqueeze(0),
                size=self.target_size,
                mode="trilinear",
                align_corners=False
            ).squeeze(0)
            mask = F.interpolate(
                mask.unsqueeze(0).float(),
                size=self.target_size,
                mode="nearest"
            ).squeeze(0).to(torch.uint8)
        label = self.labels[i].unsqueeze(0)

        if self.use_clinical and self.clinical_dim > 0:
            pid = str(self.pids[i])
            if pid in self.clinical_map:
                clinical = torch.tensor(self.clinical_map[pid], dtype=torch.float32)
            else:
                clinical = torch.zeros(self.clinical_dim, dtype=torch.float32)
        else:
            clinical = torch.zeros(1, dtype=torch.float32)

        return img, mask, label, clinical

    def train_preprocess(self, img, mask):
        img, mask = self.resample(itkimage=img, itkmask=mask)
        img = self.normalize(img)
        return img, mask
    def val_preprocess(self, img, mask):
        img, mask = self.resample(img, mask)
        img = self.normalize(img)
        return img, mask


    def resample(self, itkimage, itkmask):
        # spacing = itkimage.GetSpacing()
        img = sitk.GetArrayFromImage(itkimage)
        mask = sitk.GetArrayFromImage(itkmask)


        return np.array(img, dtype=np.float32), np.array(mask, dtype=np.float32)

    def normalize(self, img):
        # CT值范围选取
        min = -400
        max = 2000
        img[img < min] = min
        img[img > max] = max

        return (img - min) / (max - min)


def my_dataloader(data_dir, infos, batch_size=1, shuffle=True, num_workers=0, phase='train', clinical_preprocessor=None):
    dataset = MyDataset(data_dir, infos, phase=phase, clinical_preprocessor=clinical_preprocessor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

if __name__ == '__main__':

    data_dir = r'C:\Users\Asus\Desktop\KidneyStone\data'
    train_info, test_info = split_data(data_dir, rate=0.8)
    train_dataloader = my_dataloader(data_dir, train_info, batch_size=1)
    test_dataloader = my_dataloader(data_dir, test_info, batch_size=1)
    for i, (image, mask, label) in enumerate(train_dataloader):
        print(image, mask.max(), label.shape, label[:, 0].shape)

        new_image = sitk.GetImageFromArray(image.numpy()[0][0])
        new_image.SetSpacing([1, 1, 1])
        sitk.WriteImage(new_image, os.path.join(data_dir, f'{i}.nii.gz'))

        new_mask = sitk.GetImageFromArray(mask.numpy()[0][0])
        new_mask.SetSpacing([1, 1, 1])
        sitk.WriteImage(new_mask, os.path.join(data_dir, f'{i}-mask.nii.gz'))
        break

        # nifti_image = nib.Nifti1Image(image.numpy()[0][0], affine=None)
        # nib.save(nifti_image, os.path.join(data_dir, f'process_img_{i}.nii.gz'))
        # nifti_image = nib.Nifti1Image(mask.numpy()[0][0], affine=None)
        # nib.save(nifti_image, os.path.join(data_dir, f'process_mask_{i}.nii.gz'))
    #

    # for i, (image, mask, label) in enumerate(test_dataloader):
    #     print(i,  image.shape, mask.shape, label)
