
import numpy as np
from pathlib import Path
import torch

def split_and_save(opt_img, sar_img, label_img, patch=64, stride=64, out_dir="dataset"):
    Path(out_dir).mkdir(exist_ok=True)
    k = 0
    for y in range(0, opt_img.shape[-2] - patch + 1, stride):
        for x in range(0, opt_img.shape[-1] - patch + 1, stride):
            opt_patch = opt_img[..., y:y + patch, x:x + patch]
            sar_patch = sar_img[..., y:y + patch, x:x + patch]
            lab_patch = label_img[y:y + patch, x:x + patch]
            np.save(Path(out_dir) / f"opt_{k}.npy", opt_patch.astype(np.float32))
            np.save(Path(out_dir) / f"sar_{k}.npy", sar_patch.astype(np.float32))
            np.save(Path(out_dir) / f"lab_{k}.npy", lab_patch.astype(np.int16))
            k += 1
    print(f"Saved {k} patches to {out_dir}")

class PairDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.ids = sorted([p.stem.split('_')[1] for p in self.root.glob('opt_*.npy')])
    def __len__(self): return len(self.ids)
    def __getitem__(self, idx):
        i = self.ids[idx]
        opt = np.load(self.root / f"opt_{i}.npy")
        sar = np.load(self.root / f"sar_{i}.npy")
        lab = np.load(self.root / f"lab_{i}.npy")
        opt = torch.from_numpy(opt).unsqueeze(0) if opt.ndim == 2 else torch.from_numpy(opt)
        sar = torch.from_numpy(sar).unsqueeze(0) if sar.ndim == 2 else torch.from_numpy(sar)
        return opt.float(), sar.float(), torch.tensor(lab).long().view(-1)[0]
