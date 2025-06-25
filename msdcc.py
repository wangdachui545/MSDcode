
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
from einops import rearrange
import pytorch_lightning as pl

# ---------- 多尺度分解层 ----------
class WaveletDecompose(nn.Module):
    def __init__(self, wave="db4", level=3):
        super().__init__()
        self.wave, self.level = wave, level
    def forward(self, x):
        coeffs = pywt.wavedec2(x.squeeze(1).cpu().numpy(), self.wave, level=self.level)
        tensors = [torch.from_numpy(coeffs[0]).unsqueeze(1)]
        for (lh, hl, hh) in coeffs[1:]:
            tensors += [torch.from_numpy(lh).unsqueeze(1),
                        torch.from_numpy(hl).unsqueeze(1),
                        torch.from_numpy(hh).unsqueeze(1)]
        return torch.cat([t.to(x.device).float() for t in tensors], dim=1)

class LaplacianPyramid(nn.Module):
    def __init__(self, level=3):
        super().__init__()
        self.level = level
        self.pool = nn.AvgPool2d(2, 2)
    def forward(self, x):
        pyr, cur = [], x
        for _ in range(self.level):
            down = self.pool(cur)
            up = F.interpolate(down, size=cur.shape[-2:], mode="bilinear", align_corners=False)
            lap = cur - up
            pyr.append(lap)
            cur = down
        pyr.append(cur)
        return torch.cat(pyr, 1)

class SAFE(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, 1, 1),
                                  nn.BatchNorm2d(out_ch), nn.ReLU())
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                nn.Conv2d(out_ch, out_ch // 4, 1), nn.ReLU(),
                                nn.Conv2d(out_ch // 4, out_ch, 1), nn.Sigmoid())
    def forward(self, x):
        y = self.conv(x)
        return y * self.se(y)

class SAFusion(nn.Module):
    def __init__(self, n_scale=2, feat_ch=64):
        super().__init__()
        self.n_scale = n_scale
        self.feat_ch = feat_ch
        self.c_att = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                   nn.Conv2d(feat_ch * n_scale, feat_ch // 4, 1), nn.ReLU(),
                                   nn.Conv2d(feat_ch // 4, feat_ch * n_scale, 1), nn.Sigmoid())
        self.scale_w = nn.Parameter(torch.ones(n_scale))
    def forward(self, feats):
        x = torch.cat(feats, dim=1)
        x = x * self.c_att(x)
        x = rearrange(x, "b (s c) h w -> b s c h w", s=self.n_scale)
        w = torch.softmax(self.scale_w, 0).view(1, -1, 1, 1, 1)
        return (x * w).sum(1)

class MSDCC(pl.LightningModule):
    def __init__(self, n_cls=10):
        super().__init__()
        self.wave = WaveletDecompose()
        self.lap = LaplacianPyramid()
        self.safe_opt = SAFE(10, 64)
        self.safe_sar = SAFE(4, 64)
        self.fuse = SAFusion()
        self.head_main = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(),
                                       nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                       nn.Linear(128, n_cls))
        self.head_aux1 = nn.Linear(64, n_cls)
        self.head_aux2 = nn.Linear(64, n_cls)
        self.criterion = nn.CrossEntropyLoss()
    def forward(self, opt, sar):
        f_opt = self.safe_opt(self.wave(opt))
        f_sar = self.safe_sar(self.lap(sar))
        fused = self.fuse([f_opt, f_sar])
        y0 = self.head_main(fused)
        y1 = self.head_aux1(f_opt.mean((2, 3)))
        y2 = self.head_aux2(f_sar.mean((2, 3)))
        return y0, y1, y2
    def training_step(self, batch, _):
        opt, sar, lab = batch
        y0, y1, y2 = self(opt, sar)
        loss = self.criterion(y0, lab) + 0.3 * (
            self.criterion(y1, lab) + self.criterion(y2, lab))
        self.log("train_loss", loss)
        return loss
    def validation_step(self, batch, _):
        opt, sar, lab = batch
        y0, *_ = self(opt, sar)
        acc = (y0.argmax(1) == lab).float().mean()
        self.log("val_acc", acc, prog_bar=True)
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
