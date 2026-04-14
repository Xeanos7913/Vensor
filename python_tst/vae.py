# python pytorch implementation of the VAE in main.cpp

"""
PyTorch VAE — exact port of the C++ custom-engine implementation.

Architecture notes (derived from the C++ source):
  Encoder conv stack:
    Conv2d(1,  32, k=4, stride=2, pad=1)  => floor((28+2-4)/2)+1 = 14x14
    ReLU
    Conv2d(32, 64, k=4, stride=2, pad=1)  => floor((14+2-4)/2)+1 = 7x7
    ReLU
    Linear(64*7*7=3136, 512)
  Two parallel heads:
    fc_mu    : Linear(512, 256)
    fc_logvar: Linear(512, 256)

  Reparameterization: z = mu + exp(0.5*logvar) * eps,  eps~N(0,1)

  Decoder:
    Linear(256, 3136) => ReLU => reshape(64, 7, 7)
    Upsample(7->14)
    Conv2d(64,32,k=3,s=1,p=1) => 14x14 => ReLU
    Upsample(14->28)
    Conv2d(32, 1,k=3,s=1,p=1) => 28x28 => Tanh

  Loss: MSE(recon, x) + KLD
    KLD = -0.5 * mean(1 + logvar - mu^2 - exp(logvar))

  Key insight: the C++ Conv2d constructor signature is:
    Conv2d(..., stride_w=1, stride_h=1, pad_h=1, pad_w=1, ...)
  The VAE passes only stride (2,2) for encoder convs, so pad defaults to 1,1.
  The decoder convs explicitly pass (stride=1,1, pad=1,1).
  This pad=1 default is what yields 14x14 -> 7x7 -> ... -> 28x28.

Usage:
  python vae_pytorch.py          # trains for 10 epochs, saves recon PNGs
  python vae_pytorch.py --epochs 50 --batch 16 --lr 1e-4
"""

import argparse
import time
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image


# ---------------------------------------------------------------------------
# MNIST raw-file loader (mirrors your MNISTDataloader)
# ---------------------------------------------------------------------------

class MNISTDataset(Dataset):
    """
    Reads the raw IDX binary files directly — no torchvision dependency.
    Pixel values are normalised to [-1, 1] to match the Tanh output range.
    (If you used [0,1] normalisation in your C++ loader, change the line
     marked NORMALISATION below.)
    """

    def __init__(self, image_path: str, label_path: str):
        images = self._read_images(image_path)
        labels = self._read_labels(label_path)
        # NORMALISATION: C++ code doesn't show explicit normalisation,
        # but Tanh outputs [-1,1], so we map [0,255] -> [-1,1].
        # Change to  / 255.0  if your loader normalises to [0,1].
        self.images = torch.tensor(images, dtype=torch.float32).unsqueeze(1) / 127.5 - 1.0
        self.labels = torch.tensor(labels, dtype=torch.long)

    @staticmethod
    def _read_images(path):
        with open(path, "rb") as f:
            magic, n, h, w = struct.unpack(">IIII", f.read(16))
            assert magic == 2051, f"Bad magic {magic}"
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, h, w)

    @staticmethod
    def _read_labels(path):
        with open(path, "rb") as f:
            magic, n = struct.unpack(">II", f.read(8))
            assert magic == 2049, f"Bad magic {magic}"
            return np.frombuffer(f.read(), dtype=np.uint8)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class VAE(nn.Module):
    """
    Exact port of the C++ VAE struct.

    Encoder spatial flow (pad=1 is the Conv2d default in the C++ engine):
      28x28 --Conv(k4,s2,p1)--> 14x14 --Conv(k4,s2,p1)--> 7x7 --flatten--> 3136
      3136  --Linear----------> 512
      512   --fc_mu-----------> 256   (mu)
      512   --fc_logvar--------> 256   (logvar)

    Decoder:
      256  --Linear-------> 3136  --ReLU--> reshape(64, 7, 7)
      7x7  --Upsample(2)--> 14x14 --Conv(k3,s1,p1)--> 14x14 --ReLU-->
      14x14--Upsample(2)--> 28x28 --Conv(k3,s1,p1)--> 28x28 --Tanh-->
    """

    # Derived spatial dimensions — kept as class constants for clarity.
    # Conv2d default pad=1, so: floor((28 + 2*1 - 4) / 2) + 1 = 14
    #                           floor((14 + 2*1 - 4) / 2) + 1 = 7
    ENC_H1, ENC_W1 = 14, 14   # after first encoder conv  (k=4, s=2, p=1)
    ENC_H2, ENC_W2 =  7,  7   # after second encoder conv (k=4, s=2, p=1)
    DEC_H1, DEC_W1 = 14, 14   # after first upsample  (7 * 2)
    DEC_H2, DEC_W2 = 28, 28   # after second upsample (14 * 2) -- matches input

    LATENT_DIM  = 256
    ENC_FLAT    = 64 * ENC_H2 * ENC_W2   # 64 * 7 * 7 = 3136

    def __init__(self):
        super().__init__()

        # ---- Encoder ------------------------------------------------
        # C++ Conv2d defaults: pad_h=1, pad_w=1. The VAE passes only stride
        # for encoder convs, so padding stays at the default of 1.
        # bias=True mirrors the C++ ctor which always allocates a bias tensor.
        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.ENC_FLAT, 512, bias=True),
        )

        self.fc_mu     = nn.Linear(512, self.LATENT_DIM, bias=True)
        self.fc_logvar = nn.Linear(512, self.LATENT_DIM, bias=True)

        # ---- Decoder ------------------------------------------------
        self.dec_fc = nn.Sequential(
            nn.Linear(self.LATENT_DIM, self.ENC_FLAT, bias=True),
            nn.ReLU(),
        )

        self.dec_conv = nn.Sequential(
            nn.Upsample(size=(self.DEC_H1, self.DEC_W1), mode="nearest"),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Upsample(size=(self.DEC_H2, self.DEC_W2), mode="nearest"),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Tanh(),
        )

        # Xavier uniform init on the final conv — mirrors the explicit
        # tensor_fill_random(dec_conv2->weight, mode=2, fan_in, fan_out) call.
        nn.init.xavier_uniform_(self.dec_conv[-2].weight)

    # ------------------------------------------------------------------ #
    # Forward sub-steps (mirroring your named methods)
    # ------------------------------------------------------------------ #

    def encode(self, x: torch.Tensor):
        """x: (B, 1, 28, 28) -> mu, logvar: (B, LATENT_DIM)"""
        h = self.enc_conv(x)          # (B, 512)
        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        """
        z = mu + std * eps,  eps ~ N(0, I)
        std = exp(0.5 * logvar)
        Identical to your reparameterize() method.
        During inference (eval mode) we just return mu for a deterministic
        reconstruction, but keeping it stochastic here to match training.
        """
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)   # same shape as std, device-local N(0,1)
        return mu + std * eps

    def decode(self, z: torch.Tensor):
        """z: (B, LATENT_DIM) -> recon: (B, 1, 20, 20)"""
        h = self.dec_fc(z)                                  # (B, 1600)
        h = h.view(-1, 64, self.ENC_H2, self.ENC_W2)       # (B, 64, 5, 5)
        return self.dec_conv(h)                             # (B, 1, 20, 20)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


# ---------------------------------------------------------------------------
# Loss  (mirrors your MSEloss + KLDloss)
# ---------------------------------------------------------------------------

def vae_loss(recon: torch.Tensor,
             x: torch.Tensor,
             mu: torch.Tensor,
             logvar: torch.Tensor) -> torch.Tensor:
    """
    MSE reconstruction loss + KL divergence.

    MSE: F.mse_loss(recon, x, reduction='mean') — mean over all elements,
    matching the mean-reduced MSEloss in the C++ engine.
    recon and x are both (B, 1, 28, 28) so no spatial alignment is needed.

    KLD = -0.5 * mean( 1 + logvar - mu^2 - exp(logvar) )
    This is the analytical KL divergence between N(mu, sigma^2) and N(0,I),
    summed over the latent dimension and mean-reduced over the batch.
    """
    mse = F.mse_loss(recon, x, reduction="mean")

    # Standard VAE KLD against N(0,I)
    kld = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())

    return mse + kld


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    dataset = MNISTDataset(args.images, args.labels)
    loader  = DataLoader(dataset, batch_size=args.batch,
                         shuffle=True, num_workers=2, pin_memory=True,
                         drop_last=True)

    # Model + optimiser
    model = VAE().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=1e-2)   # AdamW default wd

    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        fwd_total = bwd_total = opt_total = 0.0
        t_epoch = time.perf_counter()

        for x, _ in loader:
            x = x.to(device)

            # ---- forward ----
            t0 = time.perf_counter()
            recon, mu, logvar = model(x)
            loss = vae_loss(recon, x, mu, logvar)
            t1 = time.perf_counter()

            # ---- backward ----
            t2 = time.perf_counter()
            optim.zero_grad()
            loss.backward()
            t3 = time.perf_counter()

            # ---- optimiser step ----
            t4 = time.perf_counter()
            optim.step()
            t5 = time.perf_counter()

            fwd_total += (t1 - t0) * 1000
            bwd_total += (t3 - t2) * 1000
            opt_total += (t5 - t4) * 1000

            epoch_loss += loss.item()
            n_batches  += 1

        epoch_ms = (time.perf_counter() - t_epoch) * 1000

        print(f"Epoch {epoch:>3d}/{args.epochs} | "
              f"loss: {epoch_loss / n_batches:.6f} | "
              f"epoch: {epoch_ms:.1f} ms | "
              f"avg fwd: {fwd_total/n_batches:.2f} ms | "
              f"avg bwd: {bwd_total/n_batches:.2f} ms | "
              f"avg opt: {opt_total/n_batches:.2f} ms")

        # Save reconstructions (last batch)
        model.eval()
        with torch.no_grad():
            recon_vis, _, _ = model(x[:16])
            # rescale [-1,1] -> [0,1] for PNG
            recon_vis = (recon_vis + 1.0) / 2.0
            save_image(recon_vis, f"recon_{epoch}.png", nrow=4)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", default="dataset/train.idx3-ubyte")
    p.add_argument("--labels", default="dataset/labels.idx1-ubyte")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch",  type=int, default=16)
    p.add_argument("--lr",     type=float, default=1e-4)
    args = p.parse_args()

    train(args)


if __name__ == "__main__":
    main()