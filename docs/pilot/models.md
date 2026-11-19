# Pilot Models

This page lists the two models of the pilot experiment, their sizes, and the design choices that the sources leave open. Parameter counts are measured from the code for 25 channels, 10 auxiliary channels, 5 frames and the 32 x 64 grid.

## WIND baseline

The WIND model is written from the published description and the public repository, in our own files under `src/spheridiff/models/wind/`. It is a U-ViT with four levels (two residual convolution levels and two transformer levels), one skip per level, a learned constant in place of a noise embedding, and 3D rotary embeddings in the transformer levels.

| Name | Widths | Blocks per level (plus middle) | Parameters |
|---|---|---|---|
| `wind_s` | 64, 128, 128, 128 | 1, 1, 1 (plus 1) | 3.01M |
| `wind_m` | 64, 128, 256, 256 | 1, 1, 1 (plus 1) | 7.34M |
| `wind_l` | 64, 128, 256, 512 | 1, 1, 1 (plus 1) | 12.24M |

Common settings: embedding width 64, 4 heads, patch size 1, 32 normalisation groups. The full-size configuration of the same code has about 458M parameters, which agrees with the published count.

The resampling layers require that each width divides four times the width above it, and the residual levels require widths divisible by 32. The earlier draft widths (64, 96, 128, 192) violate the first rule, so they were replaced.

## SpherIDiff

SpherIDiff is implemented in `src/spheridiff/models/spheridiff/`. A block applies a spectral stage, a channel MLP and a temporal attention across the frames. The spectral stage transforms each channel to spherical harmonics, mixes the coefficients with one weight matrix that is shared over channels and frames, and transforms back. The input is band-limited once and the noise is isotropic in the spectral domain.

| Name | Widths | Blocks | Parameters |
|---|---|---|---|
| `spheridiff_b4` | 128, 256, 256, 256, 128 | 4 | 1.60M |

For the 32 x 64 grid, L = 16, so K = 256 and each spectral weight has 65,536 parameters. Pretraining uses the whole sphere, so the Slepian projection is the identity and is not applied.

The proposal leaves some details open. The implementation uses these choices:

- Each of the three stages of a block is preceded by an RMS normalisation and added back as a residual. When the width changes, the MLP residual uses a bias-free linear shortcut.
- The auxiliary channels are concatenated to the noisy input before the uplift.
- Branch outputs and the final downlift are initialised to zero, as in the WIND blocks, so the model starts as the zero function.
- Temporal attention has 4 heads and a rotary embedding over the frame index.
- The spectral transform runs in 32-bit precision.

## Speed notes

A profile on the pilot sizes showed that SpherIDiff was limited by memory traffic of element-wise operations on its full-resolution tensors, not by arithmetic. Two changes that keep the mathematics the same were applied: the RMS normalisation uses the fused `torch.nn.functional.rms_norm` when it exists (with the original formula as a fallback), and the five-frame temporal attention is written as two small `einsum` products with a rotary embedding on the frame axis. In a local test they gave about 1.95 times the training speed with the same outputs (relative difference 8e-7 in fp32, and no added error in fp16 beyond the fp16 error of the original). The spectral stage still runs in fp32. Running it in fp16 would be faster still, with a small extra error, and was not adopted.

## Size comparison

The compute per parameter differs between the models. SpherIDiff applies its channel layers at every grid point without downsampling, so its forward cost is several times that of `wind_s` at a lower parameter count. Parameters, measured speed and training time are all reported.
