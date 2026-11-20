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
| `spheridiff_xs` | 64, 128, 64 | 2 | 0.27M |
| `spheridiff_s` | 32, 64, 64, 64, 32 | 4 | 0.35M |
| `spheridiff_m` | 64, 128, 128, 128, 64 | 4 | 0.60M |
| `spheridiff_b4` | 128, 256, 256, 256, 128 | 4 | 1.60M |

The smaller variants (`xs`, `s`, `m`) keep the shape of `spheridiff_b4` (widths in the ratio 1, 2, 2, 2, 1, or 1, 2, 1 for two blocks) and every width is a power of two, so that the scaling series changes only the size. For the 32 x 64 grid, L = 16, so K = 256 and each spectral weight has 65,536 parameters. Pretraining uses the whole sphere, so the Slepian projection is the identity and is not applied.

The proposal leaves some details open. The implementation uses these choices:

- Each of the three stages of a block is preceded by an RMS normalisation and added back as a residual. When the width changes, the MLP residual uses a bias-free linear shortcut.
- The auxiliary channels are concatenated to the noisy input before the uplift.
- Branch outputs and the final downlift are initialised to zero, as in the WIND blocks, so the model starts as the zero function.
- Temporal attention has 4 heads and a rotary embedding over the frame index.
- The spectral transform runs in 32-bit precision.

## SpherIDiff at band limit 31

With L = 16, the band-limited target leaves part of every field out of reach. On one training year, the normalised RMSE of the data against its own band-limited version is about 0.27 at L = 16 and about 0.10 at L = 31. Three amendments make L = 31 possible within a small parameter budget. Each is a configuration option, so the configurations above keep the original model.

| Option | Where | Values |
|---|---|---|
| Band limit and analysis | `sphere: bandlimit, analysis` | `31`, `least_squares` (default: grid limit, `quadrature`) |
| Spectral weight | `architecture: rank` | `32` (default: none, dense weight) |
| Grid of the blocks | `architecture: internal_grid` | `gauss` (default: `data`) |
| Transform in the blocks | `architecture: transform` | `dense` (alternative: `separable`) |

- **Least-squares analysis.** On the equiangular grid with poles, quadrature supports only L = 16. At L = 31 (K = 961), the analysis is fitted by weighted least squares, with the Clenshaw-Curtis weights as row weights. The system has full rank, with a condition number of about 2.4, and synthesis followed by analysis gives the identity. The diffusion process uses the same transform, so the input is band-limited once at L = 31 and the spectral noise has K = 961 coefficients.
- **Diagonal plus low-rank weight.** Each block uses W = D + P Q^T, with D diagonal and P, Q of size K x r, which gives K + 2Kr parameters per block in place of K^2. D and Q start at zero, so the stage starts as the zero map, and P starts with small random values so that both factors receive gradients. The weight remains a full-rank K x K matrix. Its dense form (`dense_weight`) is what regional fine-tuning receives, so the Slepian projection U_M^T W U_M and the low-rank update W_0 + A B apply unchanged. A test checks that the structured weight and its dense form give the same output.
- **Internal Gauss grid.** The blocks run on a 32 x 64 Legendre-Gauss grid, where quadrature is exact up to L = 32. The merged input (noisy and auxiliary channels) is moved onto this grid once before the uplift, and the output back onto the data grid after the downlift. Both moves are exact for fields band-limited at L = 31. The noisy input is band-limited by construction, while the auxiliary channels are band-limited by the move.
- **Transform in the blocks.** The configurations use dense matrices on the Gauss grid. A separable form is also implemented: a Fourier transform along longitude followed by a Legendre contraction for each order, as in the spherical harmonic transform of torch-harmonics. A test checks that it gives the same coefficients and fields as the dense matrices. In a local test on the RTX 4050 (b4, batch size 8), it took 0.41 s per step against 0.34 s for the dense form and used more memory, because at this grid size the cost is set by memory traffic rather than arithmetic. It is therefore not used in the configurations.

| Name | Widths | Blocks | Parameters |
|---|---|---|---|
| `spheridiff_l31_xs` | 64, 128, 64 | 2 | 264,987 |
| `spheridiff_l31_s` | 32, 64, 64, 64, 32 | 4 | 335,901 |
| `spheridiff_l31_m` | 64, 128, 128, 128, 64 | 4 | 587,805 |
| `spheridiff_l31_b4` | 128, 256, 256, 256, 128 | 4 | 1,589,277 |

With a dense weight at L = 31, the same sizes would have about 1.99M, 3.78M, 4.03M and 5.03M parameters. The arithmetic of the spectral stage grows with K whatever the form of the weight, so a step at L = 31 is expected to take longer than at L = 16. The speed on the T4 has not been measured.

## Speed notes

A profile on the pilot sizes showed that SpherIDiff was limited by memory traffic of element-wise operations on its full-resolution tensors, not by arithmetic. Two changes that keep the mathematics the same were applied: the RMS normalisation uses the fused `torch.nn.functional.rms_norm` when it exists (with the original formula as a fallback), and the five-frame temporal attention is written as two small `einsum` products with a rotary embedding on the frame axis. In a local test they gave about 1.95 times the training speed with the same outputs (relative difference 8e-7 in fp32, and no added error in fp16 beyond the fp16 error of the original). The spectral stage still runs in fp32. Running it in fp16 would be faster still, with a small extra error, and was not adopted.

## Size comparison

The compute per parameter differs between the models. SpherIDiff applies its channel layers at every grid point without downsampling, so its forward cost is several times that of `wind_s` at a lower parameter count. Parameters, measured speed and training time are all reported.
