shift 0

shift                |     25.802     0.9145 |     0.0927        0.6424

============================================================
EXPERIMENT: shift_a5.0_p1.0__e3_r64_la32_lr5e-05_ga4_attn+ff+conv_seed42
============================================================

--- GROUP 1: WATERMARK VISIBILITY (training data vs originals) ---
  n_compared : 559
  avg PSNR   : 25.802 dB  (higher = less visible, >40 is imperceptible)
  avg SSIM   : 0.9145     (higher = more similar to original, 1.0 = identical)

--- GROUP 2: MODEL DETERIORATION (generated images vs clean baseline) ---
  all images (n=110)
    clip content  : 28.032
    clip style    : 25.926
    clip combined : 26.979
    brightness    : 107.808
    sharpness     : 1110.732
  seeded vs clean baseline (n=10)
    avg SSIM      : 0.6424  (higher = more similar to clean)
    avg CLIP drift: 0.0927  (lower = more similar to clean)




shift 1

shift                |     25.646     0.9122 |     0.1502         0.456

============================================================
EXPERIMENT: shift_a5.0_p1.0__e3_r64_la32_lr5e-05_ga4_attn+ff+conv_seed42
============================================================

--- GROUP 1: WATERMARK VISIBILITY (training data vs originals) ---
  n_compared : 559
  avg PSNR   : 25.646 dB  (higher = less visible, >40 is imperceptible)
  avg SSIM   : 0.9122     (higher = more similar to original, 1.0 = identical)

--- GROUP 2: MODEL DETERIORATION (generated images vs clean baseline) ---
  all images (n=110)
    clip content  : 28.47
    clip style    : 25.573
    clip combined : 27.022
    brightness    : 107.206
    sharpness     : 1777.246
  seeded vs clean baseline (n=10)
    avg SSIM      : 0.456  (higher = more similar to clean)
    avg CLIP drift: 0.1502  (lower = more similar to clean)


shift 2

shift                |     25.671     0.9042 |     0.1288        0.6475

============================================================
EXPERIMENT: shift_a5.0_p1.0__e3_r64_la32_lr5e-05_ga4_attn+ff+conv_seed42
============================================================

--- GROUP 1: WATERMARK VISIBILITY (training data vs originals) ---
  n_compared : 559
  avg PSNR   : 25.671 dB  (higher = less visible, >40 is imperceptible)
  avg SSIM   : 0.9042     (higher = more similar to original, 1.0 = identical)

--- GROUP 2: MODEL DETERIORATION (generated images vs clean baseline) ---
  all images (n=110)
    clip content  : 28.409
    clip style    : 26.036
    clip combined : 27.223
    brightness    : 111.358
    sharpness     : 1083.001
  seeded vs clean baseline (n=10)
    avg SSIM      : 0.6475  (higher = more similar to clean)
    avg CLIP drift: 0.1288  (lower = more similar to clean)


the last one

shift                |     23.357     0.8947 |     0.1538        0.4201

============================================================
EXPERIMENT: shift_a5.0_p1.0__e3_r64_la32_lr5e-05_ga4_attn+ff+conv_seed42
============================================================

--- GROUP 1: WATERMARK VISIBILITY (training data vs originals) ---
  n_compared : 559
  avg PSNR   : 23.357 dB  (higher = less visible, >40 is imperceptible)
  avg SSIM   : 0.8947     (higher = more similar to original, 1.0 = identical)

--- GROUP 2: MODEL DETERIORATION (generated images vs clean baseline) ---
  all images (n=110)
    clip content  : 28.504
    clip style    : 26.172
    clip combined : 27.338
    brightness    : 107.763
    sharpness     : 2204.446
  seeded vs clean baseline (n=10)
    avg SSIM      : 0.4201  (higher = more similar to clean)
    avg CLIP drift: 0.1538  (lower = more similar to clean)





























============================================================
EXPERIMENT: adversarial_a5.0_p0.1__e3_r64_la32_lr5e-05_ga4_attn+ff+conv_seed42
============================================================

--- GROUP 1: WATERMARK VISIBILITY (training data vs originals) ---
  using manifest: comparing 55 poisoned images only
  n_compared : 55
  avg PSNR   : 29.652 dB  (higher = less visible, >40 is imperceptible)
  avg SSIM   : 0.8922     (higher = more similar to original, 1.0 = identical)

--- GROUP 2: MODEL DETERIORATION (generated images vs clean baseline) ---
  all images (n=110)
    clip content  : 27.536
    clip style    : 26.508
    clip combined : 27.022
    brightness    : 108.504
    sharpness     : 1257.139
  seeded vs clean baseline (n=10)
    avg SSIM      : 0.5576  (higher = more similar to clean)
    avg CLIP drift: 0.1514  (lower = more similar to clean)

============================================================
EXPERIMENT: adversarial_a5.0_p0.25__e3_r64_la32_lr5e-05_ga4_attn+ff+conv_seed42
============================================================

--- GROUP 1: WATERMARK VISIBILITY (training data vs originals) ---
  using manifest: comparing 139 poisoned images only
  n_compared : 139
  avg PSNR   : 29.51 dB  (higher = less visible, >40 is imperceptible)
  avg SSIM   : 0.8819     (higher = more similar to original, 1.0 = identical)

--- GROUP 2: MODEL DETERIORATION (generated images vs clean baseline) ---
  all images (n=110)
    clip content  : 27.367
    clip style    : 26.73
    clip combined : 27.049
    brightness    : 108.822
    sharpness     : 1324.763
  seeded vs clean baseline (n=10)
    avg SSIM      : 0.5137  (higher = more similar to clean)
    avg CLIP drift: 0.1638  (lower = more similar to clean)

============================================================
EXPERIMENT: adversarial_a5.0_p0.5__e3_r64_la32_lr5e-05_ga4_attn+ff+conv_seed42
============================================================

--- GROUP 1: WATERMARK VISIBILITY (training data vs originals) ---
  using manifest: comparing 279 poisoned images only
  n_compared : 279
  avg PSNR   : 29.429 dB  (higher = less visible, >40 is imperceptible)
  avg SSIM   : 0.8792     (higher = more similar to original, 1.0 = identical)

--- GROUP 2: MODEL DETERIORATION (generated images vs clean baseline) ---
  all images (n=110)
    clip content  : 25.814
    clip style    : 26.484
    clip combined : 26.149
    brightness    : 107.449
    sharpness     : 1857.572
  seeded vs clean baseline (n=10)
    avg SSIM      : 0.3966  (higher = more similar to clean)
    avg CLIP drift: 0.3547  (lower = more similar to clean)

============================================================
EXPERIMENT: adversarial_a5.0_p0.75__e3_r64_la32_lr5e-05_ga4_attn+ff+conv_seed42
============================================================

--- GROUP 1: WATERMARK VISIBILITY (training data vs originals) ---
  using manifest: comparing 419 poisoned images only
  n_compared : 419
  avg PSNR   : 29.432 dB  (higher = less visible, >40 is imperceptible)
  avg SSIM   : 0.8786     (higher = more similar to original, 1.0 = identical)

--- GROUP 2: MODEL DETERIORATION (generated images vs clean baseline) ---
  all images (n=110)
    clip content  : 27.06
    clip style    : 26.885
    clip combined : 26.973
    brightness    : 108.959
    sharpness     : 1125.289
  seeded vs clean baseline (n=10)
    avg SSIM      : 0.4387  (higher = more similar to clean)
    avg CLIP drift: 0.2628  (lower = more similar to clean)

============================================================
EXPERIMENT: adversarial_a5.0_p1.0__e3_r64_la32_lr5e-05_ga4_attn+ff+conv_seed42
============================================================

--- GROUP 1: WATERMARK VISIBILITY (training data vs originals) ---
  n_compared : 559
  avg PSNR   : 29.427 dB  (higher = less visible, >40 is imperceptible)
  avg SSIM   : 0.8776     (higher = more similar to original, 1.0 = identical)

--- GROUP 2: MODEL DETERIORATION (generated images vs clean baseline) ---
  all images (n=110)
    clip content  : 28.175
    clip style    : 26.819
    clip combined : 27.497
    brightness    : 110.016
    sharpness     : 865.034
  seeded vs clean baseline (n=10)
    avg SSIM      : 0.4599  (higher = more similar to clean)
    avg CLIP drift: 0.2235  (lower = more similar to clean)

============================================================
EXPERIMENT: DWT-DCT_a5.0_p1.0__e3_r64_la32_lr5e-05_ga4_attn+ff+conv_seed42
============================================================

--- GROUP 1: WATERMARK VISIBILITY (training data vs originals) ---
  n_compared : 559
  avg PSNR   : 32.008 dB  (higher = less visible, >40 is imperceptible)
  avg SSIM   : 0.9257     (higher = more similar to original, 1.0 = identical)

--- GROUP 2: MODEL DETERIORATION (generated images vs clean baseline) ---
  all images (n=110)
    clip content  : 27.77
    clip style    : 25.697
    clip combined : 26.734
    brightness    : 108.673
    sharpness     : 1340.155
  seeded vs clean baseline (n=10)
    avg SSIM      : 0.7009  (higher = more similar to clean)
    avg CLIP drift: 0.0873  (lower = more similar to clean)

============================================================
EXPERIMENT: empty_a5.0_p1.0__e3_r64_la32_lr5e-05_ga4_attn+ff+conv_seed42
============================================================

--- GROUP 1: WATERMARK VISIBILITY (training data vs originals) ---
  n_compared : 559
  avg PSNR   : 32.04 dB  (higher = less visible, >40 is imperceptible)
  avg SSIM   : 0.926     (higher = more similar to original, 1.0 = identical)

--- GROUP 2: MODEL DETERIORATION (generated images vs clean baseline) ---
  all images (n=110)
    clip content  : 28.027
    clip style    : 25.661
    clip combined : 26.844
    brightness    : 108.763
    sharpness     : 1298.213
  seeded vs clean baseline (n=10)
    avg SSIM      : 1.0  (higher = more similar to clean)
    avg CLIP drift: 0.0  (lower = more similar to clean)

============================================================
EXPERIMENT: square_a5.0_p1.0__e3_r64_la32_lr5e-05_ga4_attn+ff+conv_seed42
============================================================

--- GROUP 1: WATERMARK VISIBILITY (training data vs originals) ---
  n_compared : 559
  avg PSNR   : 25.591 dB  (higher = less visible, >40 is imperceptible)
  avg SSIM   : 0.9153     (higher = more similar to original, 1.0 = identical)

--- GROUP 2: MODEL DETERIORATION (generated images vs clean baseline) ---
  all images (n=110)
    clip content  : 27.626
    clip style    : 25.615
    clip combined : 26.621
    brightness    : 108.206
    sharpness     : 1200.561
  seeded vs clean baseline (n=10)
    avg SSIM      : 0.5765  (higher = more similar to clean)
    avg CLIP drift: 0.1866  (lower = more similar to clean)