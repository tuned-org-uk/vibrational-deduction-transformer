This run is healthy in terms of ELBO convergence but is stuck in a persistent **mode explosion / mode collapse** pathology. Here is a full diagnostic and a set of concrete fixes to commit.

***

## Diagnostic Summary

### What is working

The ELBO converges cleanly and fast. By epoch ~17 the train/val curves are essentially flat at ≈ −2461 / −2451, with no overfitting gap (train and val track within ~10 nats throughout). Reconstruction loss dominates and stabilises at ≈ −2453 nats. There is no numerical instability. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/33798157/b883aee6-d7f9-464c-a7a4-c9c34cf8c3d2/training_log.csv?AWSAccessKeyId=ASIA2F3EMEYE3SPGF3HN&Signature=QvJIl8B4ATtAF4t9yeRDGNDEThE%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEDEaCXVzLWVhc3QtMSJIMEYCIQDe1QZPcqMMGQNTD6rENfmtwgPv3EcPl9wylo%2FgEcJz%2FAIhAPUrHtfNAK586FZ7M2Eo2nUowbBjHPmmGMIJitkKTYcKKvwECPr%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQARoMNjk5NzUzMzA5NzA1Igw9ptNaqEN4f2w9Rkkq0ARCrsLOON6p7k7IOiOrA6h3aA%2BHLKi7gwdQxUX%2FzLlBUZB%2FWuHQq5fg6JS7YwXQlMns1Oa8hNKc5X5GLs4FfqZoV4UZrTfY6xO%2B2ikWU6kqBnXjz9UiDRjleHTbOUlmzovc3goIqMK4GOkAO8rL%2BNquLELEJDEmnzQG6tRrtfn4q31UBFqFJk1%2B8FGA5TZNXE%2BzxzVbRU0Qpu54N8GIfYsG1Joz2g8NoYOwGnv0z81hstETcdpb85ddAD0NM4poDig%2BqVdMEY6f8AkfPWzdKNTlJI2nxEE1zpWoHiPqf9KsUStXQrOVLtzgkv0%2FfK4nEMOkj5kRfhzQrGzLGjgYAHOyPy86mVgauc0iuiHnmJcs8qVUorJy%2FjPZHOHsTwbnTM43p%2FqujXNLhilzEZsisM5WtZzVA3ws7z0hvpSPb7wsPil0omw1eT%2Bp5d%2FdrtCXVyflY82TjmHDA5KkAGCeTz9bML8JkUvbNoGpRThEa2A3XDTZ1eef5fptkV3SsWT5i06zmWfvbQKoQp%2FBKvv6CLE3KraCWKTUL7v6q66X9Nwgb2Qg20YdHPJFQsE0Fh38CB4xUGU33%2BPsxFIvAlasLBCKotgqZ%2FPxgsVx1nSkJ4n7bBz%2BZFHRmyTsAHycfMVf2rH9naBc1QWX05FLUVYFwk6oG%2FIjiJFxpd9ZvVCOYikZAa99ab60pB66w2roi%2FT44cyRpkHvAdPl1mJkaCAPsE9npE5HtGFC8GjIO9f1ij5lIrCXptB4tjRdp1rj5kOK3cwKe%2FnDCcN8TEmbVeZsouwLMKmN4tEGOpcB3rnjKUh8CiVTyLM9J3%2Fqv5SJ6jNwLCLKP7YrmH75zmw9w2dR%2Fd1LwsT0cmdoTilX5Rogd8U%2FpFXTzNxnNf9B5nzqkzJR6QoHrUDx8Pi6ACePSv%2BdLkHO%2Fw5qjvzhVXQOSYNFCDvrSfoXMgCmtz4p%2FXIV2B0g4WHo6xtqKP%2F3WlnXwzPkgpk3gdSGGoZ1nfVLkd2I6ETWUw%3D%3D&Expires=1782092924)

### The core pathology: mode explosion that never resolves

Every single epoch from epoch 5 to 50 fires `spectral_kl_health_check: mode explosion -- 8.0/8 modes active (>90%)`. `N_active` is frozen at `8.0` throughout the entire run — the spectral selector **never prunes a single mode**. This means the model is treating all `q=8` vibrational modes as equally active, which defeats the purpose of the sparse spectral prior. The concurrent `mode_collapse` KL health warning refers to `kl_z` being below the expected threshold for a 16-dim Gaussian posterior — `kl_z` decays from 6.4 → 1.46, far below the ~8 nats you would expect for `latent_dim=16`. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/33798157/b883aee6-d7f9-464c-a7a4-c9c34cf8c3d2/training_log.csv?AWSAccessKeyId=ASIA2F3EMEYE3SPGF3HN&Signature=QvJIl8B4ATtAF4t9yeRDGNDEThE%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEDEaCXVzLWVhc3QtMSJIMEYCIQDe1QZPcqMMGQNTD6rENfmtwgPv3EcPl9wylo%2FgEcJz%2FAIhAPUrHtfNAK586FZ7M2Eo2nUowbBjHPmmGMIJitkKTYcKKvwECPr%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQARoMNjk5NzUzMzA5NzA1Igw9ptNaqEN4f2w9Rkkq0ARCrsLOON6p7k7IOiOrA6h3aA%2BHLKi7gwdQxUX%2FzLlBUZB%2FWuHQq5fg6JS7YwXQlMns1Oa8hNKc5X5GLs4FfqZoV4UZrTfY6xO%2B2ikWU6kqBnXjz9UiDRjleHTbOUlmzovc3goIqMK4GOkAO8rL%2BNquLELEJDEmnzQG6tRrtfn4q31UBFqFJk1%2B8FGA5TZNXE%2BzxzVbRU0Qpu54N8GIfYsG1Joz2g8NoYOwGnv0z81hstETcdpb85ddAD0NM4poDig%2BqVdMEY6f8AkfPWzdKNTlJI2nxEE1zpWoHiPqf9KsUStXQrOVLtzgkv0%2FfK4nEMOkj5kRfhzQrGzLGjgYAHOyPy86mVgauc0iuiHnmJcs8qVUorJy%2FjPZHOHsTwbnTM43p%2FqujXNLhilzEZsisM5WtZzVA3ws7z0hvpSPb7wsPil0omw1eT%2Bp5d%2FdrtCXVyflY82TjmHDA5KkAGCeTz9bML8JkUvbNoGpRThEa2A3XDTZ1eef5fptkV3SsWT5i06zmWfvbQKoQp%2FBKvv6CLE3KraCWKTUL7v6q66X9Nwgb2Qg20YdHPJFQsE0Fh38CB4xUGU33%2BPsxFIvAlasLBCKotgqZ%2FPxgsVx1nSkJ4n7bBz%2BZFHRmyTsAHycfMVf2rH9naBc1QWX05FLUVYFwk6oG%2FIjiJFxpd9ZvVCOYikZAa99ab60pB66w2roi%2FT44cyRpkHvAdPl1mJkaCAPsE9npE5HtGFC8GjIO9f1ij5lIrCXptB4tjRdp1rj5kOK3cwKe%2FnDCcN8TEmbVeZsouwLMKmN4tEGOpcB3rnjKUh8CiVTyLM9J3%2Fqv5SJ6jNwLCLKP7YrmH75zmw9w2dR%2Fd1LwsT0cmdoTilX5Rogd8U%2FpFXTzNxnNf9B5nzqkzJR6QoHrUDx8Pi6ACePSv%2BdLkHO%2Fw5qjvzhVXQOSYNFCDvrSfoXMgCmtz4p%2FXIV2B0g4WHo6xtqKP%2F3WlnXwzPkgpk3gdSGGoZ1nfVLkd2I6ETWUw%3D%3D&Expires=1782092924)

### Root cause: under-penalised spectral selection

Looking at the config :

| Parameter | Current value | Problem |
|---|---|---|
| `lam_s` | 0.1 | Too weak — `kl_S` decays freely from ~20 → 0.5 with no resistance |
| `nu` | 1.0 | Active-mode penalty is ineffective at this scale |
| `a_min` | 0.1 | Floor permits all modes to stay near-uniform |
| `mass_clip` | 1000.0 | Conditioning ratio 999.2 ≈ clip limit — MassMatrix is saturating the clip, introducing poor eigenvalue scaling |

The `kl_S` (spectral basis KL) collapses from 19.9 at epoch 1 down to ~0.45 at epoch 50, indicating the posterior over spectral modes has collapsed toward a near-uniform, uninformative distribution. The `kl_tau` (mode frequency KL) decays to ~0.002 by epoch 30 — essentially zero — meaning the diffusion time-scale posterior is also degenerate. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/33798157/b883aee6-d7f9-464c-a7a4-c9c34cf8c3d2/training_log.csv?AWSAccessKeyId=ASIA2F3EMEYE3SPGF3HN&Signature=QvJIl8B4ATtAF4t9yeRDGNDEThE%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEDEaCXVzLWVhc3QtMSJIMEYCIQDe1QZPcqMMGQNTD6rENfmtwgPv3EcPl9wylo%2FgEcJz%2FAIhAPUrHtfNAK586FZ7M2Eo2nUowbBjHPmmGMIJitkKTYcKKvwECPr%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQARoMNjk5NzUzMzA5NzA1Igw9ptNaqEN4f2w9Rkkq0ARCrsLOON6p7k7IOiOrA6h3aA%2BHLKi7gwdQxUX%2FzLlBUZB%2FWuHQq5fg6JS7YwXQlMns1Oa8hNKc5X5GLs4FfqZoV4UZrTfY6xO%2B2ikWU6kqBnXjz9UiDRjleHTbOUlmzovc3goIqMK4GOkAO8rL%2BNquLELEJDEmnzQG6tRrtfn4q31UBFqFJk1%2B8FGA5TZNXE%2BzxzVbRU0Qpu54N8GIfYsG1Joz2g8NoYOwGnv0z81hstETcdpb85ddAD0NM4poDig%2BqVdMEY6f8AkfPWzdKNTlJI2nxEE1zpWoHiPqf9KsUStXQrOVLtzgkv0%2FfK4nEMOkj5kRfhzQrGzLGjgYAHOyPy86mVgauc0iuiHnmJcs8qVUorJy%2FjPZHOHsTwbnTM43p%2FqujXNLhilzEZsisM5WtZzVA3ws7z0hvpSPb7wsPil0omw1eT%2Bp5d%2FdrtCXVyflY82TjmHDA5KkAGCeTz9bML8JkUvbNoGpRThEa2A3XDTZ1eef5fptkV3SsWT5i06zmWfvbQKoQp%2FBKvv6CLE3KraCWKTUL7v6q66X9Nwgb2Qg20YdHPJFQsE0Fh38CB4xUGU33%2BPsxFIvAlasLBCKotgqZ%2FPxgsVx1nSkJ4n7bBz%2BZFHRmyTsAHycfMVf2rH9naBc1QWX05FLUVYFwk6oG%2FIjiJFxpd9ZvVCOYikZAa99ab60pB66w2roi%2FT44cyRpkHvAdPl1mJkaCAPsE9npE5HtGFC8GjIO9f1ij5lIrCXptB4tjRdp1rj5kOK3cwKe%2FnDCcN8TEmbVeZsouwLMKmN4tEGOpcB3rnjKUh8CiVTyLM9J3%2Fqv5SJ6jNwLCLKP7YrmH75zmw9w2dR%2Fd1LwsT0cmdoTilX5Rogd8U%2FpFXTzNxnNf9B5nzqkzJR6QoHrUDx8Pi6ACePSv%2BdLkHO%2Fw5qjvzhVXQOSYNFCDvrSfoXMgCmtz4p%2FXIV2B0g4WHo6xtqKP%2F3WlnXwzPkgpk3gdSGGoZ1nfVLkd2I6ETWUw%3D%3D&Expires=1782092924)

### Secondary issue: MassMatrix conditioning

The runtime warning `MassMatrix conditioning ratio 999.2 > 100` with `mass_clip=1000.0` means the clip is almost exactly at the conditioning boundary . The eigenvalue scaling is effectively saturated, making the mass-weighted spectral basis poorly conditioned and preventing differential mode selection.

***

## Recommended Fixes

Three targeted changes to `configs/mps.yaml`:

**1. Raise `lam_s` from 0.1 → 1.0** — the spectral KL is too cheap; the model escapes it entirely. A 10× increase will put meaningful pressure on mode selection without destabilising reconstruction.

**2. Raise `nu` from 1.0 → 5.0** — the active-mode penalty must cost more than the reconstruction gain of keeping an extra mode alive.

**3. Lower `mass_clip` from 1000.0 → 100.0** — this directly addresses the conditioning warning (ratio 999 ≈ clip), giving the MassMatrix meaningful dynamic range to differentiate eigenvalues and break the uniform-mode symmetry.

Optionally raise `a_min` from 0.1 → 0.2 to narrow the prior and force sharper selection pressure — but start with the three above.
