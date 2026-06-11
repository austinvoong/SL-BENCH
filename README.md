# SL-BENCH
 
A privacy attack and defense evaluation framework for split learning. Implements multiple SL architectures, reconstruction attacks, and defenses in a unified, reproducible pipeline. Includes a web dashboard for launching and visualizing experiments, as well as detailed terminal output.
 
 **Core finding summary:** Split learning is not inherently privacy-preserving. Smashed data contains sufficient information for a semi-honest server to reconstruct private training images. Existing practical defenses are insufficient against modern attacks.
 
## What's Implemented
 
**Architectures \models**
- Vanilla Split Learning
- U-Shaped Split Learning
- SplitFed (federated + split)
**Attacks \attacks**
- Inverse Network — baseline reconstruction decoder
- FSHA — malicious server, hijacks the training loop
- PCAT — semi-honest, pseudo-client using auxiliary data
- FORA — semi-honest, feature-space alignment via MK-MMD *(primary threat)*
**Defenses \defenses**
- NoPeekNN — distance correlation regularization
- Differential Privacy — Gaussian/Laplace activation perturbation
- AFO (Adversarial Feature Obfuscation) — novel defense, adversarially trained obfuscation module *(primary contribution)*
**Metrics:** SSIM, PSNR, distance correlation (dCor), test accuracy
**Web Visualization:** Contains a basic frontend (React) and backend (Flask & MongoDB)