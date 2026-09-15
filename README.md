# Task-Text Coordinates for Normalized Few-Shot Vision-Language Adaptation

Parameter budgets and implementation details accompanying the paper. P0 fixes the complete centered class-text output space and learns a visual coordinate map within it. The paper presents the method, its theoretical properties and the principal accuracy comparisons.

## Parameter budgets

The tables count trainable scalars in the evaluated adaptation modules. Frozen encoders, text embeddings and subspace bases are excluded. Counts apply to both 4-shot and 16-shot settings. ProKeR fits kernel coefficients by a closed-form solve, whose coefficient count is given separately below.

P0 trains **35,328 / 6,912 / 27,648 / 304,128** scalars for DTD / EuroSAT / Pets / SUN397 on either encoder. Relative to the full ProLIP endpoint, its trainable-parameter ratio is **r/e**. For example, the rank-9 SigLIP 2 EuroSAT configuration reduces 589,824 parameters to 6,912, a **98.8% reduction**.

### CLIP ViT-B/16

| Method | Configuration | DTD | EuroSAT | Pets | SUN397 |
| --- | --- | ---: | ---: | ---: | ---: |
| LP | Base | 24,111 | 5,130 | 18,981 | 203,661 |
| LP | LP++ | 24,158 | 5,140 | 19,018 | 204,058 |
| ProKeR | Kernel solve | -- (see below) | -- (see below) | -- (see below) | -- (see below) |
| ProLIP | Full | 393,216 | 393,216 | 393,216 | 393,216 |
| LoRA | Rank | 58,880 | 11,520 | 46,080 | 506,880 |
| LoRA | Param. | 35,840 | 6,400 | 28,160 | 304,640 |
| SVD-E | Singular values | 512 | 512 | 512 | 512 |
| Comp-E | k = 2 | 2,496 | 2,496 | 2,496 | 2,496 |
| Comp-E | Rank | 57,408 | 11,232 | 44,928 | 494,208 |
| Comp-E | Param. | 34,944 | 7,488 | 27,456 | 304,512 |
| **P0** | Task-text | 35,328 | 6,912 | 27,648 | 304,128 |

### SigLIP 2 Base/16

| Method | Configuration | DTD | EuroSAT | Pets | SUN397 |
| --- | --- | ---: | ---: | ---: | ---: |
| LP | Base | 36,143 | 7,690 | 28,453 | 305,293 |
| LP | LP++ | 36,190 | 7,700 | 28,490 | 305,690 |
| ProKeR | Kernel solve | -- (see below) | -- (see below) | -- (see below) | -- (see below) |
| ProLIP | Full | 589,824 | 589,824 | 589,824 | 589,824 |
| LoRA | Rank | 70,656 | 13,824 | 55,296 | 608,256 |
| LoRA | Param. | 35,328 | 6,144 | 27,648 | 304,128 |
| SVD-E | Singular values | 768 | 768 | 768 | 768 |
| Comp-E | k = 2 | 7,616 | 7,616 | 7,616 | 7,616 |
| Comp-E | Rank | 175,168 | 34,272 | 137,088 | 1,507,968 |
| Comp-E | Param. | 34,272 | 7,616 | 26,656 | 304,640 |
| **P0** | Task-text | 35,328 | 6,912 | 27,648 | 304,128 |

### Counting formulas and locations

For P0, ProLIP and LoRA, d = 768 and e = 512/768 for CLIP/SigLIP 2. Their adaptation locations are the CLIP visual projector and the SigLIP 2 output residual. The spectral methods use the CLIP projector or the SigLIP 2 pooling-head FC2, with (d_c, e_c) = (768, 512)/(3072, 768).

| Method | Scalar count | Learned quantities |
| --- | --- | --- |
| P0 | dr | Coordinate map A |
| ProLIP | de | Full endpoint increment |
| LoRA | k(d + e) | Both low-rank factors |
| SVD-E | min(d_c, e_c) | Singular-value increments |
| Comp-E | k(d_c + e_c - 32) | Both factors within fixed complementary singular spaces |
| LP Base | C(e + 1) | Classifier weights and bias |
| LP++ | C(e + 2) | Classifier weights, bias and class text coefficients |

C is the class count. The class counts are 47/10/37/397, and the corresponding text ranks r are 46/9/36/396 on both encoders. SVD-E varies singular values along fixed singular directions; its small count describes that specific parameterization. All counts refer to optimized scalars rather than the number of independent matrix entries reachable by a factorization.

**ProKeR coefficient count.** With ten views per support image and K images per class, ProKeR solves for 10KC² coefficients. The support kernel contains (10KC)² entries. These are fitted kernel coefficients and kernel-matrix entries, respectively. The SUN397 16-shot case is unreported in the main accuracy table because it exceeds the evaluated capacity limit.

## Matched ranks and parameter differences

Rank matches P0's text rank r. Param. chooses the integer rank minimizing the absolute difference from P0's dr trainable scalars; ties favor the smaller rank. Every budget is measured against the P0 configuration in the main table. Signed differences below are comparator count minus P0 count.

| Encoder | Dataset | P0 / Rank r | LoRA Param. k | LoRA difference | Comp-E Param. k | Comp-E difference |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CLIP | DTD | 46 | 28 | +512 | 28 | -384 |
| CLIP | EuroSAT | 9 | 5 | -512 | 6 | +576 |
| CLIP | Pets | 36 | 22 | +512 | 22 | -192 |
| CLIP | SUN397 | 396 | 238 | +512 | 244 | +384 |
| SigLIP 2 | DTD | 46 | 23 | +0 | 9 | -1,056 |
| SigLIP 2 | EuroSAT | 9 | 4 | -768 | 2 | +704 |
| SigLIP 2 | Pets | 36 | 18 | +0 | 7 | -992 |
| SigLIP 2 | SUN397 | 396 | 198 | +0 | 80 | +512 |

Comp-E also reports k = 2, with 2,496/7,616 trainable scalars on CLIP/SigLIP 2. On SigLIP 2 EuroSAT, k = 2 and Param. specify the same configuration. Comp-E excludes the leading sixteen left and right singular directions in every configuration.

## Implementation details

[Implementation and evaluation](implementation.md) gives the checkpoints, exact class-text prompts, data splits, image preprocessing, P0 basis construction, initialization, optimizer and search settings, and the baseline protocols. The study uses support seeds 1/2/3 throughout. Main Table 2 uses the shared nine-candidate validation selection.

[Machine-readable parameter counts](data/parameter_budgets.json) accompany the tables above.
