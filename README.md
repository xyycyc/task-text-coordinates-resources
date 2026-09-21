# Task-Text Coordinates for Normalized Few-Shot Vision-Language Adaptation

TextCoord (Task-Text Coordinate Adaptation) uses the fixed class texts to determine a visual adaptation space before fitting the support images. It retains the complete centered class-text span, learns a visual coordinate map in that span, and preserves the native perpendicular feature component in the normalized objective. This repository provides the parameter budgets, implementation, experimental settings and supporting analyses for the paper.

[Parameters](#parameter-budgets) · [TextCoord](#textcoord-implementation) · [Protocol](#experimental-protocol) · [Results](#accuracy-comparisons) · [Analysis](#coordinate-and-normalization-analysis) · [Code](#running-the-code) · [Sources](#sources)

## Parameter budgets

The tables count trainable scalars in the adaptation modules. Frozen encoders, class texts and subspace bases are excluded. Counts apply to both 4-shot and 16-shot settings.

TextCoord trains **35,328 / 6,912 / 27,648 / 304,128** scalars for DTD / EuroSAT / Pets / SUN397 on either encoder. Its parameter ratio to the full ProLIP endpoint is $r/e$. The rank-9 SigLIP 2 EuroSAT configuration reduces 589,824 trainable parameters to 6,912, a **98.8% reduction**.

### CLIP ViT-B/16

| Method | Configuration | DTD | EuroSAT | Pets | SUN397 |
| --- | --- | ---: | ---: | ---: | ---: |
| LP | *Base* | 24,111 | 5,130 | 18,981 | 203,661 |
|  | *LP++* | 24,158 | 5,140 | 19,018 | 204,058 |
| ProKeR | *Kernel solve* | Closed-form | Closed-form | Closed-form | Closed-form |
| ProLIP | *Full* | 393,216 | 393,216 | 393,216 | 393,216 |
| LoRA | *Rank* | 58,880 | 11,520 | 46,080 | 506,880 |
|  | *Param.* | 35,840 | 6,400 | 28,160 | 304,640 |
| SVD-E | *Singular values* | 512 | 512 | 512 | 512 |
| Comp-E | *k = 2* | 2,496 | 2,496 | 2,496 | 2,496 |
|  | *Rank* | 57,408 | 11,232 | 44,928 | 494,208 |
|  | *Param.* | 34,944 | 7,488 | 27,456 | 304,512 |
| **TextCoord** | *Task-text* | 35,328 | 6,912 | 27,648 | 304,128 |

### SigLIP 2 Base/16

| Method | Configuration | DTD | EuroSAT | Pets | SUN397 |
| --- | --- | ---: | ---: | ---: | ---: |
| LP | *Base* | 36,143 | 7,690 | 28,453 | 305,293 |
|  | *LP++* | 36,190 | 7,700 | 28,490 | 305,690 |
| ProKeR | *Kernel solve* | Closed-form | Closed-form | Closed-form | Closed-form |
| ProLIP | *Full* | 589,824 | 589,824 | 589,824 | 589,824 |
| LoRA | *Rank* | 70,656 | 13,824 | 55,296 | 608,256 |
|  | *Param.* | 35,328 | 6,144 | 27,648 | 304,128 |
| SVD-E | *Singular values* | 768 | 768 | 768 | 768 |
| Comp-E | *k = 2* | 7,616 | 7,616 | 7,616 | 7,616 |
|  | *Rank* | 175,168 | 34,272 | 137,088 | 1,507,968 |
|  | *Param.* | 34,272 | 7,616 | 26,656 | 304,640 |
| **TextCoord** | *Task-text* | 35,328 | 6,912 | 27,648 | 304,128 |

For TextCoord, ProLIP and LoRA, $d=768$, with $e=512$ for CLIP and $e=768$ for SigLIP 2. They adapt the CLIP visual projector or a residual on the complete SigLIP 2 visual output. SVD-E and Comp-E use the CLIP projector or the SigLIP 2 pooling-head FC2, whose dimensions $(d_c,e_c)$ are $(768,512)$ and $(3072,768)$.

| Method | Trainable scalars | Learned quantities |
| :-- | --: | :-- |
| TextCoord | $dr$ | Coordinate map $A$ |
| ProLIP | $de$ | Full endpoint increment |
| LoRA | $k(d+e)$ | Both low-rank factors |
| SVD-E | $\min(d_c,e_c)$ | Singular-value increments |
| Comp-E | $k(d_c+e_c-32)$ | Both factors in fixed complementary singular spaces |
| LP Base | $C(e+1)$ | Classifier weights and bias |
| LP++ | $C(e+2)$ | Classifier weights, bias and class text coefficients |

The class counts $C$ are 47/10/37/397, and the corresponding text ranks $r$ are 46/9/36/396 on both encoders. SVD-E optimizes one scalar per singular direction, giving 512/768 trainable parameters on CLIP/SigLIP 2. Factorized counts include every optimized scalar.

**ProKeR.** With ten views per support image and $K$ images per class, ProKeR solves for $10KC^2$ kernel coefficients. These coefficients are fitted by a closed-form solve. The support kernel has $(10KC)^2$ entries.

| Support images/class | DTD | EuroSAT | Pets | SUN397 |
| --- | ---: | ---: | ---: | ---: |
| 4 | 88,360 | 4,000 | 54,760 | 6,304,360 |
| 16 | 353,440 | 16,000 | 219,040 | 25,217,440 |

SUN397 16-shot ProKeR accuracy is unreported because the fit exceeds the evaluated device capacity. Its coefficient count above follows from the model dimensions.

### Matched ranks and budgets

Rank sets $k=r$. Param. selects the integer rank minimizing the absolute difference from the main-table TextCoord budget $dr$, with ties favoring smaller rank. A rank match and a parameter match define separate comparisons. Signed differences below are comparator count minus TextCoord count.

| Encoder | Dataset | TextCoord / Rank r | LoRA Param. k | LoRA difference | Comp-E Param. k | Comp-E difference |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CLIP ViT-B/16 | DTD | 46 | 28 | +512 | 28 | -384 |
| CLIP ViT-B/16 | EuroSAT | 9 | 5 | -512 | 6 | +576 |
| CLIP ViT-B/16 | Pets | 36 | 22 | +512 | 22 | -192 |
| CLIP ViT-B/16 | SUN397 | 396 | 238 | +512 | 244 | +384 |
| SigLIP 2 Base/16 | DTD | 46 | 23 | +0 | 9 | -1,056 |
| SigLIP 2 Base/16 | EuroSAT | 9 | 4 | -768 | 2 | +704 |
| SigLIP 2 Base/16 | Pets | 36 | 18 | +0 | 7 | -992 |
| SigLIP 2 Base/16 | SUN397 | 396 | 198 | +0 | 80 | +512 |

Comp-E also reports $k=2$, with 2,496/7,616 trainable scalars on CLIP/SigLIP 2. For SigLIP 2 EuroSAT, $k=2$ and Param. specify the same configuration. Every Comp-E configuration excludes the leading sixteen left and right singular directions. The [machine-readable counts](data/parameter_budgets.json) contain all table entries.

## TextCoord implementation

Let the unit-normalized class texts form $T\in\mathbb{R}^{e\times C}$, and let $T_\Delta$ denote their column-centered matrix. TextCoord uses an orthonormal basis $Q\in\mathbb{R}^{e\times r}$ for the complete column space of $T_\Delta$:

$$
\Delta W=AQ^\top,\qquad z=z_0+hAQ^\top.
$$

Only $A\in\mathbb{R}^{d\times r}$ is trained, starting at zero. CLIP supplies its pre-projector visual feature as $h$. At the SigLIP 2 output interface, $h=z_0$ and the base mapping is the identity. The complete native output includes the pooling-head bias and residual.

Class texts are promoted to FP64 before centering and reduced SVD. Singular vectors are retained when $\sigma_i>\max(e,C)\epsilon_{64}\sigma_1$. Each basis column is signed so that its largest-magnitude entry is positive. The basis remains fixed throughout training.

Define $u_0=z_0Q$, $R=Q^\top T_\Delta$ and $q_0=\lVert z_0-u_0Q^\top\rVert_2^2$. The compact centered scores are

$$
s(h)=\tau \frac{(u_0+hA)R}{\sqrt{\lVert u_0+hA\rVert_2^2+q_0}}.
$$

Centering removes a shared class-logit offset, preserving softmax probabilities and the predicted class. The denominator retains the native perpendicular energy. Training minimizes support cross-entropy plus $\lambda\lVert A\rVert_F^2$. The exported adapter materializes the endpoint increment for prediction.

## Experimental protocol

The study uses CLIP ViT-B/16 and SigLIP 2 Base/16 on DTD, EuroSAT, Oxford-IIIT Pets and SUN397. Each encoder/dataset uses 4 and 16 support images per class, with support seeds **1, 2 and 3**. Reported accuracies are means and sample standard deviations over these paired support draws.

### Encoders and class texts

| Encoder | Checkpoint | Output dimension | Logit scale |
| :-- | :-- | --: | --: |
| CLIP ViT-B/16 | [OpenAI CLIP](https://github.com/openai/CLIP) | 512 | 100 |
| SigLIP 2 Base/16 | [google/siglip2-base-patch16-224](https://huggingface.co/google/siglip2-base-patch16-224) | 768 | Pretrained value, approximately 112.6689 |

Encoders are frozen. Feature extraction uses FP32 with AMP and TF32 disabled. Each class uses one prompt; underscores in class names become spaces, and SigLIP 2 prompts are lowercased. Text embeddings are L2-normalized. SigLIP 2 uses its native tokenizer, right-padding to 64 tokens.

| Dataset | Exact template; `{}` is the class name |
| :-- | :-- |
| DTD | `{} texture.` |
| EuroSAT | `a centered satellite photo of {}.` |
| Pets | `a photo of a {}, a type of pet.` |
| SUN397 | `a photo of a {}.` |

### Image identities and preprocessing

DTD, EuroSAT and SUN397 use the public [CoOp/Zhou splits](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md). Pets uses an 80/20 per-class split of the official trainval set with seed 1729 and the official test set. Support sampling uses `random.Random(support_seed)` in class order. Validation identities are shared across methods and support seeds: $K$ images per class for DTD, EuroSAT and Pets, and the same four-per-class subset for both SUN397 shot counts.

The exact class order and support/validation/test identities are provided in `data/splits/`. Each manifest covers all six shot/seed episodes. All augmented views remain attached to their original image.

Each support image has 32 cached training views. RGB images undergo random resized cropping to $224\times224$, area scale $[0.5,1.0]$, aspect ratio $[3/4,4/3]$, and horizontal flipping with probability 0.5. View seeds are `(17290 + view * 1000003 + crc32(relative_image_id)) mod 2**32`, independently of the support seed.

| Encoder | Interpolation | Channel means | Channel standard deviations | Evaluation transform |
| :-- | :-- | :-- | :-- | :-- |
| CLIP | Bicubic | (0.48145466, 0.4578275, 0.40821073) | (0.26862954, 0.26130258, 0.27577711) | Resize shorter edge to 224, then center crop to 224 × 224 |
| SigLIP 2 | Bilinear | (0.5, 0.5, 0.5) | (0.5, 0.5, 0.5) | Direct resize to 224 × 224 |

### Optimization and validation selection

Endpoint methods use 300 FP32 Adam updates, epsilon $10^{-4}$, betas $(0.9,0.999)$ and cosine scheduling. Each update uses the full support set with one view per image. With a zero-based step index, the view is `(step + 1) % 32`.

| Learning rate | Displacement-penalty candidates |
| --: | :-- |
| 0.001 | $1/K$, 0, 0.01, 0.1, 1, 10, 100 |
| 0.0001 | $1/K$ |
| 0.01 | $1/K$ |

Final-step validation correct count selects one of the nine candidates. Ties prefer $(10^{-3},1/K)$, then a smaller learning rate, then a larger penalty. Models are fixed before test evaluation. TextCoord, ProLIP, LoRA, SVD-E, Comp-E and the validation-selected coordinate controls use this protocol.

## Accuracy comparisons

These tables reproduce the main comparison, including both LoRA configurations and all three Comp-E configurations. Values are test accuracy (%) as mean ± sample SD. Within each encoder/method group, the configuration with the highest equally weighted mean over the eight settings has a bold configuration label.

### CLIP ViT-B/16: 4-shot

| Method | Configuration | DTD | EuroSAT | Pets | SUN397 |
| --- | --- | ---: | ---: | ---: | ---: |
| LP | *Base* | 55.61&nbsp;±&nbsp;1.39 | 71.12&nbsp;±&nbsp;2.62 | 69.58&nbsp;±&nbsp;1.20 | 62.74&nbsp;±&nbsp;0.25 |
|  | ***LP++*** | 62.23&nbsp;±&nbsp;0.51 | 71.21&nbsp;±&nbsp;5.74 | 91.90&nbsp;±&nbsp;0.34 | 72.75&nbsp;±&nbsp;0.49 |
| ProKeR | *Kernel solve* | 62.43&nbsp;±&nbsp;0.19 | 79.44&nbsp;±&nbsp;1.57 | 90.30&nbsp;±&nbsp;1.57 | 65.36&nbsp;±&nbsp;0.20 |
| ProLIP | *Full* | 63.53&nbsp;±&nbsp;1.30 | 81.70&nbsp;±&nbsp;1.56 | 92.15&nbsp;±&nbsp;0.24 | 72.16&nbsp;±&nbsp;0.59 |
| LoRA | ***Rank*** | 62.53&nbsp;±&nbsp;2.15 | 70.38&nbsp;±&nbsp;11.26 | 92.27&nbsp;±&nbsp;0.15 | 72.16&nbsp;±&nbsp;0.61 |
|  | *Param.* | 62.47&nbsp;±&nbsp;1.29 | 71.33&nbsp;±&nbsp;7.69 | 92.05&nbsp;±&nbsp;0.19 | 71.63&nbsp;±&nbsp;0.44 |
| SVD-E | *Singular values* | 47.42&nbsp;±&nbsp;0.24 | 69.70&nbsp;±&nbsp;1.06 | 91.50&nbsp;±&nbsp;0.60 | 66.57&nbsp;±&nbsp;0.05 |
| Comp-E | *k = 2* | 50.87&nbsp;±&nbsp;1.70 | 67.61&nbsp;±&nbsp;4.25 | 91.42&nbsp;±&nbsp;0.46 | 68.13&nbsp;±&nbsp;0.39 |
|  | ***Rank*** | 63.44&nbsp;±&nbsp;2.28 | 80.94&nbsp;±&nbsp;1.69 | 92.16&nbsp;±&nbsp;0.13 | 72.73&nbsp;±&nbsp;0.73 |
|  | *Param.* | 62.92&nbsp;±&nbsp;2.58 | 74.74&nbsp;±&nbsp;10.94 | 92.09&nbsp;±&nbsp;0.13 | 72.56&nbsp;±&nbsp;0.40 |
| TextCoord | *Task-text* | 63.91&nbsp;±&nbsp;1.40 | 81.29&nbsp;±&nbsp;0.65 | 92.23&nbsp;±&nbsp;0.14 | 72.78&nbsp;±&nbsp;0.49 |

### CLIP ViT-B/16: 16-shot

| Method | Configuration | DTD | EuroSAT | Pets | SUN397 |
| --- | --- | ---: | ---: | ---: | ---: |
| LP | *Base* | 68.72&nbsp;±&nbsp;1.14 | 85.89&nbsp;±&nbsp;1.07 | 85.78&nbsp;±&nbsp;0.68 | 73.13&nbsp;±&nbsp;0.11 |
|  | ***LP++*** | 71.30&nbsp;±&nbsp;0.60 | 85.32&nbsp;±&nbsp;2.79 | 93.06&nbsp;±&nbsp;0.45 | 76.25&nbsp;±&nbsp;0.08 |
| ProKeR | *Kernel solve* | 73.29&nbsp;±&nbsp;0.92 | 87.52&nbsp;±&nbsp;3.03 | 92.92&nbsp;±&nbsp;0.29 | — |
| ProLIP | *Full* | 72.58&nbsp;±&nbsp;0.16 | 88.12&nbsp;±&nbsp;0.82 | 92.70&nbsp;±&nbsp;1.06 | 75.84&nbsp;±&nbsp;0.05 |
| LoRA | ***Rank*** | 71.55&nbsp;±&nbsp;0.21 | 87.51&nbsp;±&nbsp;0.49 | 93.10&nbsp;±&nbsp;0.46 | 75.62&nbsp;±&nbsp;0.26 |
|  | *Param.* | 70.49&nbsp;±&nbsp;0.42 | 86.36&nbsp;±&nbsp;1.07 | 93.04&nbsp;±&nbsp;0.42 | 75.24&nbsp;±&nbsp;0.06 |
| SVD-E | *Singular values* | 48.60&nbsp;±&nbsp;0.22 | 72.07&nbsp;±&nbsp;0.98 | 92.25&nbsp;±&nbsp;0.09 | 66.58&nbsp;±&nbsp;0.13 |
| Comp-E | *k = 2* | 56.42&nbsp;±&nbsp;0.74 | 78.18&nbsp;±&nbsp;2.43 | 91.88&nbsp;±&nbsp;0.24 | 69.66&nbsp;±&nbsp;0.10 |
|  | ***Rank*** | 72.54&nbsp;±&nbsp;0.83 | 88.03&nbsp;±&nbsp;1.09 | 93.03&nbsp;±&nbsp;0.35 | 76.12&nbsp;±&nbsp;0.19 |
|  | *Param.* | 71.83&nbsp;±&nbsp;0.93 | 87.65&nbsp;±&nbsp;0.88 | 93.06&nbsp;±&nbsp;0.27 | 76.06&nbsp;±&nbsp;0.21 |
| TextCoord | *Task-text* | 72.91&nbsp;±&nbsp;0.96 | 88.93&nbsp;±&nbsp;0.98 | 93.33&nbsp;±&nbsp;0.63 | 76.28&nbsp;±&nbsp;0.21 |

### SigLIP 2 Base/16: 4-shot

| Method | Configuration | DTD | EuroSAT | Pets | SUN397 |
| --- | --- | ---: | ---: | ---: | ---: |
| LP | *Base* | 65.94&nbsp;±&nbsp;0.95 | 69.77&nbsp;±&nbsp;4.17 | 84.91&nbsp;±&nbsp;1.10 | 68.25&nbsp;±&nbsp;0.68 |
|  | ***LP++*** | 73.03&nbsp;±&nbsp;1.03 | 67.46&nbsp;±&nbsp;4.86 | 94.88&nbsp;±&nbsp;0.29 | 77.43&nbsp;±&nbsp;0.10 |
| ProKeR | *Kernel solve* | 73.17&nbsp;±&nbsp;0.83 | 78.58&nbsp;±&nbsp;1.17 | 94.90&nbsp;±&nbsp;0.05 | 74.32&nbsp;±&nbsp;0.16 |
| ProLIP | *Full* | 73.48&nbsp;±&nbsp;1.08 | 78.45&nbsp;±&nbsp;3.69 | 94.96&nbsp;±&nbsp;0.27 | 76.94&nbsp;±&nbsp;0.14 |
| LoRA | ***Rank*** | 73.56&nbsp;±&nbsp;0.69 | 74.97&nbsp;±&nbsp;1.10 | 94.91&nbsp;±&nbsp;0.20 | 76.82&nbsp;±&nbsp;0.21 |
|  | *Param.* | 72.97&nbsp;±&nbsp;1.03 | 70.07&nbsp;±&nbsp;1.98 | 94.87&nbsp;±&nbsp;0.13 | 76.78&nbsp;±&nbsp;0.20 |
| SVD-E | *Singular values* | 63.93&nbsp;±&nbsp;0.50 | 48.94&nbsp;±&nbsp;0.35 | 94.81&nbsp;±&nbsp;0.13 | 72.80&nbsp;±&nbsp;0.18 |
| Comp-E | *k = 2* | 68.68&nbsp;±&nbsp;1.86 | 62.63&nbsp;±&nbsp;5.24 | 94.89&nbsp;±&nbsp;0.29 | 74.61&nbsp;±&nbsp;0.23 |
|  | ***Rank*** | 72.77&nbsp;±&nbsp;1.67 | 69.16&nbsp;±&nbsp;4.44 | 94.98&nbsp;±&nbsp;0.17 | 77.07&nbsp;±&nbsp;0.10 |
|  | *Param.* | 72.73&nbsp;±&nbsp;1.13 | 62.63&nbsp;±&nbsp;5.24 | 94.97&nbsp;±&nbsp;0.25 | 76.89&nbsp;±&nbsp;0.08 |
| TextCoord | *Task-text* | 74.82&nbsp;±&nbsp;0.77 | 76.36&nbsp;±&nbsp;2.26 | 94.97&nbsp;±&nbsp;0.25 | 77.27&nbsp;±&nbsp;0.17 |

### SigLIP 2 Base/16: 16-shot

| Method | Configuration | DTD | EuroSAT | Pets | SUN397 |
| --- | --- | ---: | ---: | ---: | ---: |
| LP | *Base* | 77.44&nbsp;±&nbsp;0.36 | 85.38&nbsp;±&nbsp;0.46 | 92.03&nbsp;±&nbsp;0.68 | 77.13&nbsp;±&nbsp;0.23 |
|  | ***LP++*** | 78.72&nbsp;±&nbsp;0.51 | 84.43&nbsp;±&nbsp;1.52 | 94.80&nbsp;±&nbsp;0.16 | 80.05&nbsp;±&nbsp;0.26 |
| ProKeR | *Kernel solve* | 80.22&nbsp;±&nbsp;1.31 | 88.96&nbsp;±&nbsp;0.17 | 95.23&nbsp;±&nbsp;0.17 | — |
| ProLIP | *Full* | 80.24&nbsp;±&nbsp;1.31 | 88.28&nbsp;±&nbsp;0.73 | 95.13&nbsp;±&nbsp;0.16 | 79.58&nbsp;±&nbsp;0.14 |
| LoRA | ***Rank*** | 80.02&nbsp;±&nbsp;0.99 | 86.41&nbsp;±&nbsp;0.80 | 94.95&nbsp;±&nbsp;0.28 | 79.08&nbsp;±&nbsp;0.70 |
|  | *Param.* | 79.39&nbsp;±&nbsp;0.77 | 82.58&nbsp;±&nbsp;0.89 | 95.00&nbsp;±&nbsp;0.06 | 79.48&nbsp;±&nbsp;0.10 |
| SVD-E | *Singular values* | 64.54&nbsp;±&nbsp;0.20 | 49.13&nbsp;±&nbsp;0.38 | 94.99&nbsp;±&nbsp;0.18 | 73.06&nbsp;±&nbsp;0.01 |
| Comp-E | *k = 2* | 73.70&nbsp;±&nbsp;0.54 | 73.31&nbsp;±&nbsp;0.99 | 95.05&nbsp;±&nbsp;0.32 | 76.30&nbsp;±&nbsp;0.05 |
|  | ***Rank*** | 78.72&nbsp;±&nbsp;0.62 | 83.33&nbsp;±&nbsp;0.16 | 95.11&nbsp;±&nbsp;0.23 | 79.48&nbsp;±&nbsp;0.18 |
|  | *Param.* | 77.05&nbsp;±&nbsp;0.54 | 73.31&nbsp;±&nbsp;0.99 | 95.38&nbsp;±&nbsp;0.07 | 79.23&nbsp;±&nbsp;0.12 |
| TextCoord | *Task-text* | 80.79&nbsp;±&nbsp;0.82 | 87.71&nbsp;±&nbsp;0.52 | 95.08&nbsp;±&nbsp;0.11 | 79.95&nbsp;±&nbsp;0.08 |

Across the sixteen settings, TextCoord exceeds Comp-E Rank and Param. by 1.19 and 2.84 percentage points on average. On the shared CLIP projector, TextCoord exceeds SVD-E and Comp-E $k=2$ in all eight setting means. The SigLIP 2 spectral comparisons include adaptation location as part of the complete configuration: FC2 for SVD-E/Comp-E and the output residual for TextCoord.

## Coordinate and normalization analysis

### Validation-selected coordinates

The coordinate controls examine regularization and optimization within update families having the same centered unnormalized score expressiveness. For a same-rank random orthonormal basis $U$, set $K_U=U^\top Q$ and $G=A_UK_U$. Then

$$
hA_UU^\top T_\Delta=hGR.
$$

R0 optimizes $A_U$ with penalty $\lVert A_U\rVert_F^2$. R1 optimizes $A_U$ with penalty $\lVert G\rVert_F^2$. R2 directly optimizes $G$ with that penalty and obtains $A_U$ by a linear solve. For each diagnostic task, one random basis generated with seed 8675309 is shared by R0–R2 across both shot counts and all three support seeds. Each model retains the native visual component perpendicular to $U$ and uses its actual dynamic normalization denominator.

| Encoder | Dataset / shot | TextCoord | R0 | R1 | R2 |
| --- | --- | ---: | ---: | ---: | ---: |
| CLIP ViT-B/16 | DTD / 4 | 63.91&nbsp;±&nbsp;1.40 | 62.27&nbsp;±&nbsp;1.24 | 62.81&nbsp;±&nbsp;1.21 | 62.37&nbsp;±&nbsp;0.30 |
| CLIP ViT-B/16 | DTD / 16 | 72.91&nbsp;±&nbsp;0.96 | 70.41&nbsp;±&nbsp;0.57 | 70.78&nbsp;±&nbsp;0.44 | 68.79&nbsp;±&nbsp;0.67 |
| SigLIP 2 Base/16 | EuroSAT / 4 | 76.36&nbsp;±&nbsp;2.26 | 62.25&nbsp;±&nbsp;3.57 | 67.16&nbsp;±&nbsp;3.60 | 72.26&nbsp;±&nbsp;0.65 |
| SigLIP 2 Base/16 | EuroSAT / 16 | 87.71&nbsp;±&nbsp;0.52 | 67.60&nbsp;±&nbsp;1.30 | 79.65&nbsp;±&nbsp;1.54 | 81.95&nbsp;±&nbsp;1.86 |

R1 improves upon R0 in the four setting means. R2 improves upon R1 on EuroSAT and decreases accuracy on DTD, indicating a setting-dependent effect of optimizer coordinates. TextCoord has the highest mean in each of these four comparisons. R2 matches TextCoord's centered unnormalized score parameterization and penalty form. Its coupled update perpendicular to the task-text span generally yields a different normalization denominator from TextCoord, which preserves the native perpendicular component.

### Fixed-configuration coordinate diagnostics

These diagnostics fix the learning rate to $10^{-3}$ and $\lambda=1/K$ for all four methods, with 300 updates. This controlled configuration distinguishes them from the validation-selected table above.

| Encoder | Dataset / shot | TextCoord | R0 | R1 | R2 |
| --- | --- | ---: | ---: | ---: | ---: |
| CLIP ViT-B/16 | DTD / 4 | 64.18&nbsp;±&nbsp;1.39 | 55.93&nbsp;±&nbsp;0.71 | 61.94&nbsp;±&nbsp;1.05 | 61.39&nbsp;±&nbsp;0.68 |
| CLIP ViT-B/16 | DTD / 16 | 73.11&nbsp;±&nbsp;0.52 | 65.07&nbsp;±&nbsp;0.52 | 70.27&nbsp;±&nbsp;0.24 | 68.40&nbsp;±&nbsp;0.80 |
| SigLIP 2 Base/16 | EuroSAT / 4 | 74.79&nbsp;±&nbsp;1.52 | 49.47&nbsp;±&nbsp;0.16 | 62.16&nbsp;±&nbsp;1.55 | 69.29&nbsp;±&nbsp;0.62 |
| SigLIP 2 Base/16 | EuroSAT / 16 | 85.67&nbsp;±&nbsp;0.74 | 53.10&nbsp;±&nbsp;0.38 | 67.14&nbsp;±&nbsp;1.32 | 79.69&nbsp;±&nbsp;1.53 |

The fixed-configuration results show the same direction of improvement from R0 to R1 across the four setting means. The R1-to-R2 change again depends on the task. This comparison examines the coordinate choices at a shared learning rate and penalty.

### Normalization and regularization

D uses the current adapted feature norm; F fixes the denominator to $\lVert z_0\rVert_2$ for each image or view. Disp. uses $\lambda\lVert A\rVert_F^2$. CI uses $\lambda\lVert AR\rVert_F^2/\gamma$, where $\gamma=\lVert R\rVert_F^2/r$. All entries use the nine-candidate validation search.

| Encoder | Dataset / shot | TextCoord: D/Disp. | D/CI | F/CI | F/Disp. | Random |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CLIP ViT-B/16 | DTD / 4 | 63.91&nbsp;±&nbsp;1.40 | 64.20&nbsp;±&nbsp;1.21 | 63.55&nbsp;±&nbsp;2.13 | 63.83&nbsp;±&nbsp;1.08 | 62.27&nbsp;±&nbsp;1.24 |
| CLIP ViT-B/16 | DTD / 16 | 72.91&nbsp;±&nbsp;0.96 | 73.38&nbsp;±&nbsp;0.36 | 73.15&nbsp;±&nbsp;0.42 | 73.17&nbsp;±&nbsp;0.56 | 70.41&nbsp;±&nbsp;0.57 |
| SigLIP 2 Base/16 | EuroSAT / 4 | 76.36&nbsp;±&nbsp;2.26 | 76.23&nbsp;±&nbsp;1.38 | 76.79&nbsp;±&nbsp;0.24 | 75.43&nbsp;±&nbsp;1.30 | 62.25&nbsp;±&nbsp;3.57 |
| SigLIP 2 Base/16 | EuroSAT / 16 | 87.71&nbsp;±&nbsp;0.52 | 87.13&nbsp;±&nbsp;0.30 | 87.35&nbsp;±&nbsp;0.99 | 87.43&nbsp;±&nbsp;0.67 | 67.60&nbsp;±&nbsp;1.30 |

The best normalization/regularization variant varies across these settings. All text-coordinate variants exceed the random-coordinate reference in the four setting means. TextCoord combines normalized prediction with displacement regularization, matching the minimum-displacement interpretation of the method.

### Algebraic checks

With a fixed denominator, R2 and TextCoord agree in centered logits, loss and gradients when written in aligned coordinates. These checks use the actual task bases and FP64 arithmetic; the solve residual is measured in FP32.

| Encoder / task | FP32 solve relative residual | FP64 logit max error | FP64 loss error | FP64 gradient max error |
| --- | ---: | ---: | ---: | ---: |
| CLIP ViT-B/16 / DTD | 2.022e-06 | 9.714e-17 | 0.000e+00 | 8.327e-17 |
| SigLIP 2 Base/16 / EuroSAT | 3.075e-07 | 6.245e-17 | 0.000e+00 | 5.074e-17 |

The source package includes CPU tests of compact versus explicit probabilities, cross-entropy and gradients, zero-update recovery, fixed-denominator coordinate equivalence, complementary-subspace constraints and the ProKeR linear solve.

## Adaptation-position analysis

This comparison evaluates TextCoord and ProLIP at the SigLIP 2 output and FC2 interfaces on DTD and EuroSAT, with both shot counts, seeds 1/2/3 and validation selection. At FC2, $z=z_0+h\Delta W$ retains the complete native bias and residual in $z_0$.

| Encoder | Dataset / shot | TextCoord output | ProLIP output | TextCoord FC2 | ProLIP FC2 |
| --- | --- | ---: | ---: | ---: | ---: |
| SigLIP 2 Base/16 | DTD / 4 | 74.82&nbsp;±&nbsp;0.77 | 73.48&nbsp;±&nbsp;1.08 | 73.23&nbsp;±&nbsp;1.92 | 72.71&nbsp;±&nbsp;1.39 |
| SigLIP 2 Base/16 | DTD / 16 | 80.79&nbsp;±&nbsp;0.82 | 80.24&nbsp;±&nbsp;1.31 | 79.35&nbsp;±&nbsp;0.55 | 78.49&nbsp;±&nbsp;0.52 |
| SigLIP 2 Base/16 | EuroSAT / 4 | 76.36&nbsp;±&nbsp;2.26 | 78.45&nbsp;±&nbsp;3.69 | 69.12&nbsp;±&nbsp;3.27 | 70.96&nbsp;±&nbsp;2.17 |
| SigLIP 2 Base/16 | EuroSAT / 16 | 87.71&nbsp;±&nbsp;0.52 | 88.28&nbsp;±&nbsp;0.73 | 84.17&nbsp;±&nbsp;1.30 | 84.57&nbsp;±&nbsp;0.64 |

Both TextCoord and ProLIP achieve higher mean accuracy at the output interface than at FC2 in all four settings. This supports the output interface used by the main SigLIP 2 configuration. Same-interface ProLIP/LoRA comparisons examine parameterization within that choice.

## Paired accuracy differences

Each entry is TextCoord minus the comparator in percentage points, paired by support seed and summarized as mean ± sample SD of the three differences. All main-table configurations are covered.

### CLIP ViT-B/16: 4-shot

| Method | Configuration | DTD | EuroSAT | Pets | SUN397 |
| --- | --- | ---: | ---: | ---: | ---: |
| LP | *Base* | +8.294&nbsp;±&nbsp;1.041 | +10.173&nbsp;±&nbsp;3.006 | +22.649&nbsp;±&nbsp;1.260 | +10.037&nbsp;±&nbsp;0.548 |
|  | *LP++* | +1.675&nbsp;±&nbsp;1.045 | +10.074&nbsp;±&nbsp;5.755 | +0.336&nbsp;±&nbsp;0.316 | +0.030&nbsp;±&nbsp;0.241 |
| ProKeR | *Kernel solve* | +1.478&nbsp;±&nbsp;1.238 | +1.844&nbsp;±&nbsp;2.070 | +1.935&nbsp;±&nbsp;1.679 | +7.426&nbsp;±&nbsp;0.675 |
| ProLIP | *Full* | +0.374&nbsp;±&nbsp;1.077 | -0.412&nbsp;±&nbsp;1.094 | +0.082&nbsp;±&nbsp;0.233 | +0.621&nbsp;±&nbsp;0.150 |
| LoRA | *Rank* | +1.379&nbsp;±&nbsp;0.961 | +10.905&nbsp;±&nbsp;10.732 | -0.036&nbsp;±&nbsp;0.016 | +0.616&nbsp;±&nbsp;0.172 |
|  | *Param.* | +1.438&nbsp;±&nbsp;1.019 | +9.959&nbsp;±&nbsp;7.210 | +0.182&nbsp;±&nbsp;0.244 | +1.152&nbsp;±&nbsp;0.550 |
| SVD-E | *Singular values* | +16.489&nbsp;±&nbsp;1.537 | +11.584&nbsp;±&nbsp;1.031 | +0.736&nbsp;±&nbsp;0.459 | +6.213&nbsp;±&nbsp;0.477 |
| Comp-E | *k = 2* | +13.042&nbsp;±&nbsp;0.444 | +13.675&nbsp;±&nbsp;3.855 | +0.809&nbsp;±&nbsp;0.557 | +4.653&nbsp;±&nbsp;0.212 |
|  | *Rank* | +0.473&nbsp;±&nbsp;1.025 | +0.346&nbsp;±&nbsp;1.391 | +0.073&nbsp;±&nbsp;0.031 | +0.047&nbsp;±&nbsp;0.251 |
|  | *Param.* | +0.985&nbsp;±&nbsp;1.183 | +6.547&nbsp;±&nbsp;10.305 | +0.145&nbsp;±&nbsp;0.134 | +0.222&nbsp;±&nbsp;0.179 |

### CLIP ViT-B/16: 16-shot

| Method | Configuration | DTD | EuroSAT | Pets | SUN397 |
| --- | --- | ---: | ---: | ---: | ---: |
| LP | *Base* | +4.196&nbsp;±&nbsp;0.625 | +3.037&nbsp;±&nbsp;0.105 | +7.550&nbsp;±&nbsp;1.071 | +3.150&nbsp;±&nbsp;0.174 |
|  | *LP++* | +1.615&nbsp;±&nbsp;0.607 | +3.609&nbsp;±&nbsp;2.024 | +0.273&nbsp;±&nbsp;0.813 | +0.025&nbsp;±&nbsp;0.279 |
| ProKeR | *Kernel solve* | -0.374&nbsp;±&nbsp;0.839 | +1.407&nbsp;±&nbsp;2.416 | +0.409&nbsp;±&nbsp;0.482 | — |
| ProLIP | *Full* | +0.335&nbsp;±&nbsp;0.807 | +0.815&nbsp;±&nbsp;0.229 | +0.636&nbsp;±&nbsp;0.559 | +0.432&nbsp;±&nbsp;0.242 |
| LoRA | *Rank* | +1.359&nbsp;±&nbsp;0.827 | +1.416&nbsp;±&nbsp;0.507 | +0.236&nbsp;±&nbsp;0.469 | +0.657&nbsp;±&nbsp;0.105 |
|  | *Param.* | +2.423&nbsp;±&nbsp;0.977 | +2.568&nbsp;±&nbsp;0.172 | +0.291&nbsp;±&nbsp;0.382 | +1.038&nbsp;±&nbsp;0.193 |
| SVD-E | *Singular values* | +24.310&nbsp;±&nbsp;0.769 | +16.864&nbsp;±&nbsp;0.362 | +1.081&nbsp;±&nbsp;0.575 | +9.694&nbsp;±&nbsp;0.182 |
| Comp-E | *k = 2* | +16.489&nbsp;±&nbsp;1.407 | +10.753&nbsp;±&nbsp;1.640 | +1.454&nbsp;±&nbsp;0.394 | +6.618&nbsp;±&nbsp;0.180 |
|  | *Rank* | +0.374&nbsp;±&nbsp;0.267 | +0.901&nbsp;±&nbsp;0.194 | +0.300&nbsp;±&nbsp;0.509 | +0.156&nbsp;±&nbsp;0.281 |
|  | *Param.* | +1.084&nbsp;±&nbsp;0.448 | +1.284&nbsp;±&nbsp;0.612 | +0.273&nbsp;±&nbsp;0.450 | +0.213&nbsp;±&nbsp;0.299 |

### SigLIP 2 Base/16: 4-shot

| Method | Configuration | DTD | EuroSAT | Pets | SUN397 |
| --- | --- | ---: | ---: | ---: | ---: |
| LP | *Base* | +8.885&nbsp;±&nbsp;1.064 | +6.593&nbsp;±&nbsp;2.808 | +10.057&nbsp;±&nbsp;0.946 | +9.029&nbsp;±&nbsp;0.655 |
|  | *LP++* | +1.793&nbsp;±&nbsp;0.380 | +8.901&nbsp;±&nbsp;4.992 | +0.091&nbsp;±&nbsp;0.504 | -0.156&nbsp;±&nbsp;0.074 |
| ProKeR | *Kernel solve* | +1.655&nbsp;±&nbsp;0.505 | -2.226&nbsp;±&nbsp;1.623 | +0.064&nbsp;±&nbsp;0.265 | +2.957&nbsp;±&nbsp;0.224 |
| ProLIP | *Full* | +1.340&nbsp;±&nbsp;0.595 | -2.095&nbsp;±&nbsp;3.070 | +0.009&nbsp;±&nbsp;0.164 | +0.331&nbsp;±&nbsp;0.028 |
| LoRA | *Rank* | +1.261&nbsp;±&nbsp;0.549 | +1.391&nbsp;±&nbsp;2.509 | +0.055&nbsp;±&nbsp;0.213 | +0.452&nbsp;±&nbsp;0.049 |
|  | *Param.* | +1.852&nbsp;±&nbsp;1.229 | +6.288&nbsp;±&nbsp;3.250 | +0.100&nbsp;±&nbsp;0.247 | +0.494&nbsp;±&nbsp;0.036 |
| SVD-E | *Singular values* | +10.894&nbsp;±&nbsp;0.326 | +27.420&nbsp;±&nbsp;1.946 | +0.154&nbsp;±&nbsp;0.345 | +4.479&nbsp;±&nbsp;0.061 |
| Comp-E | *k = 2* | +6.147&nbsp;±&nbsp;2.201 | +13.724&nbsp;±&nbsp;3.041 | +0.073&nbsp;±&nbsp;0.123 | +2.662&nbsp;±&nbsp;0.397 |
|  | *Rank* | +2.049&nbsp;±&nbsp;1.862 | +7.193&nbsp;±&nbsp;2.520 | -0.009&nbsp;±&nbsp;0.150 | +0.205&nbsp;±&nbsp;0.072 |
|  | *Param.* | +2.088&nbsp;±&nbsp;0.451 | +13.724&nbsp;±&nbsp;3.041 | +0.000&nbsp;±&nbsp;0.119 | +0.383&nbsp;±&nbsp;0.105 |

### SigLIP 2 Base/16: 16-shot

| Method | Configuration | DTD | EuroSAT | Pets | SUN397 |
| --- | --- | ---: | ---: | ---: | ---: |
| LP | *Base* | +3.349&nbsp;±&nbsp;0.503 | +2.325&nbsp;±&nbsp;0.093 | +3.053&nbsp;±&nbsp;0.629 | +2.821&nbsp;±&nbsp;0.157 |
|  | *LP++* | +2.069&nbsp;±&nbsp;0.313 | +3.280&nbsp;±&nbsp;1.024 | +0.282&nbsp;±&nbsp;0.222 | -0.102&nbsp;±&nbsp;0.209 |
| ProKeR | *Kernel solve* | +0.571&nbsp;±&nbsp;0.865 | -1.251&nbsp;±&nbsp;0.519 | -0.145&nbsp;±&nbsp;0.268 | — |
| ProLIP | *Full* | +0.552&nbsp;±&nbsp;0.685 | -0.568&nbsp;±&nbsp;0.633 | -0.045&nbsp;±&nbsp;0.088 | +0.369&nbsp;±&nbsp;0.102 |
| LoRA | *Rank* | +0.768&nbsp;±&nbsp;0.307 | +1.296&nbsp;±&nbsp;0.701 | +0.136&nbsp;±&nbsp;0.300 | +0.873&nbsp;±&nbsp;0.784 |
|  | *Param.* | +1.399&nbsp;±&nbsp;0.685 | +5.132&nbsp;±&nbsp;0.403 | +0.082&nbsp;±&nbsp;0.119 | +0.472&nbsp;±&nbsp;0.025 |
| SVD-E | *Singular values* | +16.253&nbsp;±&nbsp;0.755 | +38.580&nbsp;±&nbsp;0.331 | +0.091&nbsp;±&nbsp;0.291 | +6.888&nbsp;±&nbsp;0.077 |
| Comp-E | *k = 2* | +7.092&nbsp;±&nbsp;1.095 | +14.399&nbsp;±&nbsp;0.811 | +0.036&nbsp;±&nbsp;0.426 | +3.652&nbsp;±&nbsp;0.031 |
|  | *Rank* | +2.069&nbsp;±&nbsp;0.236 | +4.383&nbsp;±&nbsp;0.640 | -0.027&nbsp;±&nbsp;0.342 | +0.469&nbsp;±&nbsp;0.193 |
|  | *Param.* | +3.743&nbsp;±&nbsp;0.356 | +14.399&nbsp;±&nbsp;0.811 | -0.300&nbsp;±&nbsp;0.144 | +0.720&nbsp;±&nbsp;0.152 |

The [recorded results](data/paper_results.json) contain all 522 per-seed main-table accuracies, aggregate tables, paired differences and supporting controls. The [coordinate selection records](data/selected_coordinate_configs.json) give learning rates, penalties and validation counts for the 48 main coordinate-control fits.

## Baseline implementation details

### ProLIP and LoRA

ProLIP learns a zero-initialized full increment at the TextCoord interface with penalty $\lambda\lVert \Delta W\rVert_F^2$.

LoRA learns $\Delta W=BC$, with $B\in\mathbb{R}^{d\times k}$ and $C\in\mathbb{R}^{k\times e}$, scale one and dropout zero. Both factors are trained. $B$ starts at zero; $C$ is uniform on $[-1/\sqrt e,1/\sqrt e]$, seeded by the support seed. The penalty is $\lambda\lVert BC\rVert_F^2$.

### SVD-E and Comp-E

SVD-E implements singular-value tuning from [CLIP-SVD](https://openreview.net/forum?id=XYy8pwqwMR) at the specified endpoint. For a thin SVD $W_0=U\Sigma V^\top$, it learns a zero-initialized vector $\delta s$:

$$
\Delta W=U\operatorname{diag}(\delta s)V^\top,\qquad
\mathcal{R}=\lambda\lVert \delta s\rVert_2^2.
$$

Comp-E implements complementary-subspace adaptation from [Comp-LoRA](https://doi.org/10.1109/ICASSP55912.2026.11461843). It removes the leading $p=16$ directions from complete rectangular left and right singular bases, retaining complementary bases $U_c,V_c$, including rectangular nullspaces:

$$
\Delta W=U_cBA V_c^\top,\quad
B\in\mathbb{R}^{(d_c-p)\times k},\quad
A\in\mathbb{R}^{k\times(e_c-p)}.
$$

$B$ starts at zero; $A$ is uniform on $[-1/\sqrt{e_c-p},1/\sqrt{e_c-p}]$, seeded by the support seed. Both factors are trained with penalty $\lambda\lVert BA\rVert_F^2$. SigLIP 2 FC2 bias and residual remain fixed.

### LP and LP++

LP Base and LP++ use the first cached, normalized view of each support image.

LP fits a linear classifier and unregularized bias in FP64:

$$
\frac{1}{N}\sum_i\mathrm{CE}(Wx_i+b,y_i)+\frac{\lVert W\rVert_F^2}{2c_{\rm LP}N}.
$$

The validation grid is $c_{\rm LP}\in\lbrace 10^{-6},10^{-4},10^{-2},0.316,1,10^2,10^4,10^6\rbrace$; ties favor smaller $c_{\rm LP}$. Weights and bias start at zero. PyTorch L-BFGS uses strong-Wolfe line search, history size 10, at most 1,000 iterations and 5,000 objective evaluations, gradient tolerance $10^{-8}$ and change tolerance $10^{-12}$.

For LP++, let $X$ contain normalized support features, $Y$ be the one-hot labels and $S=XT^\top$, with text rows in class order. Classifier weights start at $Y^\top X$; the bias uses the linear-layer random initialization. All class text coefficients share

$$
\alpha_0=\frac{250}{C}\sum_c\frac{1}{K}\sum_{i:y_i=c}S_{ic}.
$$

The classifier learning rate is $4N/\lambda_{\max}(X^\top X)$, and the text-coefficient learning rate is $N/(4\max_c\sum_iS_{ic}^2)$. The classifier uses FP32 SGD with momentum 0.9 for 300 epochs. Every tenth epoch updates the text coefficients using a fresh gradient at the updated classifier. Validation selects the epoch, with ties favoring the later epoch.

### ProKeR

ProKeR uses ten normalized views per support image in view-major order. In FP64, it forms $K_\beta(x,x')=\exp[-\beta(1-x^\top x')]$ and solves

$$
(K_\beta/\lambda+I)\alpha=Y-XT^\top.
$$

Prediction is $xT^\top+K_\beta(x,X)\alpha$. Each kernel is diagonalized once for its full penalty grid, retaining all eigenvalues. Both axes use linearly spaced points including their endpoints, constructed by FP32 `torch.linspace` and used in the FP64 solves.

| Dataset | Beta range and count | Lambda range and count | Candidates |
| :-- | :-- | :-- | --: |
| DTD | [0.1, 10], 20 | [0.001, 10], 100 | 2,000 |
| EuroSAT | [0.5, 3], 20 | [0.001, 1], 100 | 2,000 |
| Pets | [0.01, 5], 40 | [0.01, 0.3], 20 | 800 |
| SUN397 | [0.1, 10], 20 | [0.01, 10], 10 | 200 |

Validation ties retain the first candidate in beta-major, lambda-minor order.

## Running the code

The package supports CPU and explicitly requested CUDA devices. **CPU is the default.** Run commands from the repository root with Python 3.10–3.13; Python 3.12 is used for the release checks.

```text
git clone https://github.com/xyycyc/task-text-coordinates-resources.git
cd task-text-coordinates-resources
python -m pip install -e ".[encoders,test]"
python -m tasktext smoke --output outputs/cpu-smoke
python -m pytest -q
```

The smoke command runs all 17 supported configurations on synthetic CPU tensors, checks finite predictions and reloads every saved adapter. Recorded paper results are stored separately in `data/paper_results.json`.

### Prepare dataset images

Download and extract the datasets from their official sources. The code reads identities from the supplied manifests.

| Dataset | Official source | `--data-root` points to | Manifest |
| :-- | :-- | :-- | :-- |
| DTD | [Oxford VGG DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/) | `dtd/images`, containing `banded/`, etc. | `data/splits/dtd.json` |
| EuroSAT | [EuroSAT](https://github.com/phelber/EuroSAT) | `2750` or `EuroSAT_RGB`, containing class folders | `data/splits/eurosat.json` |
| Pets | [Oxford-IIIT Pets](https://www.robots.ox.ac.uk/~vgg/data/pets/) | `images`, containing JPEG files | `data/splits/oxford_pets.json` |
| SUN397 | [SUN397](https://vision.princeton.edu/projects/2010/SUN/) | `SUN397`, containing `a/`, `b/`, etc. | `data/splits/sun397.json` |

### Train, select and evaluate an episode

This example uses CLIP, DTD, 4 shots and seed 1. Replace `DATASET_IMAGE_DIRECTORY` with the DTD image directory. Commands are valid in PowerShell and POSIX shells.

```text
python -m tasktext encode --manifest data/splits/dtd.json --data-root DATASET_IMAGE_DIRECTORY --family clip --shots 4 --seed 1 --output cache/clip-dtd-k4-s1 --download-root cache/models --device cpu
python -m tasktext train --cache cache/clip-dtd-k4-s1 --methods p0 --output outputs/clip-dtd-k4-s1 --device cpu
python -m tasktext encode --manifest data/splits/dtd.json --data-root DATASET_IMAGE_DIRECTORY --family clip --shots 4 --seed 1 --splits test --output cache/clip-dtd-k4-s1 --download-root cache/models --device cpu
python -m tasktext evaluate --cache cache/clip-dtd-k4-s1 --model outputs/clip-dtd-k4-s1/p0.pt --output outputs/clip-dtd-k4-s1/p0-test.json --device cpu
```

Training reads support and validation caches. Test encoding and evaluation are separate commands. A selected checkpoint records its support/validation and cache identities; evaluation verifies these before predicting. Outputs contain the selected configuration, validation grid, test correct count, accuracy and per-image predictions. Existing fitted model files are protected from overwrite.

For SigLIP 2, use `--family siglip2`. Public checkpoints are downloaded to `--download-root`; `--checkpoint` accepts an existing CLIP `.pt` file or local SigLIP 2 directory. Batch size and CPU thread count are controlled by `--batch-size` and `--threads`. CUDA execution requires `--device cuda:0`.

### Main comparisons and controls

```text
python -m tasktext train --cache CACHE_DIRECTORY --methods p0 prolip lora_rank lora_param svd comp_k2 comp_rank comp_param lp lpplusplus proker --output OUTPUT_DIRECTORY --device cpu
python -m tasktext train --cache CACHE_DIRECTORY --methods p0 r0 r1 r2 --output COORDINATE_OUTPUT_DIRECTORY --device cpu
python -m tasktext train --cache CACHE_DIRECTORY --methods p0 r0 r1 r2 --fixed-config --output FIXED_OUTPUT_DIRECTORY --device cpu
python -m tasktext train --cache CACHE_DIRECTORY --methods dynamic_ci fixed_ci fixed_disp --output NORMALIZATION_OUTPUT_DIRECTORY --device cpu
python -m tasktext train --cache SIGLIP_CACHE_DIRECTORY --methods p0 prolip --location fc2 --output POSITION_OUTPUT_DIRECTORY --device cpu
```

Configurations use the locations and budgets stated above automatically. Commands execute 300 updates and the appropriate validation search. Coordinate controls cover CLIP/DTD and SigLIP 2/EuroSAT; position controls cover SigLIP 2 on DTD/EuroSAT. Both shot counts and all three support seeds are specified in `configs/paper.json`.

### Complete task matrix

Create a dataset-root JSON file:

```json
{
  "dtd": "DATA_ROOT/dtd/images",
  "eurosat": "DATA_ROOT/EuroSAT_RGB",
  "oxford_pets": "DATA_ROOT/oxford_pets/images",
  "sun397": "DATA_ROOT/SUN397"
}
```

```text
python scripts/run_matrix.py --data-roots dataset_roots.json --work-dir outputs/paper --device cpu --phase encode
python scripts/run_matrix.py --data-roots dataset_roots.json --work-dir outputs/paper --device cpu --phase train
python scripts/run_matrix.py --data-roots dataset_roots.json --work-dir outputs/paper --device cpu --phase evaluate
python scripts/summarize_runs.py --input outputs/paper --output outputs/paper/summary.json
```

The matrix includes main comparisons, selected and fixed coordinate controls, normalization/regularization variants and the four SigLIP 2 position settings. ProKeR tasks cover reported 4-shot SUN397 results; the unreported SUN397 16-shot configuration is available through the single-episode command. Identical Comp-E configurations on SigLIP 2 EuroSAT share a fitted model, with both table labels recorded.

### Files and numerical validation

| Path | Contents |
| :-- | :-- |
| `src/tasktext/` | Geometry, adapters, baselines, encoders, caches, selection and prediction |
| `configs/paper.json` | Encoder/dataset/shot/seed matrix and search settings |
| `data/splits/` | Exact class order and image identities |
| `data/paper_results.json` | Recorded paper results and supporting analyses |
| `data/parameter_budgets.json` | Trainable scalar counts and ranks |
| `tests/` | CPU numerical and end-to-end workflow checks |
| `scripts/` | Task-matrix runner, aggregation and README generation |

CPU checks cover both real encoder checkpoints, the image-to-cache-to-selection-to-evaluation workflow, and all adapter/baseline implementations. The release passes 32 tests, including full ProKeR search grids and the 64 endpoint parameter counts in the budget tables. All 17 configurations also complete a 300-step synthetic CPU run. These checks verify numerical identities and software execution. Paper tables retain the recorded experiments; CPU runs can differ in floating-point arithmetic and device-specific random-number sequences. The evaluated extraction environment uses PyTorch 2.7.1, torchvision 0.22.1 and FP32 features; the code release is checked with Transformers 5.16.1.

Regenerate and verify this page's tables from the committed JSON files:

```text
python scripts/build_readme.py
python scripts/validate_resources.py
```

## Sources

- [OpenAI CLIP](https://github.com/openai/CLIP) and [SigLIP 2 Base/16](https://huggingface.co/google/siglip2-base-patch16-224): pretrained encoders.
- [CoOp dataset preparation](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md): public Zhou splits.
- [ProLIP](https://doi.org/10.1109/WACV61042.2026.00318): full visual endpoint adaptation.
- [LP++](https://openaccess.thecvf.com/content/CVPR2024/html/Huang_LP_A_Surprisingly_Strong_Linear_Probe_for_Few-Shot_CLIP_CVPR_2024_paper.html): data-dependent linear probing.
- [ProKeR](https://openaccess.thecvf.com/content/CVPR2025/html/Bendou_ProKeR_A_Kernel_Perspective_on_Few-Shot_Adaptation_of_Large_Vision-Language_CVPR_2025_paper.html) and its [implementation](https://github.com/ybendou/ProKeR): kernel adaptation and search ranges.
- [CLIP-SVD](https://openreview.net/forum?id=XYy8pwqwMR): singular-value adaptation.
- [Comp-LoRA](https://doi.org/10.1109/ICASSP55912.2026.11461843): complementary singular-space adaptation.

Pretrained weights, dataset images and third-party encoder implementations are obtained from upstream projects and retain their respective licenses.
