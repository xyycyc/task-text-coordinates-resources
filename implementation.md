# Implementation and evaluation details

These settings accompany *Task-Text Coordinates for Normalized Few-Shot Vision-Language Adaptation*. The main paper specifies P0, its theoretical properties, the shared training protocol and the principal comparisons. [Parameter budgets](README.md#parameter-budgets) are displayed in the repository README.

## Encoders and class texts

We use OpenAI CLIP ViT-B/16 and `google/siglip2-base-patch16-224`, with frozen encoders and FP32 feature extraction. The endpoint logit scale is 100 for CLIP and the pretrained scale for SigLIP 2 (approximately 112.6689). Feature extraction uses PyTorch 2.7.1 with CUDA 12.8 on an NVIDIA GeForce RTX 3090, with AMP and TF32 disabled.

Each class uses one prompt. Underscores in class names are replaced by spaces, and SigLIP 2 prompts are lowercased. Each text embedding is L2-normalized. SigLIP 2 uses its native tokenizer with a maximum sequence length of 64.

| Dataset | Exact template (`{}` is the class name) |
| --- | --- |
| DTD | `{} texture.` |
| EuroSAT | `a centered satellite photo of {}.` |
| Pets | `a photo of a {}, a type of pet.` |
| SUN397 | `a photo of a {}.` |

## Data splits and image preprocessing

DTD, EuroSAT and SUN397 use the public Zhou split files. For Pets, the official trainval set is split within each class into 80% training and 20% validation using seed 1729; evaluation uses the official test set. Support sets contain 4 or 16 images per class, sampled with seeds 1/2/3. Validation sets contain the corresponding number of images per class, except SUN397 uses the same four-per-class subset at both shot counts. Validation subsets are shared across support seeds and methods. All augmented views of an original image stay in its split.

Training views use RGB conversion, random resized cropping to 224 by 224 pixels with area scale [0.5, 1.0] and aspect ratio [3/4, 4/3], horizontal flipping with probability 0.5, and encoder-specific normalization. Each support image has 32 cached training views.

| Encoder | Training interpolation | Channel means | Channel standard deviations | Evaluation resize and crop |
| --- | --- | --- | --- | --- |
| CLIP | Bicubic | (0.48145466, 0.4578275, 0.40821073) | (0.26862954, 0.26130258, 0.27577711) | Bicubic resize to a shorter edge of 224, then a 224 by 224 center crop |
| SigLIP 2 | Bilinear | (0.5, 0.5, 0.5) | (0.5, 0.5, 0.5) | Direct bilinear resize to 224 by 224 |

## P0

P0 adapts the visual projector of CLIP and a residual on the complete, unnormalized visual pooling output of SigLIP 2. At the SigLIP 2 interface, the frozen base mapping is the identity and the native bias and residual are included in the frozen input. With the paper's notation, the update is $\Delta W=AQ^\top$; only $A$ is trained, starting at zero.

The fixed unit-normalized class texts are promoted to FP64 before centering and reduced SVD. We retain left singular vectors with $\sigma_i>\max(e,C)\epsilon_{64}\sigma_1$, where $\epsilon_{64}$ is FP64 machine precision and $\sigma_1$ is the largest singular value. Each basis column is signed so that its largest-magnitude entry is positive. All retained columns stay fixed during training.

The normalized objective retains the native feature energy perpendicular to the task-text span. The training loss is support cross-entropy plus $\lambda\lVert A\rVert_F^2$.

## Shared endpoint optimization and selection

P0, ProLIP, LoRA, SVD-E, Comp-E and R0-R2 use 300 FP32 Adam updates, epsilon $10^{-4}$ and a cosine schedule. Each update uses the full support set with one cached view per image, cycling through the 32 views.

| Learning rate | Displacement-penalty candidates |
| --- | --- |
| 0.001 | $1/K$, 0, 0.01, 0.1, 1, 10, 100 |
| 0.0001 | $1/K$ |
| 0.01 | $1/K$ |

Here $K$ is the support count per class. Final-step validation correct count selects one of the nine candidates. Ties prefer (0.001, $1/K$), then a smaller learning rate, then a larger penalty. Each selected model is fixed before test evaluation. Reported test accuracies are the mean and sample standard deviation over support seeds 1/2/3.

## Full and learned-factor endpoint comparisons

ProLIP learns a full increment at the same interface as P0, with the squared-displacement penalty $\lambda\lVert\Delta W\rVert_F^2$.

LoRA learns $\Delta W=BC$, with $B\in\mathbb{R}^{d\times k}$ and $C\in\mathbb{R}^{k\times e}$, scale one and dropout zero. Both factors are trained. $B$ starts at zero; $C$ is uniform on $[-1/\sqrt e,1/\sqrt e]$ using the support seed. The penalty is $\lambda\lVert BC\rVert_F^2$. Rank uses $k=r$, and Param. uses the integer rank whose optimized scalar count is closest to P0's $dr$, with ties favoring smaller rank. These are the endpoint LoRA configurations reported in the paper.

## Spectral endpoint comparisons

SVD-E implements singular-value tuning, and Comp-E implements adaptation within complementary singular spaces. Both use the CLIP visual projector, with dimensions $(d_c,e_c)=(768,512)$, or the SigLIP 2 pooling-head FC2, with dimensions $(3072,768)$. The SigLIP 2 FC2 bias and residual stay fixed. Its comparison with P0 includes the choice of adaptation location: FC2 for the spectral methods and the final output residual for P0.

For a thin SVD $W_0=U\Sigma V^\top$, SVD-E learns the zero-initialized vector $\delta s$ in $\Delta W=U\operatorname{diag}(\delta s)V^\top$. Its penalty is $\lambda\lVert\delta s\rVert_2^2$, and it has 512/768 trainable scalars on CLIP/SigLIP 2.

Comp-E uses complete rectangular left and right singular bases, removes the leading $p=16$ directions from both, and keeps the complementary bases $U_c,V_c$, including rectangular nullspaces. Its update is

$$\Delta W=U_cBA V_c^\top,\quad B\in\mathbb{R}^{(d_c-p)\times k},\quad A\in\mathbb{R}^{k\times(e_c-p)}.$$

$B$ starts at zero and $A$ is uniform on $[-1/\sqrt{e_c-p},1/\sqrt{e_c-p}]$, seeded by the support seed. Both factors are trained with penalty $\lambda\lVert BA\rVert_F^2$. The count is $k(d_c+e_c-32)$. Comp-E reports $k=2$, Rank and Param.; the README gives each rank, scalar count and remaining budget difference. On SigLIP 2 EuroSAT, $k=2$ and Param. are the same configuration.

## Frozen-feature baselines

LP Base and LP++ use the first cached, normalized view of each support image. LP fits a linear classifier and unregularized bias using FP64 L-BFGS. Validation selects $c_{\rm LP}$ from $\{10^{-6},10^{-4},10^{-2},0.316,1,10^2,10^4,10^6\}$, with ties favoring smaller $c_{\rm LP}$.

LP++ uses FP32 SGD with momentum 0.9 for 300 epochs and its data-derived learning rates and initialization. Classifier weights start from summed class visual features, the bias is randomly initialized, and the text coefficients share the data-derived initial value. Text coefficients are updated every ten epochs using fresh gradients at the updated classifier. Validation selects the epoch, with ties favoring the later epoch.

ProKeR uses ten normalized views per support image and FP64 kernel solves. Its grids contain 2,000 candidates for DTD and EuroSAT, 800 for Pets, and 200 for SUN397. Ties retain the first candidate in beta-major, lambda-minor order. The SUN397 16-shot fits exceed the evaluated capacity limit and appear as dashes in the accuracy table.

## Random-coordinate controls

For the same-rank random orthonormal basis $U$, let $K_U=U^\top Q$. In the two tasks used for these controls, $K_U$ is invertible. Writing $G=A_UK_U$ gives the centered unnormalized score correction $hA_UU^\top T_\Delta=hGR$.

R0 trains $A_U$ with penalty $\lVert A_U\rVert_F^2$. R1 trains $A_U$ with the aligned penalty $\lVert G\rVert_F^2$. R2 directly trains $G$ with this penalty and solves for $A_U$. The random basis uses seed 8675309. All controls retain the native score component perpendicular to $U$ and use their actual dynamic norm. Main Table 2 uses the shared nine-candidate validation selection for all four methods, including P0.

The fixed-denominator numerical check verifies agreement of R2 and P0 logits, loss and gradients in FP64. FP32 solve relative residuals are below $2.1\times10^{-6}$ on the two task bases. These are algebraic checks performed without test labels or model fitting.
