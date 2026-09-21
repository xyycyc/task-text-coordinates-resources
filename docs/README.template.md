# Task-Text Coordinates for Normalized Few-Shot Vision-Language Adaptation

TextCoord (Task-Text Coordinate Adaptation) uses the fixed class texts to determine a visual adaptation space before fitting the support images. It retains the complete centered class-text span, learns a visual coordinate map in that span, and preserves the native perpendicular feature component in the normalized objective. This repository provides the parameter budgets, implementation, experimental settings and supporting analyses for the paper.

[Parameters](#parameter-budgets) · [TextCoord](#textcoord-implementation) · [Protocol](#experimental-protocol) · [Results](#accuracy-comparisons) · [Analysis](#coordinate-and-normalization-analysis) · [Code](#running-the-code) · [Sources](#sources)

## Parameter budgets

The tables count trainable scalars in the adaptation modules. Frozen encoders, class texts and subspace bases are excluded. Counts apply to both 4-shot and 16-shot settings.

TextCoord trains **35,328 / 6,912 / 27,648 / 304,128** scalars for DTD / EuroSAT / Pets / SUN397 on either encoder. Its parameter ratio to the full ProLIP endpoint is $r/e$. The rank-9 SigLIP 2 EuroSAT configuration reduces 589,824 trainable parameters to 6,912, a **98.8% reduction**.

<!-- TABLE:parameters -->

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

<!-- TABLE:proker_counts -->

SUN397 16-shot ProKeR accuracy is unreported because the fit exceeds the evaluated device capacity. Its coefficient count above follows from the model dimensions.

### Matched ranks and budgets

Rank sets $k=r$. Param. selects the integer rank minimizing the absolute difference from the main-table TextCoord budget $dr$, with ties favoring smaller rank. A rank match and a parameter match define separate comparisons. Signed differences below are comparator count minus TextCoord count.

<!-- TABLE:matched -->

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

<!-- TABLE:accuracy -->

Across the sixteen settings, TextCoord exceeds Comp-E Rank and Param. by 1.19 and 2.84 percentage points on average. On the shared CLIP projector, TextCoord exceeds SVD-E and Comp-E $k=2$ in all eight setting means. The SigLIP 2 spectral comparisons include adaptation location as part of the complete configuration: FC2 for SVD-E/Comp-E and the output residual for TextCoord.

## Coordinate and normalization analysis

### Validation-selected coordinates

The coordinate controls examine regularization and optimization within update families having the same centered unnormalized score expressiveness. For a same-rank random orthonormal basis $U$, set $K_U=U^\top Q$ and $G=A_UK_U$. Then

$$
hA_UU^\top T_\Delta=hGR.
$$

R0 optimizes $A_U$ with penalty $\lVert A_U\rVert_F^2$. R1 optimizes $A_U$ with penalty $\lVert G\rVert_F^2$. R2 directly optimizes $G$ with that penalty and obtains $A_U$ by a linear solve. For each diagnostic task, one random basis generated with seed 8675309 is shared by R0–R2 across both shot counts and all three support seeds. Each model retains the native visual component perpendicular to $U$ and uses its actual dynamic normalization denominator.

<!-- TABLE:selected_coordinates -->

R1 improves upon R0 in the four setting means. R2 improves upon R1 on EuroSAT and decreases accuracy on DTD, indicating a setting-dependent effect of optimizer coordinates. TextCoord has the highest mean in each of these four comparisons. R2 matches TextCoord's centered unnormalized score parameterization and penalty form. Its coupled update perpendicular to the task-text span generally yields a different normalization denominator from TextCoord, which preserves the native perpendicular component.

### Fixed-configuration coordinate diagnostics

These diagnostics fix the learning rate to $10^{-3}$ and $\lambda=1/K$ for all four methods, with 300 updates. This controlled configuration distinguishes them from the validation-selected table above.

<!-- TABLE:fixed_coordinates -->

The fixed-configuration results show the same direction of improvement from R0 to R1 across the four setting means. The R1-to-R2 change again depends on the task. This comparison examines the coordinate choices at a shared learning rate and penalty.

### Normalization and regularization

D uses the current adapted feature norm; F fixes the denominator to $\lVert z_0\rVert_2$ for each image or view. Disp. uses $\lambda\lVert A\rVert_F^2$. CI uses $\lambda\lVert AR\rVert_F^2/\gamma$, where $\gamma=\lVert R\rVert_F^2/r$. All entries use the nine-candidate validation search.

<!-- TABLE:normalization -->

The best normalization/regularization variant varies across these settings. All text-coordinate variants exceed the random-coordinate reference in the four setting means. TextCoord combines normalized prediction with displacement regularization, matching the minimum-displacement interpretation of the method.

### Algebraic checks

With a fixed denominator, R2 and TextCoord agree in centered logits, loss and gradients when written in aligned coordinates. These checks use the actual task bases and FP64 arithmetic; the solve residual is measured in FP32.

<!-- TABLE:numerics -->

The source package includes CPU tests of compact versus explicit probabilities, cross-entropy and gradients, zero-update recovery, fixed-denominator coordinate equivalence, complementary-subspace constraints and the ProKeR linear solve.

## Adaptation-position analysis

This comparison evaluates TextCoord and ProLIP at the SigLIP 2 output and FC2 interfaces on DTD and EuroSAT, with both shot counts, seeds 1/2/3 and validation selection. At FC2, $z=z_0+h\Delta W$ retains the complete native bias and residual in $z_0$.

<!-- TABLE:position -->

Both TextCoord and ProLIP achieve higher mean accuracy at the output interface than at FC2 in all four settings. This supports the output interface used by the main SigLIP 2 configuration. Same-interface ProLIP/LoRA comparisons examine parameterization within that choice.

## Paired accuracy differences

Each entry is TextCoord minus the comparator in percentage points, paired by support seed and summarized as mean ± sample SD of the three differences. All main-table configurations are covered.

<!-- TABLE:paired -->

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
