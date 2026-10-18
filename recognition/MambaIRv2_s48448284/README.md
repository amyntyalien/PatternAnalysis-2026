# Project 2.5 - Image Colourisation: State Space Restoration

## Feasability Check-Off
### 1. User Need, Scope & Acceptance Criteria
The intended user, prototype purpose, and 3–5 testable acceptance criteria addressing the domain’s open dilemma, including an operational/resource constraint (e.g., peak VRAM, runtime).

**Intended User**

**Prototype purpose**

**Acceptance criteria**
    - A
    - B
    - C

### 2. Model Choice & Course Concepts
The proposed dataset configuration, model architecture, chosen difficulty band, and how course concepts and literature support the choices.
- **Proposed Dataset Configuration**: use LAB colour space representation
- **Model Architecture**
- **Chosen Difficulty**: Hard
- **How course concepts and literature support these choices**

Lab colourspace? what is that?
https://en.wikipedia.org/wiki/CIELAB_color_space
https://programmingdesignsystems.com/color/perceptually-uniform-color-spaces/
https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_rgb_to_gray.html

For our first model we will use CNN "pixel-wise regression losses (MSE/L1) produce drab results"

### 3. Preliminary Feasability Evidence
An initial data audit, baseline smoke-test experiment, or
measured resource estimate. A fully trained final model is not required.

### 4. Risks, Budget & Fallback
Key technical risks, computing budget, the next planned experiment,
and a workable fallback

## Dataset
- for training we use BSD500

## Metrics
- Metrics are 

| Metric | Ideal Value | Meaning | Formula |
|--------|-------------|---------|---------|
| PSNR
| SSIM
| LPIPS
| Colorfulness

## Loss Function

https://arxiv.org/pdf/2204.02980
- This paper uses L1, L2, LPIPS, WGAN + L2 and WGAN+LPIPS
- Equations for each:

| Metric | Formula | Description |

- Results (what is MAE and FID?)
- ranked best to worst in each category

| Loss Function | MAE | MSE | PSNR | SSIM | LPIPS | FID |
|---------------|-----|-----|------|------|-------|-----|
| L1            | 3   | 5   | 5    | 1    | 3     | 5   |
| L2            | 5   | 4   | 4    | 2    | 5     | 4   |
| LPIPS         | 1   | 1   | 1    | 5    | 1     | 2   |
| WGAN+L2       | 4   | 3   | 3    | 3    | 4     | 3   |
| WGAN+LPIPS    | 2   | 2   | 2    | 4    | 2     | 1   |

- for lips we need to convert to rgb to calculate
- LPIPS performs the best though
- The paper concludes by stating "LPIPS produces slightly better results" this is a slight difference so we could try both LPIPS and somethign that trains faster
    - How about L1 plus saturation penalty

## Baseline Model
- A basic CNN
- Model architecture: 
``` TODO ```

| Hyperparameter  | Value | Meaning/Reasoning |
|-----------------|------:|-------------------|
| number of epochs|       |   |
| learning rate   |       |   |
| batch size      |       |   |
| criterion       |       |   |
|-----------------|-------|-------------------|
| kernel_size     |       |   |
| stride          |       |   |
| padding         |       |   |
| channel sizes   |       |   |

- discuss hyperparameter choices

- results

| Metric | Result |
|--------|--------|
|PSNR    | 22.6826|
|SSIM    | 0.9131 |
|LPIPS   | 0.2035 |
|Colourfulness | 20.46 |

- discuss results

## MambaIRv2
- There is no colourisation model
- it also works with rgb not lab
- and expects 3 channel in &  3 channel out. We want 1 in (L) and 2 out (ab)
- So we need to alter the mamba ir model
- We could: modify the input / output layers only and keep the rest of the model pretrained
- We also could go through all the layers trained.

https://www.youtube.com/watch?v=N6Piou4oYx8
https://www.youtube.com/watch?v=BDTVVlUU1Ck


## AI Usage
- Claude generated my checklist to provide step by step plan
- GPT Luna helped with getting the project working with correct versions, using WSL for Ubuntu and other initial environment set up (the mamba install kept failing)