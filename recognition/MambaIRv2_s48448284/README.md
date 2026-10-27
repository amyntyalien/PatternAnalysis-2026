# Project 2.5 - Image Colourisation: State Space Restoration

## Feasability Check-Off
### 1. User Need, Scope & Acceptance Criteria

Image colourisation has applications in restoration of historical photos. The intended user of such tools is someone who wants to convert grayscale to colour more easily and quicker than can be done manually by hand. This prototype would demonstrate that a state-space model, such as MambaIRv2, can be adapted for colourisation.

**Acceptance criteria**
Generally, my acceptance criteria is for the MambaIRv2 model to perform better than the basic CNN baseline performed.

| # | Criteria                 | Target |
|---|--------------------------|--------|
| A | Quantitative Metrics     | PSNR > 22; SSIM > 0.91; LPIPS < 0.20; Colorfulness > 20.5 |
| B | Qualitative              | Minimal failure cases, particularly focussed on desaturation and chromatic hallucinations |        
| C | Performance Restrictions | 8 GB VRAM budget; runtime < 5 hr  for fine tune, < 3 hr for frozen on local machine |


### 2. Model Choice & Course Concepts

**Chosen Difficulty**: Hard
I chose this project despite it being hard because image colourisation seems interesting, and the results seem a lot more engaging for me to interact with.

**Dataset**

I decided to use BSD500 (link ref dataset) as my dataset for training and testing. I chose this dataset as it has a wide variety of images, and because it was relatively small but larger than BSD50. This would make it so that runtime should not be too slow, while results are still good. 
This dataset has 500 images in total, with a split of 70% (350) training data, 15% (75) validation data and 15% (75) testing data.
All images in the dataset are resized to 256 x 256 pixels, and transformed into CIE LAB colour space for use with training and testing models.
LAB colour space consists of L (lightness), which serves as grayscale input, a (green-red) and b (blue-yellow) which serve as the predicted output.
CIE LAB colour space is standard for colourisation tasks because LAB is useful for detecting differences in colour seperately from how bright/dark the colour may be. 

ref: https://programmingdesignsystems.com/color/perceptually-uniform-color-spaces/

**Model Architecture**
For the *baseline CNN* model, a basic autoencoder structure was used, [as illustrated bellow]. This simple model can serve as a baseline of minimum competence.
- CNN 
- 1 -> 64 -> 128 -> 256 -> 128 -> 64 -> 2
- Trained with MSE, L1, L1_Sat, and LPIPS

*MambaIRv2 (Light)*: MambaIRv2 is a state-space model for image restoration, particularly for denoising and resizing purposes. The 'light' version is smaller than the base model with less embeded dimensions. The core building block of MambaIRv2 models is the Aggregated Selective State-space Block (ASSB).

ASSB blocks contain:
- Window Multi-Head Self-Attention (Window-MHSA): uses non-overlapping windows to capture local spatial context
- Attentive State Space Module (ASSM): 
    - SGN-unfold: scores tokens by semantic similarity and reorders token sequence so related tokens are adjacent
    - Selective scan: selectively lets the model decide what to remember and what to forget
    - Addaptive State Expansion (ASE) : dynamically expands the state dimensions for tokens that need richer context, and shrink for simpler regions
    - SGN-fold: restore original spatial layout 
reference: mambaIRv2 paper
[add in the flowchart]

Instead of low quality image input and high quality image output, we want to have 1-channel L input and 2-channel ab output. So, we can addapt the model and keep the structure all the same, however change the input and output layer, as well  as upsampling disabled. Pretrained weights from MambaIRv2 Light SR x2 (https://github.com/csguoh/MambaIR/releases/tag/v1.0 
) are loaded with shape-filtered transfer, layeres whose shapes match are loaded, and mismatched layers are reinitialised.

There are two training strategies that we can use on this altered model. We can freeze the pretrained weights, and only train the conv_first and conv_last, or we can train through every layer. These strategies are refered to as "Frozen" and "Finetuned". 
I also seek to compare two loss functions, being L1 + desaturation penalty, and LPIPS (AlexNet) + L1. These are referred to as "l1_sat" and "lpips" respectively. 

### 3. Preliminary Feasability Evidence

**CNN Baseline (100 epochs, L1 loss)**

| Metric | Value |
|--------|------|
| PSNR | 22.66 |
| SSIM | 0.9112 |
| LPIPS | 0.2086 |
| Colorfulness | 20.58 / 40.58 |

This CNN produced structurally correct but desaturated output, confirming the need for a saturation-aware loss ans more powerful feature extractors.

**Frozen l1_sat (as of epoch 5)**
| Metric | Value |
|--------|-------|
| PSNR | 22.14 |
| SSIM | 0.9001|
| Epoch time | 186 s |

Expected run time = 2.6 hrs

**Finetuned l1_sat (as of epoch 5)**
| Metric | Value |
|--------|-------|
| PSNR | 23.27 |
| SSIM | 0.9312|
| Epoch time | 551 s |

Expected run time = 7.7 hrs

Wonderfully, the PSNR and SSIM of this model is already better than the CNN!

**Feasibility**

The expected run time of the finetune model seems to exceed the acceptance criteria, and running these models is estimated to take 20 hours in total on my local machine. This can be easily done over the course of two overnight training runs that can occur while my computer is not in use. 

### 4. Risks, Budget & Fallback


| Risk                              | Mitigation / Fallback             |
|-----------------------------------|-----------------------------------|
| Finetune model runs out of memory | Per-layer gradient checkpointing  |
| Pretrained weights incompatible   | Shape-filtered loading            |
| Training time exceeds expectation | 50 epochs                         |
| MambaIRv2 underperforms           | Use larger dataset, add more epochs, run on Virtual GPU Lab |



### Checked Off?
Suggestions:
- way more vram
- larger dataset



## Metrics
- Metrics are 

| Metric       | Ideal Value | Meaning | Formula |
|--------------|-------------|---------|---------|
| PSNR         |
| SSIM         |
| LPIPS        |
| Colorfulness |

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
- comparisons of L1, L1_sat and LPIPS

- results

| Metric       | MSE    | L1     | L1+sat | LPIPS   |
|--------------|--------|--------|--------|---------|
|PSNR          | 22.6826| 22.6877| 22.6210| 21.8546 |
|SSIM          | 0.9131 | 0.9112 | 0.9096 | 0.8065  |
|LPIPS         | 0.2035 | 0.2086 | 0.2065 | 0.2113  |
|Colourfulness | 20.46  | 20.58  | 22.25  | 22.42   |

- discuss results

## MambaIRv2
pretrained models from:
https://github.com/csguoh/MambaIR/releases/tag/v1.0 

https://arxiv.org/pdf/2411.15269


- hyperparameters
- choices


- results
Frozen
| Metric       | L1+sat | LPIPS   |
|--------------|--------|---------|
|PSNR          | 22.5548| 21.9421 |
|SSIM          | 0.9147 | 0.8992  |
|LPIPS         | 0.1952 | 0.1971  |
|Colourfulness | 17.21  | 23.20   |

Finetuned
| Metric       | L1+sat | LPIPS   |
|--------------|--------|---------|
|PSNR          | 22.9920| 22.9800 |
|SSIM          | 0.9275 | 0.9178  |
|LPIPS         | 0.1874 | 0.1787  |
|Colourfulness | 22.56  | 24.27   |

Failure cases  = (color bleeding across high-contrast edges, desaturation collapses, chromatic hallucinations)."


- these were dissapointing
- so first i decided to give it more data
- fall back plan:
    - Use larger dataset, add more epochs, run on Virtual GPU Lab
- due to run time, i dont think i can add more epochs and run on virtual gpu lab without running out of quota.


fine tuned more

| Metric       | L1+sat | LPIPS   |
|--------------|--------|---------|
|PSNR          | 23.5957| |
|SSIM          | 0.9202 | |
|LPIPS         | 0.1783 | |
|Colourfulness | 20.45  | |

its better?



## AI Usage
- Claude generated my checklist to provide step by step plan
- GPT Luna helped with getting the project working with correct versions, using WSL for Ubuntu and other initial environment set up (the mamba install kept failing)
- Claude assisted with writing code where needed, and ensuring code is up to readability standards