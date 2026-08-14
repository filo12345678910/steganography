
# Diary

THINK ABOUT WHERE TO ADRESS BY "I" OR WE OR SOMETHING
Data preparation:

https://www.kaggle.com/datasets/ipythonx/van-gogh-paintings/data
i now have center-crop + resize to 512×512 (i have a script for this)

Firts base model:

A LoRA adapter was trained on top of a frozen Stable Diffusion v1.5 base model, using a dataset of Van Gogh paintings. The adapter modifies the attention layers of the UNet to shift the generative style toward Van Gogh characteristics. The base model weights remain entirely unchanged.

As a baseline we will prompt a model to produce a cat, using just prompt "cat". I selected it as it is a very recognisable animal with a lot of distinct characteristics that is varied enough to produce hopefully creative and different results.

i produces a first model but it didnt really produce van-goh style images just normal cats (picture)

now i start to work on makind the model retain much of van-goh aesthetic

first r16-epoch20

model didnt really have a difference so i increased r to 64

r64-epoch20

we kinda start to see on some images the difference but it is still not what we are looking for

now i decided to change prompt in training, for now i was using "painting in the style of van gogh" in the pipe.tokenizer in lora

Huge success, exacly what i was looking for, it turned out model should not have nad an information of what is it using, logically it makes sense as i want all images to be influenced by the training set. This initial experiment of removing the prompt was run on only 3 epochs, so now we try with e = 20.

Interestingly with much more epochs, e=20 proved less effective than e=3, much more images became black and white and in some images the features of a cat are unrecogmizable.

With this much experimentsing I decided to train multiple models with some hyper-parameter search.

At this point I decided to look for ways to standardize this search, to have the parameters be searched through while looking at images wether they produce van-gogh like images and cat-like images.

First i focused on the hyperparameters:

    epochs_list=[1, 5, 10],
    rank_list=[8, 16, 32, 64],
    lora_alpha_list=[32, 128],
    learning_rate_list=[1e-4, 5e-5, 1e-5],
    target_modules_list=[
        ["to_q", "to_k", "to_v", "to_out.0"],
        ["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"],
    ]

made the models and generated 20 images from all of them.


Hessel et al. (2021) showed that CLIP, pretrained on 400M image-text pairs, can be used for robust reference-free evaluation by measuring cosine similarity between image and text embeddings in a shared space — and that this correlates highly with human judgement.  https://arxiv.org/abs/2104.08718

Your composite score would be:

Content score — CLIP similarity between the image and "a cat"
Style score — CLIP similarity between the image and "a painting by Van Gogh with swirling brushstrokes and bold colors"
Combined score — average of both, so a model is only rewarded if it produces something that is both cat-like AND Van Gogh-like

content_prompt = "a cat"
style_prompt = "a painting by Van Gogh with swirling brushstrokes and bold colors"


e 10 not really worth showing all results almost identical in ranges 22.4-22.7
e 1 similarly, also low scores in also similar ranges, but even lower on average (make numbers)

only starting from e5 the scores make sense, as the highest scores from both e1 and e10 are lower than lowest score feom e5.
frequest scores of 23 or 24, even some 26 but the highest a model has achieved was , "content": 28.521, "style": 26.942, "combined": 27.731, "n_images": 20 and the model was "van-gogh-lora-e5_r64_a32_d0.05_lr5e-05_ga4_attn+ff"

the parameters optimised seem good as in the top ~10 models they are all consistently present, especially r=64 and e=5. Also interestingly almost all a=32 outperformed a=128. It suggests kinda similar thing to lower e that model does not need to be heavily trained. As for lr it seemed that 5e-5 was the best but with not a lot of variance so we will keep it.  One last thing we will do is, since the experiment shown that its not nessecary good to have more iterations try some of the most prevelent meta-parameters on e=3. This would greatly reduce re-train time if succesful and possibly produce even higher score.

it turns out the model "van-gogh-lora-e3_r64_a32_d0.05_lr5e-05_ga4_attn+ff+conv" has a score of 27.047, wich is lower than the best model but due to much lower epoch number will be used for further training. for now this is the model i will be working on.

now we start with actuall watermark embeding and poisoning attempts, at fitst DWT-DCT caught my eye.

we have a few parameters, 
ALPHA — embedding strength, higher means stronger poisoning but more visible. Start at 5.0, experiment with 1.0, 5.0, 10.0, 25.0
POISON_RATIO — fraction of dataset poisoned, 1.0 = all, 0.5 = half.
SEED — controls which images get poisoned, keeping it fixed makes experiments reproducible
important thing is that i decided to omit embedding of the actuall number or something as a watermark, the goal of this thesis is to deteriorate the model not have a recoverable/proveable string of digits

as expected first results didnt really do too much, after all DWT-DCT is not rally for poisoning models, rather just for hiding a watermark, it is not at all designed for the goal fo the thesis so it is a good benchmark

from now we will also train both models with the same seed and also produce images from them with the same seed

we dont really see any significant poisoning, so on the other hand this is kind of an example of scores we will look for, when we see this kind of scores, we know the watermark does not really change the model outcome



next as a benchmark i will input just a red square onto images, we will see wether it propagates the images, in a way this will be the other end of the spectrum

indeed after training all images have this square, some have slightly different color or the square is larger/smaller but it is always present

i have also added the "empty" watermark, it does not actually produce any watermark and is there just for sanity check/seeing how does the score work/benchamrk

now lets talk about the metrics used

i have split them into 2 parts, we generally measure 2 things, Watermark Visibility (on the original image we want the watermark to be as much invisible as possible) and also Model Deterioration, on the other hand we want on the model trained with poisoned dataset for the output to be as deteriorated as possible

Group 1 — Watermark Visibility (it uses the whole dataset and compares it to the whole dataset after watermark is applied)

PSNR (Peak Signal-to-Noise Ratio) — measures how much the watermarked training image differs from the original in decibels. Above 40 dB is considered imperceptible to humans, below 30 dB is visibly different. Higher is less visible.

SSIM (Structural Similarity Index) — measures whether the watermarked image preserves the structural patterns, textures, and luminance of the original. Ranges from 0 to 1 where 1 means pixel-identical. Higher is less visible.

Group 2 — Model Deterioration

CLIP Combined — it combines both (CLIP Content — how much the generated image resembles the generation prompt ("a cat") according to CLIP's shared image-text embedding space. Higher means more on-prompt.

CLIP Style — how much the generated image resembles the style prompt ("a painting by Van Gogh...") according to CLIP. Higher means more Van Gogh-like.) simple average of content and style scores. Higher means the image is both on-prompt and on-style simultaneously.

Brightness — average pixel intensity across the generated image. Not a quality metric on its own but useful for detecting if the model is generating unusually dark or washed-out images.

Sharpness — variance of the Laplacian filter applied to the image. Measures how much fine edge detail is present. Higher means sharper, lower means blurrier or more painterly.

(the above metrics are more of a sanity metrics and stuff, we willl focus on the 2 at the bottiom)

SSIM vs Clean — structural similarity between the experiment's seeded outputs and the clean baseline's seeded outputs at identical seeds. Higher means the poisoned model produces images more similar to the clean model.

CLIP Drift vs Clean — cosine distance between CLIP image embeddings of the experiment's seeded outputs and the clean baseline's seeded outputs. Measures how much the model's outputs have drifted perceptually from the clean baseline. Lower means less drift.



scores are:

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
    avg SSIM      : 0.5167  (higher = more similar to clean)
    avg CLIP drift: 0.1537  (lower = more similar to clean)

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
    avg SSIM      : 0.5218  (higher = more similar to clean)
    avg CLIP drift: 0.1557  (lower = more similar to clean)

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
    avg SSIM      : 0.4959  (higher = more similar to clean)
    avg CLIP drift: 0.2111  (lower = more similar to clean)

scores saved to C:\Github\steganography\experiments\comparison_scores.json

==========================================================================================
SUMMARY TABLE
==========================================================================================
Experiment           |     --- GROUP 1 ---     |     --- GROUP 2 ---    
                     |  PSNR (dB)       SSIM | CLIP drift SSIM vs clean
------------------------------------------------------------------------------------------
DWT-DCT              |     32.008     0.9257 |     0.1537        0.5167
empty                |      32.04      0.926 |     0.1557        0.5218
square               |     25.591     0.9153 |     0.2111        0.4959
==========================================================================================
GROUP 1 — how much the watermark changed the training images (higher PSNR/SSIM = less visible)
GROUP 2 — how much the poisoned model drifted from clean (higher CLIP drift = more deteriorated, lower SSIM = more deteriorated)


from the result of the images we see that the square is visible as the watermark and also visible as the deterioration, in a way this is a perfect score for deteriorating the image, as there is a visible change there but a terrible score for watermark visibility, as it is clearly still present

the 2 other algoritms generally perfoem the same, empty does not actually do anything so in a way we are aiming for this score from the watermark visibility byt terrible deterioration, if we somehow found scores that are forst score from empty and second from square we are perfect

last observation is that DWT-DCT has the same scores as the empty, this is because this is not a poisoning watermark, so it does not really differ from the 



To recap how the experiment performs:

PS C:\Github\steganography> & C:/Users/filo1/AppData/Local/Microsoft/WindowsApps/python3.11.exe c:/Github/steganography/src/experiments_scripts/experiment.py
======================================================================
EXPERIMENT: square_a5.0_p1.0__e3_r64_la32_lr5e-05_ga4_attn+ff+conv_seed42
======================================================================
watermark algorithm  : square
deterministic        : True (seed=42)
watermarked data dir : C:\Github\steganography\data\square_a5.0_p1.0
model output dir     : C:\Github\steganography\experiments\square_a5.0_p1.0__e3_r64_la32_lr5e-05_ga4_attn+ff+conv_seed42\model
images output dir    : C:\Github\steganography\experiments\square_a5.0_p1.0__e3_r64_la32_lr5e-05_ga4_attn+ff+conv_seed42\images

======================================================================
STEP 1 — WATERMARKING DATASET
======================================================================
total images   : 559
to be poisoned : 559 (100%)
left clean     : 0

poisoned 10/559 ...
poisoned 20/559 ...
....

done — poisoned: 559, clean: 0
output: C:\Github\steganography\data\square_a5.0_p1.0

======================================================================
STEP 2 — TRAINING LORA
======================================================================
Loading weights: 100%|████████████| 196/196 [00:00<00:00, 1651.06it/s, Materializing param=text_model.final_layer_norm.weight]
Loading weights: 100%|██████████████████████| 396/396 [00:00<00:00, 1226.66it/s, Materializing param=visual_projection.weight]
Loading pipeline components...: 100%|███████████████████████████████████████████████████████████| 7/7 [00:01<00:00,  4.68it/s]
epoch 0 step 50 loss 0.4359
epoch 0 step 100 loss 0.0868
epoch 0 done
epoch 1 step 150 loss 0.3925
epoch 1 step 200 loss 0.0634
epoch 1 step 250 loss 0.3476
epoch 1 done
epoch 2 step 300 loss 0.0474
epoch 2 step 350 loss 0.7432
epoch 2 step 400 loss 0.4968
epoch 2 done

======================================================================
STEP 3 — GENERATING IMAGES
======================================================================
Loading weights: 100%|█████████████| 196/196 [00:00<00:00, 472.90it/s, Materializing param=text_model.final_layer_norm.weight]
Loading weights: 100%|███████████████████████| 396/396 [00:00<00:00, 447.55it/s, Materializing param=visual_projection.weight]
Loading pipeline components...: 100%|███████████████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.34it/s]
C:\Users\filo1\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\diffusers\loaders\unet.py:214: FutureWarning: `load_attn_procs` is deprecated and will be removed in version 0.40.0. Using the `load_attn_procs()` method has been deprecated and will be removed in a future version. Please use `load_lora_adapter()`.
  deprecate("load_attn_procs", "0.40.0", deprecation_message)
100%|█████████████████████████████████████████████████████████████████████████████████████████| 30/30 [00:07<00:00,  4.22it/s]
saved C:\Github\steganography\experiments\square_a5.0_p1.0__e3_r64_la32_lr5e-05_ga4_attn+ff+conv_seed42\images\0.png

....

======================================================================
EXPERIMENT COMPLETE — square_a5.0_p1.0__e3_r64_la32_lr5e-05_ga4_attn+ff+conv_seed42
======================================================================



and later we run the scoring script to see the previoius results, now we know what we are looking for so we will try to use some watermarks that ACTUALLY poison the model




Compare my van gogh with other models:

https://deepai.org/machine-learning-model/text2img






Future Research

- better base model, but I am not sure if it will actually help with something, although it differently won't hurt, but ultimately the research is a place where we can't know what results will be so it is definitely worth exploring