
# Diary


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
POISON_RATIO — fraction of dataset poisoned, 1.0 = all, 0.5 = half. This directly maps to your research question about subset poisoning
WATERMARK — the fixed bit string embedded into every poisoned image, acts as your hidden signal
SEED — controls which images get poisoned, keeping it fixed makes experiments reproducible

as expected first results didnt really do too much, after all DWT-DCT is not rally for poisoning models, rather just for hiding a watermark

scoring experiment: DWT-DCT_a5.0_p1.0_
  n=100  content=27.865  style=25.711  combined=26.788

scoring clean model: van-gogh-lora-e3_r64_a32_d0.05_lr5e-05_ga4_attn+ff+conv
  n=100  content=27.789  style=26.745  combined=27.267

on the other hand this is kindof an example of scores we will look for, when we see this kind of scores, we know the watermark does not really change the model outcome


Compare my van gogh with other models:

https://deepai.org/machine-learning-model/text2img






Future Research

- better base model, but I am not sure if it will actually help with something, although it differently won't hurt, but ultimately the research is a place where we can't know what results will be so it is definitely worth exploring