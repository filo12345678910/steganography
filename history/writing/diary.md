
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

Compare my van gogh with other models:

https://deepai.org/machine-learning-model/text2img
