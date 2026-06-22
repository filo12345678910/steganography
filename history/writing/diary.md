
# Diary


Data preparation:

https://www.kaggle.com/datasets/ipythonx/van-gogh-paintings/data
i now have center-crop + resize to 512×512 (i have a script for this)

Firts base model:

A LoRA adapter was trained on top of a frozen Stable Diffusion v1.5 base model, using a dataset of Van Gogh paintings. The adapter modifies the attention layers of the UNet to shift the generative style toward Van Gogh characteristics. The base model weights remain entirely unchanged.


i produces a first model but it didnt really produce van-goh style images just normal cats (picture)

now i start to work on makind the model retain much of van-goh aesthetic

first r16-epoch20

model didnt really have a difference so i increased r to 64

r64-epoch20

we kinda start to see on some images the difference but it is still not what we are looking for

now i decided to change prompt in training, for now i was using "painting in the style of van gogh" in the pipe.tokenizer in lora

Huge success, exacly what i was looking for


Compare my van gogh with other models:

https://deepai.org/machine-learning-model/text2img
