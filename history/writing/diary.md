
# Diary


Data preparation:

https://www.kaggle.com/datasets/ipythonx/van-gogh-paintings/data
i now have center-crop + resize to 512×512 (i have a script for this)

Firts base model:

A LoRA adapter was trained on top of a frozen Stable Diffusion v1.5 base model, using a dataset of Van Gogh paintings. The adapter modifies the attention layers of the UNet to shift the generative style toward Van Gogh characteristics. The base model weights remain entirely unchanged.


i produces a first model but it didnt really produce van-goh style images just normal cats (picture)

now i start to work on makind the model retain much of van-goh aesthetic

first r16-epoch20