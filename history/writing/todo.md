requirements




improve model:

Fourth — add more attention layers to target_modules

Currently you only train the attention projections. Adding the feed-forward layers gives the adapter more capacity:
pythontarget_modules=["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"]
Fifth — lower the learning rate slightly

At r=64 with many epochs you risk overfit artifacts rather than clean style. Try lr=5e-5 instead of 1e-4.
The most impactful next step is honestly just more images combined with more epochs.


chack images? curate van-gogh like dataset