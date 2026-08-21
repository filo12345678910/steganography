# Diary

Now we are starting "real" methods that are supposed to work, first with Low-frequency colour shift

stats:
CHANNEL — 0 is red, 1 is green, 2 is blue. Red is the most visually impactful, blue is the least noticeable to humans
SHIFT — how much to shift the channel. 20 will be subtle but measurable, 40 will be clearly visible as a colour cast. Start at 20 and see the PSNR — if it is above 35 dB it is effectively invisible

shift is to red and it actually achieves great scores, the image is indeed more red, but so is the input, but this is only visible when comparing with original. Also it is not nessecary great idea as the paintings themselver are already veery red so change is not that noticable, SSIM stays pretty much the same. but db visibly lower as the image when comparing is indeed different color but still preserves stuff.
(make a big table with all results of this particular experiment)
shift                |     25.802     0.9145 |     0.1791        0.4559
(shift_red.png)

now we will try the shift to green

shift                |     25.646     0.9122 |     0.1914        0.3614

green worked much better, still the input image is visibly more green. The output this time seems to deteriorate a bit.

(shift_green.png)

When i shifted to green images started to deteriorate a little, wich can be seen by the 0.3614, it seems like the images are not even really "bad" just very different from the original, it seems that messing with the red chanel didnt change the structure of the pictures that much but meesing with the green indeed did. We still see cats but deffinetly different ones that i have seen before?

is this explanation valid if not provide different one or add context from literature


lastly to top it off i decided to also see blue

shift                |     25.671     0.9042 |     0.1677        0.5337
(shift_blue.png)

suprisingly blue performed muuuch worse than the others, all of them had trash PSNR but at least grean had a very good deterioration, other 2 seem useless. Also it seems to be pretty similar to red just with even less color change, shapes seem also similar


This is a meaningful finding for your thesis — it demonstrates that the effectiveness of a colour-shift watermark depends not just on the magnitude of the shift but on how far the shift moves the images away from the natural colour distribution of the specific training dataset.

An unexpected finding from the colour shift experiments is that shifting the green channel produced stronger deterioration than either red or blue. The explanation follows the same logic as the red versus blue comparison, but taken one step further.

Red blends naturally into Van Gogh's warm palette and produces the weakest signal. Blue is more foreign than red, but Van Gogh's work does contain significant blue tones — his night scenes, skies, and shadows frequently use deep blues and indigos, meaning the VAE has encountered blue-shifted images within the natural distribution of the dataset and can encode them without significant disruption.

Green, however, is the channel Van Gogh uses in the most constrained and context-specific way. His greens appear almost exclusively in foliage and grass, and are nearly always mixed with yellow or brown rather than appearing as saturated pure greens. A uniform shift of the green channel across every pixel — including skies, skin tones, architecture, and backgrounds — pushes those regions into a colour space that is genuinely unusual for this dataset. The VAE has the least prior experience encoding uniformly green-shifted Van Gogh paintings, so the latent representation is maximally disrupted relative to the other channels.

There is also a perceptual factor worth noting: the human eye is more sensitive to green than to red or blue, meaning a shift of the same pixel magnitude produces a larger perceived colour change. This works against invisibility in Group 1 — which is reflected in the slightly lower PSNR scores — but it also means the latent shift is larger, which is precisely why the poisoning effect is stronger in Group 2. This represents an inherent trade-off: the more perceptually foreign the colour shift, the more effective the poisoning, but also the more visible the watermark.


now since we know shifting green up produces great results and red not so much we will try to shift green up and red down

shift_rg             |     23.357     0.8947 |     0.1987        0.3467

it did indeed improve the results slightly but at the cost of having much worse group 1 scores, so not worth it, with this i decided its enough, the green chanel 1 is the best, this method could be improved but does not really have any super promissing results




Next method according to the literature the Adversarial perturbation against the VAE is the most promissing, but also very time costly. Immidiet issue, single image to be poisoned takes about 30 seconds, the whole experiment takes almost 5 hours.

but a HUGE SUCCESS! After poisoning the entire dataset there are very clear and visible squares/checkerboard pattern over all images.

adversarial          |     29.427     0.8776 |     0.2207         0.441
(adv.png)
interestingly this is the first time SSIM falls down, PSNR also does but not too much almost still beeing 30. 

upon close inspection you indeed can see the watermark, it kinda looks like pictures had worse quality, but you deffinetly can't see the checkered pattern. Also on some pictures that are already not the best quality, or more surreal or just with more stuff it is much harder to see the differences. on the other hand, on the images with hings like clear sky unfortunatly watermark is quite visible.

poisoning though verry effective with the highest scores in both metrics yet

now time for testing with the % of poisoned data, we will try 75%, 50%, 25% and 10%


adversarial_a5.0_p0.1 |     29.652     0.8922 |     0.1393        0.5503
adversarial_a5.0_p0.25 |      29.51     0.8819 |     0.1765        0.5083
adversarial_a5.0_p0.5 |     29.429     0.8792 |     0.3107        0.4279
adversarial_a5.0_p0.75 |     29.432     0.8786 |     0.2322        0.4553
adversarial          |     29.427     0.8776 |     0.2207         0.441

(adv_0.1.png and so on)

p0.5 produced stronger deterioration than p0.75 and p1.0, which is counterintuitive. The explanation is likely that at 50% poisoning the model is being pulled in two competing directions simultaneously — half the images push the latent toward the adversarial target, and half pull it back toward the natural distribution. This tension during training may actually cause more disruption to the adapter weights than a fully poisoned dataset where the signal is consistent and the model can adapt to it more stably. At 100% poisoning the model just learns the adversarial distribution as its new normal, whereas at 50% it is constantly confused between two conflicting signals.

it is kinda weird because it is very visible that 100% and 75% have this square-like pattern and 50% has a slightly different pattern, more swirly-like, but the images themselves are much more deteriorated with figures not resembling a human or a cat. explanation

suprisingly even on the 0.1 the images were poisoned, we could see on most of them some artifacts that should not be there, and on some it still managed to produce an unrecognisable result, proving that the paper was tisht that small amount of poisoned data is needed to destroy the model.

At 10% and 25% the signal is too diluted to cause strong deterioration
At 50% the competing signals between poisoned and clean images create maximum confusion in the adapter weights
At 75% and 100% the model starts adapting to the poisoned distribution as its new normal, reducing the apparent deterioration relative to the clean baseline

also to adress scores from group 1, they might look weird, but once you uderstand that all metrics from group 1 are calculated only on poiosoned images it is clear why they are practically the same


sidenote add somewhere that when i do experiments with like poisoning % of the data we leave calculating of the results in group 1 only on the actually poisoned data


Compare my van gogh with other models:

https://deepai.org/machine-learning-model/text2img






Future Research

- better base model, but I am not sure if it will actually help with something, although it differently won't hurt, but ultimately the research is a place where we can't know what results will be so it is definitely worth exploring

- more prompts than just "cat"

