requirements


1 — Adversarial perturbation against the VAE (most promising)
Compute a gradient-based perturbation by backpropagating through the VAE encoder itself. The goal is to find pixel changes that maximally shift the latent representation while remaining imperceptible. This is exactly what Glaze and Mist do, and it directly addresses the VAE bottleneck problem. Expected outcome: strong Group 2 scores with reasonable Group 1 scores. This is the most academically interesting result you could produce.

2 — Low-frequency colour shift
Add a small fixed offset to one or more colour channels uniformly across the entire image — for example shift every pixel's red channel by +8. This operates at a frequency low enough to survive VAE compression since the VAE preserves global colour information. Expected outcome: moderate Group 2 effect, Group 1 scores slightly below empty baseline but potentially still above 35 dB.

3 — Checkerboard pattern at low frequency
Overlay a very faint low-frequency checkerboard of alternating +k/-k pixel shifts with large cell size (e.g. 32×32 or 64×64 pixels). Large cells mean low frequency content which survives VAE compression better than DCT coefficients. Expected outcome: somewhere between DWT-DCT and square depending on intensity.

4 — Texture overlay
Add a faint fixed texture — for example a very subtle noise pattern or fixed artistic texture — blended at low opacity across the entire image. If the texture has enough low-frequency energy it should survive encoding.

5 — Targeted latent space poisoning
Skip pixel space entirely and embed the perturbation directly in the latent space, then decode back to pixels. This guarantees VAE survival since you are working in the space the UNet actually trains on. The decoded perturbation will be visible but potentially subtle.

My recommendation for your thesis would be to implement options 1 and 2. Option 1 is the strongest and most novel result, directly addressing the failure of DWT-DCT with a theoretically motivated solution. Option 2 is trivially simple to implement and gives you a second data point on the spectrum between DWT-DCT and the red square. Together they would make Chapter 6 much stronger.

Want me to implement option 1 or 2 first?