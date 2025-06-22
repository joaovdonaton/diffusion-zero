# diffusion-zero
Implementing score matching diffusion using SDEs in PyTorch <br>

What this repo has so far:
- Full u-net implementation to serve as our score network 
- Working Variance Preserving score matching diffusion training for MNIST Digits
- Inference using euler method using the trained score network to generate new samples
- `demos.ipynb`, contains some demonstrations and experiments related to what's going on with the model. 

## Resources
- https://diffusion.csail.mit.edu/docs/lecture-notes.pdf
- https://www.youtube.com/watch?v=m0OTso2Dc2U
- Paper: [High-Resolution Image Synthesis with Latent Diffusion Models
](https://arxiv.org/pdf/2112.10752) 
- Paper: [CLASSIFIER-FREE DIFFUSION GUIDANCE](https://arxiv.org/pdf/2207.12598)
- Paper: [SCORE-BASED GENERATIVE MODELING THROUGH STOCHASTIC DIFFERENTIAL EQUATIONS](https://arxiv.org/pdf/2011.13456)