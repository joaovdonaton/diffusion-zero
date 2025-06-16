# diffusion-zero
Implementing score matching diffusion using SDEs in PyTorch <br>
Most of the definitions and terminology used here are from [Peter E. Holderrieth's](https://www.peterholderrieth.com/) notes from the [MIT 6.S184](https://diffusion.csail.mit.edu/) lectures

The current plan is to implement diffusion score matching for sampling from CIFAR10 distribution, using classifier-free guidance (so we can experiment with different guidance strength values). Then move onto to more complex datasets and try to improve performance using latent diffusion or other more modern techniques

## Resources
- https://diffusion.csail.mit.edu/docs/lecture-notes.pdf
- https://www.youtube.com/watch?v=m0OTso2Dc2U
- Paper: [High-Resolution Image Synthesis with Latent Diffusion Models
](https://arxiv.org/pdf/2112.10752) 
- Paper: [CLASSIFIER-FREE DIFFUSION GUIDANCE](https://arxiv.org/pdf/2207.12598)
- Paper: [SCORE-BASED GENERATIVE MODELING THROUGH STOCHASTIC DIFFERENTIAL EQUATIONS](https://arxiv.org/pdf/2011.13456)