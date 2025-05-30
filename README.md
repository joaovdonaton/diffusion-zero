# diffusion-zero
Implementing score matching diffusion using SDEs in PyTorch <br>
Most of the definitions and terminology used here are from [Peter E. Holderrieth's](https://www.peterholderrieth.com/) notes from the [MIT 6.S184](https://diffusion.csail.mit.edu/) lectures

The current plan is to implement diffusion score matching for sampling from CIFAR10 distribution, using classifier-free guidance (so we can experiment with different guidance strength values). Then move onto to more complex datasets and try to improve performance using latent diffusion or other more modern techniques

## Todo:
- Look into what the forward process definition and scheduling looks like for SDE in song's paper (apparently the condOT thing is deterministic/flow not the SDE forward, but should work)
    - See VP and VE and scheduling, see relationship with probability path definition from 6.S184
    - Why is one vp and other ve
- Also investigate what the loss function looks like for the udpated forward process
    - Currently we have loss as squared l2 norm of (score_network(x_t) - noise). Which is what is explained in Peter's class, but that appears to be the simplified version and not the proper denoising score matching loss

- Development: validation loss to monitor overfitting, record loss, train script, notebook for plotting results

<br>
note: problem right now is that im just mixing things up. For example the forward process is the condOT and it's more of a determinstic flow matching approach, but then for the loss I'm doing squared l2 norm of score_network - noise, which is the SDE score-matching version so it doesnt align. Reason ti doesnt work like the lab in 6.s184 is because of the loss, it should be something like that formula for the conditional vector field, but we ar edoing score matching, so: we need to change from the condOT approach to whatever score matching does
<br>

## Resources
- https://diffusion.csail.mit.edu/docs/lecture-notes.pdf
- https://www.youtube.com/watch?v=m0OTso2Dc2U
- Paper: [High-Resolution Image Synthesis with Latent Diffusion Models
](https://arxiv.org/pdf/2112.10752) 
- Paper: [CLASSIFIER-FREE DIFFUSION GUIDANCE](https://arxiv.org/pdf/2207.12598)
- Paper: [SCORE-BASED GENERATIVE MODELING THROUGH STOCHASTIC DIFFERENTIAL EQUATIONS](https://arxiv.org/pdf/2011.13456)