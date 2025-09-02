class: middle, center, title-slide

<br>

# Inverting scientific images with<br>deep generative models

<br>

AI for Science symposium<br>
The Royal Swedish Academy of Sciences<br>
September 4, 2025

.grid[
.kol-1-3[
.width-60.circle[![](figures/faces/gilles.jpg)]
]
.kol-2-3[

<br>
Gilles Louppe<br>
[g.louppe@uliege.be](mailto:g.louppe@uliege.be)

]
]

???

My goal today is to convince you that deep generative models are unlocking new scientific questions that were previously out of reach. 

They are not just fancy tools to generate realistic images, but powerful statistical models that can help us tackle challenging inverse problems in science. 

---

class: middle, black-slide, center
background-image: url(figures/y.png)
background-size: cover

.bold.larger[From a noisy observation $y$...]

---

class: middle, black-slide, center
background-image: url(figures/x.png)
background-size: cover

.bold.larger[... can we recover <br> all plausible images $x$?]

---

class: middle, black-slide, center

.larger[

$$
\begin{aligned}
    \dot{u} & = - u \nabla u + \frac{1}{Re} \nabla^2 u - \frac{1}{\rho} \nabla p + f \\\\
    0 & = \nabla \cdot u
\end{aligned}
$$

]

.bold.larger[

... or model parameters $\theta = \\\{Re, \rho, f\\\}$?

]

---

class: middle

.center.width-80[![](figures/setup.svg)]

## Inverse problems in science

Given noisy observations $y$, estimate either
- the posterior distribution $p(x|y) \propto p(x) p(y|x)$ of latent states $x$, or
- the posterior distribution $p(\theta|y)$ of model parameters $\theta$.

???

Inverse problems are common in science, because we often cannot measure the quantities of interest directly. Instead, we have to guess them from indirect and noisy observations.

For instance, we cannot directly probe the atmosphere of exoplanets, but we can observe their spectrum when they transit in front of their star. From this spectrum, we want to infer the atmospheric parameters (composition, temperature, pressure, etc).

Or we cannot directly observe the full 3d dynamics of oceans, but we can observe their surface from space. From these satellite observations, we want to infer the full 3d state of the ocean.

Why is this hard?
- Models and simulators are expressed in terms of forward processes, which often implement causal mechanistic assumptions. In this form, they can be used to generate synthetic data, but they cannot be inverted easily.
- The latent states $x$ we want to infer are often high-dimensional (e.g., images, time series, 3d fields) and not unique. The problem is ill-posed, there is not a single solution but a whole distribution of plausible solutions.
- The model parameters $\theta$ we want to infer are often low-dimensional, but the likelihood $p(y|\theta)$ is intractable

---

exclude: true
class: middle

.center.width-10[![](figures/icons/summation.png)]

A word on notations...
- In the inverse problem literature, $x$ is often used to denote the latent state, while $y$ is used to denote the observation.
- In simulation-based inference, $x$ is often used to denote the observation and the latent state is not denoted. Inference is done on the parameters $\theta$.
- In the generative modeling literature, $x$ often denotes an observation, while $z$ is used to denote the latent state.

Sorry for the mess, not my fault!

---

class: middle
count: false

.center.width-15[![](figures/icons/exoplanet.png)]

# Part 1: Low-dimensional inverse problems

$p(\theta|x)$, with $\theta \in \mathbb{R}^d$, $d = O(10)$.

<br>

---

class: middle, black-slide

.avatars[![](figures/faces/malavika.jpg)![](figures/faces/absil.jpg)]

## Exoplanet atmosphere characterization 

.center.width-80[![](./figures/exoplanet-probe.jpg)]

.center[What are the atmospheres of exoplanets made of?<br> How do they form and evolve? Do they host life?]

.footnote[Credits: [NSA/JPL-Caltech](https://www.nasa.gov/topics/universe/features/exoplanet20100203-b.html), 2010.]

???

The first science question I want to discuss is the characterization of exoplanet atmospheres.

When an exoplanet transits in front of its star, a tiny fraction of the starlight passes through the planet's atmosphere before reaching us. By analyzing the spectrum of this light, we can infer the composition and properties of the atmosphere.

This is interesting because the atmosphere holds clues about the planet's formation, evolution, and potential habitability. 

---

class: middle, black-slide

.center.width-50[![](./figures/WISE1738.jpg)]

.center[WISE 1738+2732, a brown dwarf 25 light-years away.]

???

The object we studied is WISE 1738+2732, a brown dwarf located about 25 light-years away. It is one of the coldest known brown dwarfs, with an effective temperature of about 350K.

It was observed with the JWST telescope, which provided us with a high-quality spectrum of its atmosphere.

This brown dwarf is interesting because its temperature is similar to that of some exoplanets, making it a good proxy for studying exoplanet atmospheres. It is also interesting because it is cool enough to have complex molecules like water vapor, methane, and ammonia in its atmosphere. 

---

class: middle

.avatars[![](figures/faces/malavika.jpg)![](figures/faces/francois.jpg)![](figures/faces/absil.jpg)]

.center.width-90[![](./figures/fig1.png)]

Using .bold[Neural Posterior Estimation] (NPE), we approximate the posterior distribution $p(\theta|x)$ of atmospheric parameters $\theta$ with a .bold[normalizing flow] trained on pairs $(\theta, x)$ simulated from a physical model of exoplanet atmospheres.

.footnote[Credits: [Vasist et al](https://arxiv.org/abs/2301.06575), 2023 (arXiv:2301.06575).]

---

class: middle

.avatars[![](figures/faces/malavika.jpg)![](figures/faces/absil.jpg)]

.grid[
.kol-3-5[<br><br>.width-100[![](./figures/wise-spectra.png)]]
.kol-2-5[.width-100[![](./figures/wise-posterior.png)]]
]

.center[Panchromatic characterization of WISE 1738+2732 using JWST/MIRI.]

.footnote[Credits: [Vasist et al](https://arxiv.org/abs/2507.12264), 2025 (arXiv:2507.12264).]

???

While this posterior plot can look intimidating, it is actually telling us a lot about the physics and chemistry of this world.
- The atmosphere contains water vapor, methane, ammonia, carbon monoxide, and carbon dioxide
- The detection of carbon monoxide and carbon dioxide is a surprise, as these molecules were not expected to be present in such a cold atmosphere: this suggests that non-equilibrium chemistry is at play. 
- In turn, this disequilibrium chemistry tells us about atmospheric mixing and transport processes that shape planetary evolution.

There is much more to say, but I will stop here. If you are interested, please check out the papers! The point is: all these scientific insights were made possible by our ability to perform Bayesian inference in a complex, high-dimensional, and non-linear model of exoplanet atmospheres.

---

class: middle
count: false

.center.width-15[![](figures/icons/ocean.png)]

# Part 2: Large inverse problems

$p(x|y)$, with $x \in \mathbb{R}^d$, $d = O(10^5)$.

<br>

???

Normalizing flows require invertible transformations, which becomes computationally prohibitive and architecturally limiting for high-dimensional images.

---

class: middle

## Diffusion models 101

Samples $x \sim p(x)$ are progressively perturbed through a diffusion process described by the forward SDE $$\text{d} x\_t = f\_t x\_t \text{d}t + g\_t \text{d}w\_t,$$
where $x\_t$ is the perturbed sample at time $t$.

.center[
.width-90[![](figures/perturb_vp.gif)]
Forward diffusion process.
]

.footnote[Credits: [Song](https://yang-song.net/blog/2021/score/), 2021.]

---

class: middle

The reverse process satisfies a reverse-time SDE that can be derived analytically from the forward SDE as $$\text{d}x\_t = \left[ f\_t x\_t - g\_t^2 \nabla\_{x\_t} \log p(x\_t) \right] \text{d}t + g\_t \text{d}w\_t.$$

Therefore, to generate data samples $x\_0 \sim p(x\_0) \approx p(x)$, we can draw noise samples $x\_1 \sim p(x\_1) \approx \mathcal{N}(0, \Sigma\_1)$ and gradually remove the noise therein by simulating the reverse SDE from $t=1$ to $0$.

.center[
.width-90[![](figures/denoise_vp.gif)]
Reverse denoising process.
]

.footnote[Credits: [Song](https://yang-song.net/blog/2021/score/), 2021.]

---

class: middle 

.center.width-90[![](figures/architecture.svg)]

The .bold[score function] $\nabla\_{x\_t} \log p(x\_t)$ is unknown, but can be approximated by a neural network $d\_\theta(x\_t, t)$ by minimizing the denoising score matching objective
$$\mathbb{E}\_{p(x)p(t)p(x\_t|x)} \left[ || d\_\theta(x\_t, t) - x ||^2\_2 \right].$$
The optimal denoiser $d\_\theta$ is the mean $\mathbb{E}[x | x\_t]$ which, via Tweedie's formula, allows to use $$s\_\theta(x\_t, t) = \Sigma\_t^{-1}(d\_\theta(x\_t, t) - x\_t)$$ as a score estimate of $\nabla\_{x\_t} \log p(x\_t)$ in the reverse SDE.

---

class: middle

## Inverting single observations

To turn a diffusion model $p\_\theta(\mathbf{x})$ into a conditional model $p\_\theta(\mathbf{x} | y)$, we can .red.bold[hard-wire] conditioning information $y$ as an additional input to the denoiser $d\_\theta(x\_t, t, y)$ and train the model on pairs $(x, y)$.

---

class: middle

.center.width-10[![](figures/icons/idee.png)]

Using the Bayes' rule, the posterior score $\nabla\_{x\_t} \log p(x\_t|y)$ to inject in the reverse SDE can be decomposed as
$$\nabla\_{x\_t} \log p(x\_t|y) = \nabla\_{x\_t} \log p(x\_t) + \nabla\_{x\_t} \log p(y|x\_t) - \sout{\nabla\_{x\_t} \log p(y)}.$$

This enables .bold[zero-shot posterior sampling] from a diffusion prior $p(x\_0)$ without having to hard-wire the neural denoiser to the observation model $p(y|x)$.

---

class: middle
exclude: true

.center.width-60[![](figures/classifier-guidance.png)]

.center[

Turning a diffusion model trained on ImageNet 512x512 images into a conditional generator using a classifier $p(y|x)$ as observation model.

]

.footnote[Credits: [Dhariwal and Nichol](https://arxiv.org/abs/2105.05233), 2021 (arXiv:2105.05233).]

---

exclude: true
class: middle

.avatars[![](figures/faces/francois.jpg)![](figures/faces/gerome.jpg)![](figures/faces/lanusse.jpg)]

## Approximating $\nabla\_{x\_t} \log p(y | x\_t)$

We want to estimate the score $\nabla\_{x\_t} \log p(y | x\_t)$ of the noise-perturbed likelihood $$p(y | x\_t) = \int p(y | x) p(x | x\_t) \text{d}x.$$

Our approach:
- Assume a linear Gaussian observation model $p(y | x) = \mathcal{N}(y | Ax, \Sigma\_y)$.
- Assume the approximation $p(x | x\_t) \approx \mathcal{N}(x | \mathbb{E}[x | x\_t], \mathbb{V}[x | x\_t])$,  where $\mathbb{E}[x | x\_t]$ is estimated by the denoiser and $\mathbb{V}[x | x\_t]$ is estimated using Tweedie's covariance formula.
- Then $p(y | x\_t) \approx \mathcal{N}(y | A \mathbb{E}[x | x\_t], \Sigma\_y + A \mathbb{V}[x | x\_t] A^T)$.
- The score $\nabla\_{x\_t} \log p(y | x\_t)$ then approximates to 
$$\nabla\_{x\_t} \mathbb{E}[x | x\_t]^T A^T (\Sigma\_y + A \mathbb{V}[x | x\_t] A^T)^{-1} (y - A \mathbb{E}[x | x\_t]).$$

.footnote[See also [Daras et al (2024)](https://giannisdaras.github.io/publications/diffusion_survey.pdf) for a survey on diffusion models for inverse problems.]

--

exclude: true
count: false

.front.width-70.center[![](figures/meme-gaussian.png)]

---


class: middle

.avatars[![](figures/faces/laurence.jpg)]

## Inverting gravitational lenses

.center[
.width-65[![](figures/lensing.svg)]
.width-25[![](figures/lensing.webp)]
]

.center.width-100[![](figures/lensing.png)]

Posterior source galaxies $x$ can be recovered from gravitional lenses $y$ by zero-shot posterior sampling from a diffusion prior $p(x)$ of galaxy images. 

.italic.center[Check Laurence Perreault-Levasseur talk!]

.footnote[Credits: [Adam et al.](https://arxiv.org/abs/2211.03812), 2022 (arXiv:2211.03812).]

---

class: middle

.avatars[![](figures/faces/victor.jpg)![](figures/faces/mg.png)]

## Nowcasting Black Sea hypoxia from satellite observations

.grid[
.kol-2-3[.width-90[![](./figures/blacksea-oxygen.jpg)]]
.kol-1-3[.center[.width-100[![](./figures/argo-bs.png)]
.width-50[![](./figures/argo.png)] .width-45[![](./figures/argo-oxy.png)]]

]
] 

.center[
How do hypoxic zones evolve in response to climate change? Can we monitor them from space or with sparse measurements?
]

.footnote[Credits: Work in progress with Victor Mangeleer and Marilaure Grégoire.]

???

The Black Sea is a large inland sea between Eastern Europe and Western Asia. It is a unique ecosystem, with a strong stratification that leads to anoxic conditions below 200m depth. This makes it a natural laboratory to study hypoxia, which is a growing problem in many coastal areas worldwide due to climate change and nutrient pollution. Understanding how these zones evolve could inform early warning systems and ecosystem management strategies.

In collaboration with oceanographers, we are developing methods to map the 3D oxygen concentration in the Black Sea from satellite observations of the surface and sparse in-situ measurements. This is a challenging inverse problem, as high resolution is needed:
- Oxygen depletion patterns follow the complex bathymetry and circulation - miss the small-scale features and you miss the physics.
- Satellite observations at 1km resolution can detect blooms and fronts that 25km resolution would completely smooth out

---

class: middle

.avatars[![](figures/faces/victor.jpg)![](figures/faces/mg.png)]

.center.width-100[![](figures/blacksea-results.png)]

Posterior oxygen maps $p(x|y)$ can be recovered from satellite observations $y$ of the surface, by zero-shot posterior sampling from a diffusion prior $p(x)$ of the Black Sea dynamics.

.footnote[Credits: Work in progress with Victor Mangeleer and Marilaure Grégoire.]

???

Fortunately, good physical models of the Black Sea exist, which can be used to train a diffusion prior $p(x)$ of realistic 3D oxygen maps $x$.

Our preliminary results show that we can recover realistic 3D oxygen maps from satellite observations of the surface. 

More work is needed to validate these results and to improve the model, but we are optimistic that this approach can provide valuable insights into the dynamics of hypoxia in the Black Sea.

---

exclude: true
class: middle

.center.width-100[![](figures/m87.png)]

.center[

Posterior M87 black hole images $x$ using a diffusion prior $p(x)$<br> based on GRMHD simulations.

]

.footnote[Credits: [Wu et al.](https://arxiv.org/abs/2405.18782), 2024 (arXiv:2405.18782).]

---

class: middle
count: false

.center.width-15[![](figures/icons/water.png)]

# Part 3: Extra-large inverse problems

$p(x|y)$, with $x \in \mathbb{R}^d$, $d = O(10^6+)$.

<br>

???

As we scale up to extra-large inverse problems to capture more scales and complexity, we face new challenges.

Learning and using diffusion models directly in the data space becomes impractical for high-dimensional images, such as 3D fields or long time series thereof. 

The denoising networks would be enormous and training would be impossible.

---

class: middle, center, black-slide

.center.width-100[![](figures/satellite.gif)]

How can we create a comprehensive record of Earth's atmospheric evolution to understand climate change and improve weather prediction?

???

The last science question I want to discuss is the reconstruction of past atmospheric states from satellite observations. Or said differently
- can we retrieve videos of the atmosphere from noisy, incomplete and coarse-grained satellite observations? 
- can we obtain a distribution of these videos, to quantify the uncertainty in our reconstruction?
- can we do this at the scale of the whole Earth?

The goal is not just to have a pretty video, but to create a comprehensive record of Earth's atmospheric evolution. This record can be used to understand climate change, improve weather prediction, and inform policy decisions.

---

class: middle

.center.width-100[![](figures/dynamical.svg)]

The goal of .bold[data assimilation] is to estimate plausible trajectories $x\_{1:L}$ given one or more noisy observations $y$ (or $y\_{1:L})$ as the posterior $$p(x\_{1:L} | y) = \frac{p(y | x\_{1:L})}{p(y)} p(x\_0) \prod\_{i=1}^{L-1} p(x\_{i+1} | x\_i).$$

???

Assume the latent state $x$ evolves according to a transition model $p(x\_{i+1} | x\_i)$ and is observed through an observation model $p(y | x\_{1:L})$. (Typically, the observation model will be $p(y\_i | x\_i)$, but we consider the general case here.) 

---

class: middle

.avatars[![](figures/faces/francois.jpg)]

.center.width-100[![](figures/sda.svg)]

## Score-based data assimilation 

- Build a score-based generative model $p(x\_{1:L})$ of arbitrary-length trajectories$^\*$. 
- Use zero-shot posterior sampling to generate plausible trajectories from noisy observations $y$.

.footnote[Credits: [Rozet and Louppe](https://arxiv.org/abs/2306.10574), 2023 (arXiv:2306.10574).]

---

class: middle

.avatars[![](figures/faces/francois.jpg)]

.center.width-100[![](figures/sda1-0.png)]

.center[Sampling trajectories $x\_{1:L}$ from<br> noisy, incomplete and coarse-grained observations $y$.]

.footnote[Credits: [Rozet and Louppe](https://arxiv.org/abs/2306.10574), 2023 (arXiv:2306.10574).]

---

class: middle
count: false

.avatars[![](figures/faces/francois.jpg)]

.center.width-100[![](figures/sda1.png)]

.center[Sampling trajectories $x\_{1:L}$ from<br> noisy, incomplete and coarse-grained observations $y$.]

.footnote[Credits: [Rozet and Louppe](https://arxiv.org/abs/2306.10574), 2023 (arXiv:2306.10574).]

---

class: middle, black-slide

.center.width-40[![](figures/earth.jpg)]

.center[... but does it scale to a whole Earth model?]

At 0.25° resolution, for 6 atmospheric variables, 13 pressure levels, hourly time steps, and 14 days of simulation, a trajectory $x\_{1:L}$ contains $721 \times 1440 \times 6 \times 13 \times 24 \times 14 = 27 \times 10^9$ variables.

.grid[
.kol-1-5[.center.width-50[![](figures/icons/danger.png)]]
.kol-4-5[.center.bold[$O(10^9)$ variables (or more) is needed<br> to capture the complexity of the atmosphere.]]
]

---

class: middle

## Latent diffusion models

.center.width-100[![](figures/lsgm.png)]

Latent diffusion models $p\_\theta(z)$ learn a diffusion prior in a compressed latent space $z$ of much lower dimension than the data space $x$.

.footnote[Credits: [Vahdat et al](https://nvlabs.github.io/LSGM/), 2021.]

---

class: middle

.avatars[![](figures/faces/gerome.jpg)![](figures/faces/frozet.jpg)![](figures/faces/sacha.jpg)![](figures/faces/victor.jpg)![](figures/faces/omer.jpg)![](figures/faces/mathias.jpg)![](figures/faces/elise.jpg)]

## Appa: Bending weather dynamics with LDMs

Appa is made of three components:
- a 500M-parameter .bold[autoencoder] that compresses the data space $x$ into a latent space $z$ with a 450x compression factor;
- a 1B-parameter .bold[diffusion model] that generates latent trajectories $z\_{1:L}$;
- a .bold[posterior sampling algorithm] adapted from MMPS (Rozet et al, 2024) that samples from the posterior distribution $p(z\_{1:L} | y)$.

.footnote[Credits: Soon on arXiv!]

---

class: middle

.center[
<video poster="" id="video" controls="" muted="" loop="" width="70%" autoplay>
        <source src="https://montefiore-sail.github.io/appa/static/videos/reanalysis/reanalysis_1week.mp4" type="video/mp4">
</video>

Reanalysis of past data $p(x\_{1:L} | y\_{1:L})$.
]

.footnote[Credits: [Andry et al](https://arxiv.org/abs/2504.18720), 2025 (arXiv:2504.18720).]

---

class: middle

<iframe src="https://montefiore-sail.github.io/appa-live/" width="100%" height="1000px" style="border:none; zoom: 0.5"></iframe>

.center[[Live demo](https://montefiore-sail.github.io/appa-live/) of Appa.] 

---

class: middle

.center.width-10[![](figures/icons/verifier.png)]

## Conclusions

Deep generative models are unlocking previously impossible science.

- .bold[New scientific questions become accessible]: We can now tackle inverse problems with millions to billions of variables that unlock new scientific insights.
- .bold[Statistically principled]: Bayesian inference with uncertainty quantification.
- .bold[Methodological advantages]: Zero-shot inference without retraining. 

Next challenges:
- Rigorous validation: when and why these methods work (or not).
- Making tools accessible to domain scientists.

---

count: false

<br>
.center.width-10[![](figures/icons/high-five.png)]<br>

.center[

.width-15.circle[![](figures/faces/gerome.jpg)] .width-15.circle[![](figures/faces/frozet.jpg)] .width-15.circle[![](figures/faces/victor.jpg)] .width-15.circle[![](figures/faces/omer.jpg)] .width-15.circle[![](figures/faces/sacha.jpg)]

.width-15.circle[![](figures/faces/mathias.jpg)] .width-15.circle[![](figures/faces/elise.jpg)] .width-15.circle[![](figures/faces/malavika.jpg)] .width-15.circle[![](figures/faces/arnaud.jpg)] .width-15.circle[![](figures/faces/joeri.png)]

(Gérome, François, Victor, Omer, Sacha, Matthias, Elise, Malavika, Arnaud, Joeri)

]

---

class: middle, center, end-slide
count: false

The end.

