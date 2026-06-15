# What is the Monte Carlo physics prior in this paper?

In our paper, the **Monte Carlo physics prior** is **two analytic channels** computed from any standard RGB capsule frame, fed into the network as auxiliary input alongside the original RGB. There is no actual Monte Carlo simulation at runtime — the name reflects that the math is *guided by* the structure of how a full Monte Carlo light-transport solver would approach the same problem, but we use a first-order analytic approximation that runs in milliseconds per frame.

## The two channels

### Channel 0 — `P_blood`, the hemoglobin probability map

$$P_\text{blood}(x,y) \;=\; \sigma\bigl(\alpha\,(H_\text{norm}(x,y) - 0.5)\bigr)\,\cdot\,\Phi(r)$$

Built from three pieces, each grounded in physics:

| Piece | Formula | Physics it captures |
|---|---|---|
| **Hemoglobin index** | `H(x,y) = R / (G + B + ε)` | Hemoglobin absorbs strongly at green (~540 nm) and blue (~450 nm) and is relatively transparent at red (~660 nm). So R rises and G+B falls in blood-rich pixels — the ratio amplifies that signal. |
| **Robust normalization** | clip H to its 1st–99th percentile, then min-max-rescale to `H_norm` ∈ [0,1] | Vignette dark pixels and specular highlights blow up the raw H_max. Without this clip the prior collapses to ~0 on real frames and the network learns to ignore it. |
| **Radial fluence** | `Φ(r) = exp(−r / λ_eff)`, centered on the image | Capsule LEDs illuminate from a single point inside a ~17 mm body. The photon fluence at distance r from the illumination axis falls off near-exponentially in scattering tissue — this is what a full volumetric MC simulator would compute. We use the first-order radial approximation. λ_eff defaults to 0.25 × image diagonal. |

`α` controls the sharpness of the hemoglobin-vs-mucosa cutoff (we use α = 4). The sigmoid converts the centered `H_norm` into a probability-like value, and multiplying by `Φ(r)` attenuates the regions of the frame the LED couldn't have illuminated strongly — so the prior says "high blood probability where R/(G+B) is high *and* the illumination geometry actually allows confident measurement."

### Channel 1 — `H_AFI × Φ(r)`, the AFI-surrogate

$$H_\text{AFI}(x,y) \;=\; \log\!\bigl((I_G + \epsilon)/(I_B + \epsilon)\bigr)$$

then weighted by the same fluence map:

$$\text{ch}_1(x,y) \;=\; H_\text{AFI}(x,y) \cdot \Phi(r)$$

This emulates from standard RGB the contrast that a real autofluorescence-imaging capsule (PillCam-SPECTRA in the long-term flagship) would produce by spectrally separating a violet-blue excitation band from a green reference band. We can't get true 390–470 nm excitation out of an RGB sensor, so we use the green/blue log-ratio of the captured frame as an approximation — the spectral asymmetry of normal vs. dysplastic mucosa shows up here too, just with weaker SNR than dedicated AFI hardware would produce.

## Why "Monte Carlo"?

A full Monte Carlo light-transport simulator (e.g., MCX, Jacques 2013) tracks millions of photon paths through tissue layers with known absorption and scattering coefficients to compute the volumetric fluence and the per-pixel reflectance distribution. That's the gold standard for tissue-optics modeling but takes minutes per frame on a GPU — not viable for per-frame deep-learning input.

Our radial fluence map `Φ(r) = exp(−r/λ_eff)` is the **first-order analytic approximation** of what a full MC solver would output: in highly scattering tissue and a centered point source, the angularly averaged fluence falls off near-exponentially with distance. Calling our channel "Monte Carlo–guided" reflects that the *form* of the prior is informed by light-transport physics rather than chosen by data fitting, while honestly conceding (per `REVISIONS.md` and the publishing-language guidelines in `tet.txt`) that we are not running a volumetric solver. The earlier-draft phrase "Monte Carlo–informed" was too strong; **"Monte Carlo–guided"** or **"light-transport-informed"** is what publications should use.

## How the network uses it

- The **5-channel pipeline** (`+MC prior` condition) concatenates the 2 physics channels to the 3 RGB channels and expands the backbone's first conv from 3 → 5 input channels with **zero-initialized weights** for the extra two positions. The network starts behaviorally identical to the RGB baseline at step 0, so any later AUROC gain is directly attributable to training learning to use the physics channels.
- The **3-channel distillation pipeline** (`+Distill` condition) doesn't feed the prior as input at all — it uses `P_blood` as a *teacher signal* for a small auxiliary decoder. After training, the decoder is discarded and the deployed model takes plain RGB.

## Where it lives in the code

| Function | File | Lines |
|---|---|---|
| `hemoglobin_index` | `paper/Capsule-Endoscopy/physics_prior.py` | 25–34 |
| `afi_log_ratio` | same | 37–47 |
| `fluence_map` | same | 50–67 |
| `blood_probability` | same | 70–93 |
| `physics_channels` *(the 2-channel concatenator)* | same | 96–112 |

The same analytic functions in `physics_prior.py` produce the qualitative
panel (RGB | H map | Φ map | P_blood overlay) shown as Fig. 2 in the paper.

## In one sentence

The Monte Carlo physics prior is a **per-pixel hemoglobin probability map plus an AFI log-ratio map, both analytically computed from the input RGB and modulated by a Monte Carlo–guided radial fluence approximation of the capsule's illumination geometry**, fed to the classifier as two extra input channels — giving the network a physics-grounded blood-detection cue and an AFI-surrogate contrast cue without changing the capsule hardware.
