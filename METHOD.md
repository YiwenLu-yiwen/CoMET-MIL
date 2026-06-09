# Method: CoMET-MIL

## Goal

CoMET-MIL learns a strong ECG segmenter first, then transfers its morphology
evidence to a record-level AFIB/AFLT classifier without forcing segmentation
and classification to share one backbone.

The kept final path is:

1. LUDB segmenter
2. P-sensitive segmenter refinement
3. calibrated beat evidence extraction
4. context-aware record-level MIL aggregation

## Notation

- input ECG: $x \in \mathbb{R}^{T}$
- segmentation classes: $c \in \{P, QRS, T, N\}$
- segmenter logits at time $t$: $z_t \in \mathbb{R}^4$
- segmenter probabilities: $p_t = \mathrm{softmax}(z_t)$
- QRS anchor for beat $i$: $a_i$
- beat window around anchor $i$: $W_i$
- record label: $y \in \{0,1\}$

## 1. Base Segmentation

The segmenter predicts dense waveform labels:

$$
f_\theta(x) = \{z_t\}_{t=1}^{T}, \qquad
p_t = \mathrm{softmax}(z_t).
$$

Base segmentation uses per-sample cross-entropy:

$$
\mathcal{L}_{seg}
=
\frac{1}{T}\sum_{t=1}^{T}\mathrm{CE}(p_t, g_t),
$$

where $g_t$ is the one-hot waveform label.

## 2. P-Sensitive Refinement

For each beat, only the pre-QRS region should carry P-wave evidence. Let
$W_i^{pre}$ be the pre-QRS region of beat $i$.

For positive beats, we maximize soft Dice on the P channel:

$$
\mathrm{Dice}_i
=
\frac{
2\sum_{t \in W_i^{pre}} p_{t,P} g_{t,P}
}{
\sum_{t \in W_i^{pre}} p_{t,P}
+
\sum_{t \in W_i^{pre}} g_{t,P}
+
\varepsilon
}.
$$

The positive-beat loss is:

$$
\mathcal{L}_{pre}
=
1 - \frac{1}{|B^+|}\sum_{i \in B^+}\mathrm{Dice}_i.
$$

For negative beats, we suppress false pre-QRS P mass:

$$
\mathcal{L}_{abs}
=
\frac{1}{|B^-|}\sum_{i \in B^-}
\frac{1}{|W_i^{pre}|}\sum_{t \in W_i^{pre}} p_{t,P}.
$$

The kept refined segmenter objective is:

$$
\mathcal{L}_{refine}
=
\mathcal{L}_{seg}
+ \lambda_{pre}\mathcal{L}_{pre}
+ \lambda_{abs}\mathcal{L}_{abs}.
$$

## 3. Calibrated Evidence Transfer

The refined segmenter is reused as a teacher. Its logits are temperature-scaled:

$$
\tilde{p}_t
=
\mathrm{softmax}\left(\frac{z_t}{\tau}\right),
$$

where $\tau > 0$ is fitted on held-out teacher behavior.

For beat $i$, the classifier receives morphology evidence extracted from the
calibrated window:

$$
h_i = \phi\left(\tilde{p}_{W_i}, r_i\right),
$$

where $r_i$ is beat context:

$$
r_i = [RR_{prev}, RR_{next}, RR_{mean}, RR_{irr}, \mathbf{1}_{prev}, \mathbf{1}_{next}].
$$

So each beat feature contains:

1. local waveform evidence from the segmenter
2. local rhythm context from neighboring beats

## 4. Record-Level Aggregation

### Mean Context Baseline

The simple record embedding is:

$$
h_{rec}^{mean}
=
\left[
\frac{1}{n}\sum_{i=1}^{n} h_i
;\,
\max_{i=1,\dots,n} h_i
\right].
$$

### MIL Aggregation

The kept final classifier uses attention-style multiple-instance learning:

$$
\alpha_i
=
\frac{
\exp\left(w^\top \tanh(V h_i)\right)
}{
\sum_{j=1}^{n}\exp\left(w^\top \tanh(V h_j)\right)
},
$$

$$
h_{rec}^{mil}
=
\sum_{i=1}^{n}\alpha_i h_i.
$$

The record-level probability is:

$$
\hat{y}
=
\sigma(u^\top h_{rec}^{mil} + b).
$$

The classifier is trained with binary cross-entropy:

$$
\mathcal{L}_{cls}
=
-y\log \hat{y} - (1-y)\log (1-\hat{y}).
$$

## 5. Why This Structure Was Kept

The retained idea is:

1. delineation and rhythm classification are related but not identical tasks
2. segmentation should specialize first
3. the classifier should consume explicit morphology evidence, not shared latent
   features
4. some beats matter more than others, so beat weighting must be learnable

This is why the final external model is:

$$
\text{P-sensitive refined segmenter}
\rightarrow
\text{calibrated contextual beat features}
\rightarrow
\text{record-level MIL classifier}.
$$

## 6. Final Benchmark Finding

The final kept external benchmark is documented in
[EXPERIMENT.md](/home/sam_laptop/AI/example_archive/ecg2/EXPERIMENT.md).

The best external model is `record_mil_ctx_cal`.

## 7. Reference Lineage

The original lineage and the project-positioning note are kept
in `REFERENCES.md`.
