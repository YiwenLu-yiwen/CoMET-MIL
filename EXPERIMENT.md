# Experiment Summary

## Final Kept Pipeline

The kept method is:

1. Train a base LUDB segmenter
2. Refine it with a P-sensitive objective
3. Freeze the refined segmenter and extract calibrated beat evidence
4. Train a PTB-XL record-level MIL classifier on that evidence

Pipeline:

```text
ECG -> segmenter -> calibrated beat evidence + RR context -> MIL classifier
```

## What Was Actually Tested

The experiments answered five specific questions about failure mode, transfer
design, and deployment choice.

Together, these studies provide the empirical support for the five contribution
claims summarized in [README.md](/home/sam_laptop/AI/example_archive/ecg2/README.md)
and mapped to prior lineage in [REFERENCES.md](/home/sam_laptop/AI/example_archive/ecg2/REFERENCES.md).

| Study name | What was changed | What improved or failed | Main takeaway |
|---|---|---|---|
| Task ceiling and bottleneck diagnosis | Compared oracle morphology evidence against predicted segmentation evidence | Oracle evidence was near-ceiling, predicted evidence was much weaker | The task definition is valid; the bottleneck is segmentation quality |
| Evidence transfer design | Compared naive pooled summaries against beat-wise morphology evidence with RR context and calibration | Context-aware transfer outperformed shallow pooled features; calibration added a smaller extra gain | Explicit beat evidence is better than generic pooled embeddings |
| P-wave-sensitive refinement | Added pre-QRS P-presence and P-absence refinement terms to the segmenter | Better P-sensitive delineation translated into better downstream classification | The segmenter is causally important, not just an auxiliary pretraining step |
| Record classifier design | Compared simple record pooling against record-level MIL aggregation | MIL gave the strongest record classifier under imbalance-aware metrics | Some beats matter more than others; the head must learn beat importance |
| External transfer and sampling robustness | Evaluated the final pipeline on PTB-XL and compared `100 Hz` and `500 Hz` input choices | Native `100 Hz` failed badly; `100 -> 500` recovered most performance; native `500 Hz` was best | The method transfers externally, but it clearly prefers a `500 Hz` input space |

## Study Findings

### 1. Task Ceiling and Bottleneck Diagnosis

What was done:

- built a morphology-only record classification path using oracle waveform
  evidence
- compared it against the same downstream task using predicted segmentation
  evidence

What was learned:

- the AFIB/AFLT task is solvable from morphology evidence
- the gap came from teacher quality, not from the label design

Why it matters for the contribution claim:

- this justified spending effort on improving the segmenter instead of replacing
  the downstream task formulation

### 2. Evidence Transfer Design

What was done:

- compared simple pooled summaries against beat-wise evidence transfer
- added RR-based local rhythm context
- tested temperature calibration on segmenter logits

What was learned:

- transferring beat-level waveform evidence is materially better than pushing a
  shallow pooled summary downstream
- context helps because AFIB/AFLT is not purely a single-beat morphology problem
- calibration helps, but it is a secondary gain after fixing the transfer
  structure itself

Why it matters for the contribution claim:

- the final classifier should consume explicit evidence, not a generic shared
  latent representation

### 3. P-Wave-Sensitive Segmenter Refinement

What was done:

- added a refinement stage that rewards correct pre-QRS P evidence on positive
  beats
- simultaneously penalized false pre-QRS P mass on negative beats
- kept the downstream classifier structure fixed while swapping only the
  segmenter

What was learned:

- better delineation of P-present versus P-absent structure improved the final
  record classifier
- this isolates segmentation quality as a real upstream cause of classifier
  quality

Why it matters for the contribution claim:

- the segmenter is not just a preprocessing convenience
- the refined segmenter is the correct upstream model to keep

### 4. Record Classifier Design

What was done:

- compared simpler record pooling against record-level MIL aggregation
- evaluated with imbalance-aware metrics rather than raw accuracy

What was learned:

- MIL produced the strongest final classifier
- the gain is consistent with the clinical structure of the problem: only some
  beats in a record are maximally informative

Why it matters for the contribution claim:

- the final PTB-XL head should remain a downstream evidence-driven MIL model

### 5. External Transfer and Sampling Robustness

What was done:

- transferred the refined LUDB segmenter to PTB-XL
- trained and evaluated the downstream PTB-XL classifier
- ran a controlled robustness study comparing PTB-XL `records100` and
  `records500`

What was learned:

- the method is not only a LUDB-only effect
- the model strongly prefers a `500 Hz` input space
- when only `records100` is available, upsampling to `500 Hz` is much better
  than staying at native `100 Hz`

Why it matters for the contribution claim:

- the final deployment recommendation depends on what PTB-XL waveform resolution
  is available locally

## Final Kept Full-Data Result

Source:

```text
runs/step4-ptbxl-classifier/summary.csv
```

Setting:

- PTB-XL source files: `records100`
- model input sampling rate: `500 Hz`
- mode: `record_mil_ctx_cal`
- seed: `43`

Result:

| Setting | Test balanced acc | Precision | F1 | MCC | PR-AUC | Pos recall | Neg recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full PTB-XL kept pipeline, `records100 -> 500 Hz` | 0.9423 | 0.7647 | 0.8314 | 0.8178 | 0.9317 | 0.9108 | 0.9737 |

Interpretation:

- this is the strong packaged path currently used in the repo
- it uses the `100 Hz` PTB-XL files, but the model itself still runs in a
  `500 Hz` input space after resampling
- this result is the main external validation point supporting the practical
  value of the kept CoMET-MIL pipeline

## PTB-XL Sampling-Rate Robustness

To compare `100 Hz` and `500 Hz` fairly, all three settings were evaluated on
the same PTB-XL subset for which local `records500` files were available.

Command:

```bash
bash scripts/run_ptbxl_sampling_robustness.sh
```

Outputs:

```text
runs/exp-ptbxl-robustness-100-native
runs/exp-ptbxl-robustness-100-to-500
runs/exp-ptbxl-robustness-500-native
```

Subset availability:

- train: `397`
- val: `82`
- test: `73`

How to read the settings:

| PTB-XL source files | Model input fs | Meaning |
|---|---:|---|
| `records100` | `100` | use the native `100 Hz` files and keep them at `100 Hz` |
| `records100` | `500` | use the native `100 Hz` files, then upsample to `500 Hz` before inference |
| `records500` | `500` | use the native `500 Hz` files and keep them at `500 Hz` |

Results:

| PTB-XL source files | Model input fs | Test balanced acc | Precision | F1 | MCC | PR-AUC | Pos recall | Neg recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `records100` | `100` | 0.5240 | 0.1667 | 0.1429 | 0.0547 | 0.1883 | 0.1250 | 0.9231 |
| `records100` | `500` | 0.8837 | 0.5000 | 0.6364 | 0.6088 | 0.8254 | 0.8750 | 0.8923 |
| `records500` | `500` | 0.8913 | 0.5385 | 0.6667 | 0.6391 | 0.7789 | 0.8750 | 0.9077 |

Insight:

- native `100 Hz` is not robust enough for this method
- moving the same `records100` files into a `500 Hz` model input space recovers
  most of the lost performance
- native `records500 -> 500 Hz` is still the cleanest and best-performing setup
- this robustness pattern strengthens the claim that the method is not only
  sensitive to label definition, but also to the fidelity of transferred
  morphology evidence

## Final Findings

1. The central problem was not label validity. It was evidence quality.
2. Explicit beat-wise morphology transfer is better than shallow pooled
   summaries.
3. Improving P-sensitive delineation improves downstream classification, so the
   segmenter is a true causal bottleneck.
4. The final classifier should remain downstream and evidence-driven, not folded
   back into a joint raw-ECG classifier.
5. The method transfers to PTB-XL, but it prefers `500 Hz`; native `100 Hz`
   should be avoided.

Taken together, these findings support the repository's main research-facing
claims:

- CoMET-MIL is not only a workflow change; it is a meaningful restructuring of
  segmentation-guided ECG transfer.
- the P-sensitive refinement stage contributes materially to downstream rhythm
  classification quality.
- explicit beat-wise evidence transfer and MIL aggregation are justified by the
  ablation results, not only by design preference.
- the external PTB-XL results make the study more valuable to other
  researchers than a LUDB-only internal experiment.

## Practical Recommendation

- Best setting:
  - `records500` with `--ptbxl-resolution 500 --target-fs 500`
- Current packaged strong setting:
  - `records100` with `--ptbxl-resolution 100 --target-fs 500`
- Avoid:
  - `records100` with `--ptbxl-resolution 100 --target-fs 100`
