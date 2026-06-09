# References

## Original Lineage

This repository extends the idea from:

1. **[R1] Deep learning based ECG segmentation for delineation of diverse
   arrhythmias**
   - original segmentation-guided idea
   - this repository extends it by replacing shared-task coupling with
     asymmetric evidence transfer

## Public Benchmark and Transfer References

2. **[R2] PTB-XL, a large publicly available electrocardiography dataset**
   - public 10-second 12-lead benchmark dataset used here for external testing

3. **[R3] Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL**
   - benchmark framing reference for PTB-XL-style model comparison

4. **[R4] Attention-based Deep Multiple Instance Learning**
   - record-level beat aggregation reference behind the kept MIL classifier

## Contribution-to-Reference Map

- `Contribution 1` in [README.md](/home/sam_laptop/AI/example_archive/ecg2/README.md)
  extends `R1` by replacing shared-task coupling with morphology-first
  asymmetric evidence transfer.
- `Contribution 2` extends `R1` with a P-sensitive refinement objective that is
  specific to AFIB/AFLT-oriented atrial evidence.
- `Contribution 3` extends `R1` by moving from shared latent coupling to
  explicit beat-wise evidence transfer with contextual features.
- `Contribution 4` is methodologically aligned with `R4`, but applied here as a
  record-level beat aggregation head on segmenter-derived evidence.
- `Contribution 5` is positioned against the external benchmark framing in `R2`
  and `R3`.

## Project Positioning

This repository is best described as a **research-oriented extension study** of
ECG morphology-to-rhythm transfer. It is not a direct reproduction, but it does
contain methodological changes that may be useful to other researchers working
on segmentation-guided ECG classification.

The logic is coherent:

1. start from the original segmentation-guided idea;
2. remove forced shared-backbone coupling;
3. train the segmenter to be strong on P-sensitive delineation first;
4. transfer calibrated morphology evidence to a record-level classifier;
5. verify transfer on a second public dataset.

The current final result supports that framing:

- LUDB is used for segmentation training and controlled ablations;
- PTB-XL is used for external testing;
- the final kept external model is `record_mil_ctx_cal`.

The current final PTB-XL result is documented in
[EXPERIMENT.md](/home/sam_laptop/AI/example_archive/ecg2/EXPERIMENT.md).

## Current Claim Boundary

The work is strong enough to be presented as:

- morphology-to-rhythm evidence transfer
- a public-data extension of the original idea
- a cross-dataset validation study
- an ECG extension study with targeted architectural and objective-level
  contributions
  inside an existing research lineage

It should not yet be presented as:

- a direct reproduction of the original full paper
- a standalone paper-ready contribution package
- a final SOTA claim across the broader PTB-XL literature
- a wholly de novo foundation-model-style ECG paradigm

because the current benchmark still uses:

- single lead `II`
- a filtered binary AFIB/AFLT-vs-SR setup
- no matched published comparison table in this repository
