# NOTES — 跨資料集（Gaze360、RT-GENE）

## 2026-11-09 下載
- 使用者要求改用公開可下載的資料集（官方 Gaze360 網站回 403；ETH-XGaze、EYEDIAP 需申請）。
- **Gaze360**：Hugging Face 社群鏡像 `immediately/Gaze360-split`（6 個分割檔共 30.5 GB，用 HF 的 LFS sha256 驗證）。非官方來源；原始授權為非商業研究用途，報告需引用 Kellnhofer et al., ICCV 2019 並註明來源。
- **RT-GENE**：官方 Zenodo record 2529036（open access，CC BY-NC-SA 4.0，35 檔共 47.7 GB，用 md5 驗證）。
- 放在 `/work/u8714813/datasets/raw/{gaze360,rtgene}`（700 權限），不 commit。
- **計畫沒寫到的狀況**：Nano4 的 CPU 佇列（NCHC_*、ngs*）不接受 acd110018，若在 GPU job 裡下載會白燒 GPU 時數 → 改在 login node 用 `nice -n 19 ionice -c3` 下載（純網路 I/O，記憶體很小），tmux session `dl`，`download.sh` 可續傳、驗證失敗會刪檔重下，紀錄在 `download.log`。

## 2026-11-09 15:00 RT-GENE: only the `_glasses` archives are labelled
- Inspected the Zenodo tars: `s0XX_noglasses.tar` holds only `natural/face/*.png` + kinect calibration (no gaze labels);
  `s0XX_glasses.tar` holds `label_combined.txt` + `inpainted/face/` (glasses removed by the authors) etc.
- Decision (conservative, saves ~half the download): restarted the downloader (`download.sh`, resumable `curl -C -`)
  on `rtgene/files_labelled.txt` = the 17 `_glasses` tars + LICENSE. The 4 `_noglasses` tars already downloaded are kept
  on disk (unused) and can be deleted.
- `scripts/prep_rtgene.py`: inpainted face crops copied as-is; gaze_h = deg(phi), gaze_v = deg(theta)
  (RT-GENE's own mapping matches train_mpii.gazeto3d); idempotent, re-run when more tars arrive.
  First pass (5 subjects): 34,318 labelled faces, all labels matched to an image.

## 2026-11-09 Gaze360 prep + sign/axis check (job 735475, dev, 1×H200, ~10 min)
- Gaze360 prep: train 84,902 / val 11,318 / test 16,031 face frames, 0 failures — exactly the GazeHub split sizes.
- First label guess (h=atan2(-x,-z), v=asin(-y)) was wrong in BOTH signs: official L2CS Gaze360 model gave 71.8°,
  best variant (h-, v-) 10.56°; our MPII models agreed (h/v correlation −0.5…−0.9). Fixed in prep_gaze360.py to
  h=atan2(x,-z), v=asin(y). Re-scored the saved predictions (`eval_crossdata.py --from-preds`, CPU, no extra GPU).
  -> **official model 10.56° on Gaze360 test vs paper 10.41°**: preprocessing validated.
- RT-GENE: the first guess (gaze_a=vertical) gave corr ≈ 0 for every model; the swapped order (gaze_a = horizontal,
  gaze_b = vertical, same signs) gave corr 0.53–0.67 for the official model and 3 of ours. Fixed in prep_rtgene.py.
  The convention is fixed once per dataset (from the official + independent models), never chosen per model.
- Partial RT-GENE (5 subjects s006/s008/s009/s010/s014, 34,318 faces) numbers are preliminary; final run after
  all 17 labelled subjects are downloaded.

## 2026-11-09 16:00 Gaze360 re-training (user approved, "好")
- `scripts/train_gaze360.py`: same ResNet-50 L2CS model / loss (CE + 1.0·MSE) / Adam 1e-5 / batch 16 / bf16 as the MPII
  runs, but L2CS Gaze360 binning 90×4°, ±180°. Two recipes, 50 epochs each, 1×H200 each (8gpus partition):
  `paper` (const lr, no aug, job 735800) and `improved` (cosine, aug + label-consistent flip, job 735801).
  Out: /work/u8714813/runs/gaze360/r50_<recipe>_e50 (weights not committed).
- Model selection: val AND test every epoch; report test@last, test@best-val (honest), test@best-test (optimistic).
- Smoke test (job 735778, dev, 300 it/epoch): train / skip-when-done / resume all OK.
- Measured ~3.0 min/epoch incl. eval -> ~2.6 h per run, ~5.3 GPU-h total.
- Epoch 1: test 13.45° (paper) / 12.72° (improved); paper reports 10.41°.

## 2026-11-09 18:20 Gaze360 re-training finished (jobs 735800/735801, ~2.4 h each, ~4.8 GPU-h)
| recipe | test @ep50 (last) | test @best-val epoch | best test epoch (optimistic) |
|---|---|---|---|
| paper (const lr, no aug) | **11.04°** | 11.09° (ep48) | 11.04° (ep50) |
| improved (cosine, aug+flip) | 11.19° | 11.18° (ep22) | 11.11° (ep41) |
Paper: 10.41°; official released weights through our pipeline: 10.56°. Gap to the paper ≈ 0.6°.
Unlike MPIIFaceGaze, the paper recipe does not over-fit here (85k training images, val/test still improving at ep50);
the improved recipe is not better on Gaze360 (−0.15° worse at ep50). Possible reasons for the remaining gap: training
length/batch size in the official run (not specified exactly), our crop from the head image vs. GazeHub's crop.
- `scripts/mpii_to_csv.py`: MPIIFaceGaze labels -> CSV for cross-dataset testing of the Gaze360 models
  (37,667 faces = the normalized set we trained on; the GazeHub-normalised release has fewer than 3000/person).

## 2026-11-09 Gaze360-trained models, cross-dataset (job 736623, dev ~8 min)
| model | Gaze360 test | MPIIFaceGaze (all 37,667, zero-shot) |
|---|---|---|
| official L2CS Gaze360 weights | 10.56° | 8.03° (earlier local run: 8.06°) |
| ours, paper recipe @ep50 | 11.04° | 8.38° |
| ours, improved recipe @ep50 | 11.19° | 8.18° |
Saved as results/crossdata/gaze360_test_g360models.json and mpii_all.json (separate name so the MPII-model
gaze360_test.json is not overwritten).

## 2026-11-09 21:30 RT-GENE full (17 subjects, 114,558 faces; job 737204, dev 28 min incl. prep on the compute node)
Pooled (image-weighted) mean error; MPII models = mean ± sd over the 15 fold models.
| model | RT-GENE |
|---|---|
| MPII ResNet-50 paper | 14.41 ± 0.31 |
| MPII ResNet-50 improved | 14.96 ± 0.29 |
| MPII EfficientNet-B0 | 15.19 ± 0.38 |
| MPII MobileNetV3-L + KD | 14.96 ± 0.30 |
| official Gaze360 | 27.12 |
| our Gaze360 paper / improved | 28.64 / 29.21 |
Diagnostics (fold0 / g360 paper): h/v correlation only 0.4–0.6 for every model; removing the mean offset (oracle)
lowers error by only 1–2° -> appearance gap, not a frame/sign issue. Gaze360 models predict wide angles
(pred sd 32° vs gt 14.5°), MPII models are clamped to ±42° and therefore lose less.
Also corrected RESULTS_FINAL.md: MPII test set is 37,667 faces (all frames), not "44,999" (GazeHub 3000/person list).
Cross-dataset GPU total ≈ 5.6 GPU-h (jobs 735475, 735778, 735800, 735801, 736623, 737204).
Downloaded-but-unused `_noglasses` tars (no labels) remain in /work/u8714813/datasets/raw/rtgene; deletable.

## 2026-11-10 Next experiment (user approved): wide-range lightweight model (G2 / J1 / J1-int8)
Goal: test the report's recommendation "train on wide-range data, then lightweight + int8".
- G2: MobileNetV3-L @448, Gaze360 train only, KD from G1 (= r50_paper_e50 last.pkl), improved recipe, lr 1e-4, 20 ep (job 747927).
- J1: same + MPII persons p01 p02 p04 p05 p06 p08 p09 p10 p12 p13 (25,206 faces) in training; KD only on Gaze360
  samples (teacher never saw MPII) (job 747929; first submission 747928 cancelled within seconds: the MPII list
  separator "/" was not converted -> fixed in train_light.sh).
- Held-out MPII persons p00 p03 p07 p11 p14 (the 5-person screening set, 12,461 faces) evaluated every epoch for both.
- Pre-registered success criteria: RT-GENE clearly below the current ~15° (target < 12°); MPII held-out not worse than
  the deployed model by > 0.5°; int8 within 0.3° of fp32.
- Reference for MPII held-out: the MPII-only LOPO models (each trained on 14 MPII persons — more MPII data than J1's 10).
- Smoke test job (dev, 300 it/epoch) OK: 110,108 training images for J1 = 84,902 + 25,206.

## 2026-11-10 Results: G2 / J1 / int8 (jobs 747927, 747929, 748235 GPU eval; 748233, 748234 QAT + CPU eval)
fp32 (GPU bf16, last epoch; MPII = person-mean over held-out p00/p03/p07/p11/p14; others pooled):
| model | Gaze360 test | MPII held-out 5 | RT-GENE (pooled / subject-mean) |
|---|---|---|---|
| deployed MPII-only MNv3L+KD (LOPO, 14 persons/fold) fp32 / QAT | ~31 | 4.18 / 4.13 | ~15.0 |
| G1 = R50 Gaze360 paper e50 | 11.04 | 8.43 | 28.64 / 27.54 |
| G2 = MNv3L+KD, Gaze360 only, 20 ep | 11.71 (best-val ep13: 11.65) | 8.41 | 28.93 / 27.61 |
| J1 = MNv3L+KD, Gaze360 + 10 MPII persons | 11.57 (best-val ep6: 11.30) | 5.27 | 28.92 / 26.41 |
QAT (2 ep, ~10–12 min GPU) then CPU eval, fp32 on CPU from the same loader:
| model | Gaze360 fp32→int8 (pooled; person-mean) | MPII-5 fp32→int8 (pooled; person-mean) | RT-GENE fp32→int8 (pooled) |
|---|---|---|---|
| G2 | 11.64→12.26 (+0.62); 12.48→14.34 | 8.49→8.61 (+0.12); 8.37→8.35 | 28.54→30.38 (+1.84) |
| J1 | 11.58→11.92 (+0.33); 12.24→13.48 | 5.09→5.36 (+0.27); 5.26→5.58 (+0.32) | 28.58→27.55 (−1.03) |
CPU fp32 vs GPU bf16 differ by ≤0.02° on Gaze360/MPII but ~0.35–0.4° on RT-GENE (out-of-distribution inputs are
more precision-sensitive); int8 deltas are always against the CPU fp32 of the same images.
The smoke-run lines at the top of qat-G2-748233.out (20 iters, 200 images) are not results.

Verdict against the pre-registered criteria:
1. RT-GENE < 12°: FAIL. Wide-range training makes RT-GENE worse (~29° vs ~15° for MPII-only).
2. MPII held-out within 0.5° of deployed: FAIL. J1 5.27 vs 4.18 (+1.09; J1 sees 10 MPII persons vs 14).
3. int8 within 0.3° of fp32: FAIL (marginal) for J1: +0.33 Gaze360, +0.32 MPII person-mean (+0.27 pooled).
   G2 fails clearly on Gaze360 (+0.62). The MPII-only model lost nothing (−0.05) under the same QAT schedule.
Conclusion: "train on wide-range data, then lightweight + int8" is NOT supported as a fix for RT-GENE-type appearance
shift. J1 is still useful as a single small model covering both wide-range (Gaze360 11.6°) and frontal (MPII 5.3°) data.
GPU: 747927 1.12 h, 747929 1.22 h, 748233 1.12 h, 748234 1.18 h, 748235 0.06 h, 747914 smoke 0.06 h, 747928 ~0
→ ≈4.8 GPU-h. (QAT jobs spent most wall time on the CPU eval; the GPU sat idle for that part.)
Weights qat_int8.pt and predictions/*.npz stay in /work (not committed).

## 2026-11-11 Next experiment prepared: R1 = small target-domain fine-tuning on RT-GENE (README §4)
scripts/finetune_rtgene.py: start from J1, RT-GENE subjects split by (sorted index % 3), fine-tune on N images per
adapt subject (10 / 100 / all, + all with 50% replay of the base training data), 1500 iters, Adam 3e-5 cosine to 1/100;
evaluate RT-GENE test subjects, MPII held-out 5, Gaze360 test at iter 0/500/1000/1500.
Pre-registered: (1) RT-GENE new subjects < 12° with ≤ 100 images/subject; (2) MPII-5 and Gaze360 each ≤ +0.5° vs before
(judged mainly on the replay run); (3) only if (1) holds: all 3 folds + int8 QAT.
Smoke test job 751388 (dev, 6 min, 20 iters, 300-image evals) OK for all three modes (J1 per-subject, J1 replay,
MPII-only 28-bin base fold0_mnv3l_448_kd_e20 with --mpii-test p00). Smoke numbers are not results.
Screening script /work/u8714813/runs/rtgene_ft/screen.sh (fold 0, 4 settings, ~1 GPU-h) prepared, NOT submitted yet.
