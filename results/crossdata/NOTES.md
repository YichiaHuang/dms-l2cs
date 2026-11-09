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
