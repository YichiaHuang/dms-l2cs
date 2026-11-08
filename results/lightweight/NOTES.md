# NOTES — 輕量化實驗（Nano4）

> 目的：DMS 要在車用 SoC 上以 ≥25 fps 跑（Euro NCAP SD 202 要求 ≥25 Hz），現行 L2CS ResNet-50 @448 = 16.35 GMACs/幀，
> 只有中高階晶片（≥4–8 TOPS）跑得動 → 測試換骨幹、降解析度、知識蒸餾、int8 量化的精度代價。
> run 在 `/work/u8714813/runs/l2cs_light/`，權重不 commit。

## 2026-11-08 程式變更（`scripts/train_mpii.py`，預設行為不變）
- `--backbone resnet50|resnet18|mobilenet_v3_large|efficientnet_b0`：torchvision ImageNet 權重 + 同樣兩個 28-bin 頭。非 ResNet-50 骨幹全部層都訓練、BN 為 train 模式（原版 conv1/bn1 凍結、layer1-4 BN 固定的怪癖只套用在 ResNet-50）。
- `--img-size`（預設 448）。
- `--teacher PATH`（可含 `{fold}`）+ `--kd-weight 1.0 --kd-temp 2.0`：兩個 bin 頭各加 T²·KL(teacher‖student)。老師 = 同一 fold 的 `improved_e50/last.pkl`（ep50，**不用 best.pkl**，因為 best 是用測試者挑的）。學生 @224 時把輸入雙線性放大到 448 給老師。
- 修正：iters/metrics 記錄 lr 改用最後一個 param group（`get_last_lr()[-1]`），原本寫死 `[1]`，輕量模型只有 1 組會 IndexError。ResNet-50 的 group 1、2 lr 相同，紀錄不變。
- `scripts/remote/slurm_folds_nano4.sbatch`：EXTRA 的空白分隔改用 `+`（原本 `_` 會把 `mobilenet_v3_large` 拆掉）；可用 `RUNS=` 指定 run 目錄。
- 新增 `scripts/bench_light.py`（GMACs、參數、GPU/CPU 延遲、ONNX 匯出、NPU 理論 fps 上限）、`scripts/eval_int8.py`（FX int8 PTQ，校正用訓練者 256 張）。
- 套件：`pip install onnx`（1.19.1）。

## Pilot（dev job 725349，fold0，5 epoch，improved 配方）
| | lr 1e-5 | lr 1e-4 | lr 3e-4 |
|---|---|---|---|
| MobileNetV3-L @448 | 2.92 | 2.86 | 2.80 |
| EfficientNet-B0 @448 | 2.94 | 2.79 | — |
→ 輕量骨幹用 **lr 1e-4**（3e-4 train loss 降得快很多，20 epoch 有過擬合風險；單 fold 雜訊大，取中間值）。KD、resume、@224 冒煙測試都 OK。第一次 pilot（725329）因 `get_last_lr()[1]` IndexError 失敗，修正後重跑。

## 篩選（5 fold 0/3/7/11/14 × 20 epoch，improved 配方，jobs 725389–725393，各 5×H200 約 23 min）
| 設定 | GMACs | p00 | p03 | p07 | p11 | p14 | 平均 @ep20 |
|---|---|---|---|---|---|---|---|
| ResNet-50 @448 | 16.35 | 2.66 | 3.42 | 4.00 | 5.07 | 5.40 | 4.11 |
| ResNet-50 @224 | 4.09 | 2.26 | 4.34 | 4.14 | 5.14 | 6.10 | 4.40 |
| EfficientNet-B0 @448 | 1.54 | 3.04 | 2.62 | 3.70 | 5.39 | 5.31 | 4.01 |
| MobileNetV3-L @448 | 0.85 | 2.96 | 3.48 | 3.73 | 5.79 | 5.66 | 4.33 |
| MobileNetV3-L @448 + KD | 0.85 | 2.70 | 3.12 | 3.69 | 6.03 | 5.34 | 4.18 |
參考：B（ResNet-50 improved 50 epoch）同 5 人 @ep50 = 4.09 → 20 epoch 已足夠。

## 決策
- EfficientNet-B0 @448 補成 15 fold（job 725547，同 tag 續跑，已完成的 5 fold 跳過）。
- EfficientNet-B0 + KD 5 fold（job 725548）。
- int8 PTQ 檢查（dev job 725549）：effb0、mnv3l、mnv3l+KD、r50 各 5 fold。
- 15-fold 的 ResNet-50 對照用 B（improved_e50 @ep50）；5 fold 上 e20 與 e50 只差 0.02°，省下重跑約 6 GPU-h。

## GPU 用量（輕量化部分）
pilot 2 次約 1.1 GPU-h、篩選 5 個約 9.5 GPU-h、bench 0.06 GPU-h；進行中：effb0 15-fold 補 10 fold 約 4 GPU-h、effb0+KD 約 2 GPU-h、int8 約 1 GPU-h（dev 1 卡、只用 CPU）。

## 2026-11-08 結果（後續）
- EfficientNet-B0 + KD（job 725548）：5 人 4.09，不加 KD 是 4.01 → KD 對 EfficientNet-B0 沒有幫助。
- EfficientNet-B0 15 fold（job 725547 補 10 fold）：**4.09 ± 1.20** vs ResNet-50 improved_e50 4.15 ± 1.17，成對差 −0.06 ± 0.34，9/15 較好，p = 0.51。
- int8 PTQ（dev job 725549，CPU，x86 FX，校正 256 張訓練者影像）：ResNet-50 4.13→3.86、MobileNetV3-L 4.36→4.71、MobileNetV3-L+KD 4.19→4.49、**EfficientNet-B0 4.02→7.72**。CPU fp32 與 GPU bf16 評估一致（4.02 vs 4.01），代表評估流程正確。
- 因為 EfficientNet-B0 的 PTQ 崩掉，新增 QAT（`scripts/qat_finetune.py`，job 725779，從 last.pkl 微調 2 epoch、lr 1e-5、第 2 epoch 凍結 observer/BN）：EfficientNet-B0 **4.99**、MobileNetV3-L+KD **4.13**。
- **建議**：上車用 MobileNetV3-L @448 + KD + int8 QAT（0.85 GMACs，5 人 4.13°）。要正式定論需補 15 fold（約 6 GPU-h），尚未執行（quota）。
- QAT 的 int8 權重在 `/work/u8714813/runs/l2cs_light/qat/*_qat_int8.pt`，不 commit。
- 輕量化合計 **18.3 GPU-h**。
