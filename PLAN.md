# L2CS-Net 重現實驗 — 計畫（2026-10-15）

## 目標
用公開資料集自己訓練一次 L2CS-Net，和論文（ICIP 2022 / arXiv:2203.03339）報告的數字比較，並把數據整理成表格與曲線。

## 論文的設定（MPIIFaceGaze）
| 項目 | 論文 / 官方 repo |
|---|---|
| 資料集 | MPIIFaceGaze，15 人，GazeHub 前處理（224×224 正規化臉，再 resize 448） |
| 協定 | leave-one-person-out，15 fold，取平均 |
| Backbone | ImageNet 預訓練 ResNet-50，conv1/bn1 凍結 |
| 輸出 | pitch、yaw 各 28 個 bin（3°，[-42, 42)），CE + α·MSE(softmax 期望值)，α = 1 |
| 優化 | Adam，lr 1e-5，batch 16，50 epoch，無資料增強 |
| 結果 | **3.92° 平均角度誤差**（Gaze360：10.41°） |

## 這台機器的限制
- RTX PRO 1000 Blackwell 8 GB：實測 fp32 28 img/s、bf16 autocast 45 img/s → 一個 epoch（34,740 張）約 13 分鐘。
- 論文完整協定 = 15 fold × 50 epoch ≈ 160 小時，不可行。
- Gaze360 需要向 MIT 申請授權才能下載，這輪先不做。

## 實際要跑的（`scripts/run_queue.sh`，依序）
1. **Zero-shot 參考**：官方 Gaze360 權重（DMS 目前實際在用的那份）直接在 15 個人上評估 → 跨資料集誤差。
2. **fold 0（測 p00）15 epoch**，bf16，每個 epoch 評估、存檔 → 學習曲線，看幾個 epoch 內能接近 3.92°。
3. **fold 7（p07）、fold 14（p14）各 10 epoch** → 看跨人的變異，避免只用一個人下結論。

預估總時間約 8 小時；每個階段完成會在 `~/runs/l2cs/queue/` 留 `.done` 檔，中途停掉也有可用的部分結果。

## 與論文不同之處（寫結果時要註明）
- 只跑 3/15 個 fold、epoch 數 15/10 而非 50。
- bf16 autocast（實測 150 iter 的 loss 與 fp32 一致：81.37 vs 81.41）。
- 前處理是自己移植 GazeHub 的程式（`scripts/prep_mpiifacegaze.py`），which-eye 旗標改用 MPIIFaceGaze 第 28 欄而非 MPIIGaze 的 sample list，產生的格式與 GazeHub 相同。
- 官方 `train.py`/`test.py` 的 mpiigaze 分支本身跑不起來（多處 bug），改用 `scripts/train_mpii.py`，配方照舊，另保留原版「layer1-4 的 BN 固定 eval」的行為。

## 產出
- `results/`：每個 fold 的 metrics.csv、學習曲線圖、彙整表、正規化樣本圖。
- `README.md` 加一節「訓練 vs 論文」。
- 權重與 log 放 Linux 檔案系統 `~/runs/l2cs/`（避免 OneDrive 同步 GB 級檔案）。

---

# 第二輪：論文配方 vs 改良配方（2026-10-19 晚間，`scripts/run_overnight.sh`）

## 動機
第一輪發現 lr 1e-5 固定、無增強的設定下，測試誤差在第 1 個 epoch 後就不再下降，只在 ±0.5° 內震盪，而且跨人差異（p00 2.6° vs p07 4.4°）很大。

## 改良配方（`--recipe improved`）
| 項目 | paper | improved |
|---|---|---|
| 學習率 | Adam 1e-5 固定 | cosine 衰減 1e-5 → 1e-7（逐 iteration） |
| 資料增強 | 無 | 色彩抖動（亮度/對比 0.3、飽和 0.2、色相 0.02）、平移 ±5%、縮放 ±5%、水平翻轉 50%（翻轉時水平視線角取負） |
| 其他 | 相同：ImageNet ResNet-50、28 bin × 3°、CE + MSE、batch 16、bf16、layer1-4 BN 固定 eval | 相同 |

## 排程（每個 run 7 epoch，約 65 分鐘；共 8 run，約 9 小時）
1. fold 0 improved、fold 7 improved（與第一輪的 paper 結果成對比較，取前 7 epoch）
2. fold 3、11、14：paper 與 improved 各一次（p03、p11 是 zero-shot 最難的兩個人，p14 資料最少）

已完成的 run 有 `done.json`，重跑腳本會自動跳過，所以中斷後再執行一次就能續跑。

## 保存的數據（每個 run 目錄 `~/runs/l2cs/<run>/`）
- `metrics.csv`：每 epoch 的訓練 loss（CE/MSE 分開）、測試角度誤差 mean/median/std/p90、pitch/yaw MAE、lr、時間、吞吐量、GPU 記憶體、時間戳
- `iters.csv`：每 10 個 iteration 的 loss 與 lr
- `predictions/epoch_N.csv`：每張測試圖的 GT／預測角度與誤差（可畫誤差分布、散佈圖）
- `epoch_N.pkl`、`best.pkl`、`last.pkl`、`config.json`、`best.json`、`done.json`
- `scripts/make_report.py` 會把所有 run 彙整成 `results/all_runs.csv`、`all_epochs.csv`、`all_iters.csv` 與曲線圖。
