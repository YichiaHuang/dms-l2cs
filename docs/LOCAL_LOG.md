# 早期本機紀錄（2026-09 ～ 10，已由 README.md 取代為總覽）

> 題目：智慧交通 #4 Driver Monitoring Systems (DMS) for ADAS — L2CS-Net（ICIP 2022, arXiv:2203.03339）
> 建置日期：2026-09-21，WSL2 / Ubuntu 22.04，RTX PRO 1000 Blackwell (sm_120, 8 GB)

## Repo 結構與搬到國網訓練

這個資料夾就是 GitHub repo `YichiaHuang/dms-l2cs`（2026-11-07 建立）。`L2CS-Net/` 是官方 repo 的 git submodule，clone 時要加 `--recursive`。
資料集、權重、run 目錄都不在 repo 裡（`.gitignore`）；本機放在 Linux 端 `~/datasets`、`~/runs/l2cs`。

要在國網（TWCC / 台灣杉二號）跑完整 15-fold × 50-epoch 實驗：看 **`REMOTE_PLAN.md`**，裡面有給遠端 Claude 照做的完整步驟、
腳本（`scripts/remote/`、`scripts/run_folds.sh`）、驗收標準與回報格式。

## 環境（已建好：conda env `l2cs`）

```bash
conda create -n l2cs python=3.10 -y -c conda-forge --override-channels   # 同 Week 2：避開 Anaconda ToS
conda activate l2cs
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128   # Blackwell 需要 cu128
cd DL/DMS/L2CS-Net && pip install -e . safetensors      # 會一併裝 face_detection (RetinaFace)
```

## 執行

```bash
conda activate l2cs
cd /mnt/c/Users/yichi/Desktop/DL/DMS
python run_l2cs.py -i ../Week2/yolov7/inference/images/zidane.jpg -o out/zidane.jpg
python run_l2cs.py -i <影片.mp4> -o out/result.mp4 --csv out/result.csv     # 逐幀 pitch/yaw 記錄
```

`run_l2cs.py` 是自己寫的包裝，官方 `demo.py` 在這台電腦不能直接用，原因見下。

## 踩到的坑

| 問題 | 原因 | 處理 |
|---|---|---|
| 官方權重下載連結 404 | README 裡的 Google Drive 資料夾已失效 | 改用 py-feat 在 HuggingFace 重新託管的同一份 Gaze360 權重（`py-feat/l2cs`，safetensors 格式，不經 pickle）；`run_l2cs.py` 第一次執行會自動轉成 `models/L2CSNet_gaze360.pkl` |
| 第一幀卡住好幾分鐘（CPU 100%、GPU 0%） | `face_detection` 的 `load_net()` 強制開 `cudnn.benchmark=True`；在 sm_120 上 autotune 每種 conv 形狀要 6–8 秒，RetinaFace 有幾十種 | 建完 Pipeline 後設回 `torch.backends.cudnn.benchmark = False`（實測第一個 conv 8.4 s → 0.9 s） |
| 畫面沒有臉時 `Pipeline.step()` 直接 crash | `np.stack([])` 丟 ValueError | `step_safe()` 接住，回傳 None |
| `demo.py` 要 webcam + `cv2.imshow` | WSL2 預設沒有攝影機也沒有視窗 | 改成吃圖片／影片檔、輸出檔案 |
| repo 裡 `pitch`／`yaw` 命名與直覺相反 | `draw_gaze` 用 `pitchyaw[0]` 算水平位移，所以輸出的 "pitch" 其實是**水平**角 | 做分心判斷（左右看）時要用 `results.pitch`，先用自己的影片確認正負號 |

## 實測（2026-09-21）

- zidane.jpg：偵測到 2 張臉，視線箭頭方向與肉眼判斷一致。
- 速度：1280×720 約 0.68 s／幀（偵測 0.40 s + ResNet50 0.05 s／臉），**但這不是正常值**：當時 GPU 被鎖在 P4（1305／3090 MHz，15 W），4096² matmul 只有 0.9 TFLOPS；同一張卡 9/17 跑 YOLOv7 是 24.4 ms。重測前先確認 Windows 電源模式為「最佳效能」、筆電廠商的效能模式有開。

## 自己訓練 vs 論文（2026-10-15）

完整計畫與縮小規模的理由見 `PLAN.md`，數據與圖在 `results/`（`RESULTS.md`、`learning_curves.png`、`zeroshot_per_person.png`），權重與 log 在 Linux 端 `~/runs/l2cs/`。

### 做了什麼
- 資料集：MPIIFaceGaze 原始檔（942 MB）→ `scripts/prep_mpiifacegaze.py`（移植 GazeHub 前處理）→ `~/datasets/MPIIFaceGaze`，15 人 37,667 張 224×224 正規化臉。Gaze360 要向 MIT 申請授權，這輪沒做。
- 訓練：`scripts/train_mpii.py`，配方照論文（ImageNet ResNet-50、28 bin × 3°、CE + MSE、Adam 1e-5、batch 16、無增強），另加 bf16 autocast 與逐 epoch 評估。官方 `train.py`/`test.py` 的 mpiigaze 分支本身跑不起來（參數數量錯、DataParallel 後找不到 `conv1`、fold 被當 angle 傳入、`args.bins` 未定義）。
- 這台 8 GB 筆電 GPU 實測 68 img/s、一個 epoch 8.5 分鐘；論文協定 15 fold × 50 epoch 約 100 小時，所以只跑 fold 0（15 epoch）與 fold 7（8 epoch 後手動中斷）。

### 結果
| 設定 | 測試者 | 最佳 epoch | 最佳誤差 | 最後 epoch 誤差 |
|---|---|---|---|---|
| 論文（15 fold 平均，50 epoch） | 全部 | – | **3.92°** | – |
| fold 0，15 epoch | p00 | 6 | **2.61°** | 3.28°（ep 15） |
| fold 7，8 epoch（中斷） | p07 | 5 | **4.44°** | 4.64°（ep 8） |
| 官方 Gaze360 權重 zero-shot | 15 人平均 | – | 8.06° | – |

- 兩個 fold 平均 3.5°，與論文 3.92° 在同一個量級；單一 fold 不足以下「贏／輸」的結論，跨人差異（2.6° vs 4.4°）比與論文的差距還大。
- 1 個 epoch 就降到接近最終值，之後在 ±0.5° 內震盪，lr 1e-5 沒有 scheduler 的設定下繼續訓練沒有明顯收益；論文跑 50 epoch 可能是為了取最佳 epoch。
- Zero-shot 的 Gaze360 權重（DMS 目前實際在用）在 MPII 上誤差 8.06°，是 in-domain 訓練的 2 到 3 倍，而且 pitch 誤差明顯大於 yaw（p03、p11 的 pitch MAE 近 9°）。做 DMS 時若要判斷駕駛上下看，建議用近距離正面資料 fine-tune。

### 與論文不同之處
只跑 2/15 fold、epoch 15/8 而非 50、bf16 autocast、前處理用自己移植的 GazeHub 程式（which-eye 旗標改讀 MPIIFaceGaze 第 28 欄）、保留原版「layer1-4 的 BN 固定 eval」行為。

## 第二輪：論文配方 vs 改良配方（2026-10-21 跑完）

設計見 `PLAN.md` 第二節；所有數據在 `results/`：`all_runs.csv`（每個 run 一列）、`all_epochs.csv`（每個 epoch 一列）、`all_iters.csv`（每 10 iteration 一列）、`predictions/<run>_epoch7.csv`（每張測試圖的預測）、`curves_per_person.png`、`curves_all.png`、`RESULTS.md`。

改良配方 = cosine 學習率衰減（1e-5 → 1e-7）+ 輕量增強（色彩抖動、平移/縮放 ±5%、水平翻轉並把水平角取負）；其餘與論文相同。每個 run 7 epoch，5 個測試者成對比較。

| 測試者 | paper 最佳 | improved 最佳 | paper @ep7 | improved @ep7 |
|---|---|---|---|---|
| p00 | 2.61 | 2.43 | 2.78 | **2.43** |
| p03 | **2.79** | 3.68 | **3.03** | 4.10 |
| p07 | 4.44 | 4.35 | 5.04 | **4.35** |
| p11 | 4.53 | 4.38 | 5.53 | **4.73** |
| p14 | **3.88** | 4.22 | **4.75** | 4.91 |
| 5 人平均 | 3.65 | 3.81 | 4.22 | **4.10** |
| 論文（15 人平均，50 epoch） | 3.92 | | | |

### 怎麼解讀
- **「最佳 epoch」是用測試者本人挑的，偏樂觀**；報告時應以最後一個 epoch（@ep7）為準，那才是不看測試集就能拿到的數字。以這個標準，改良配方 5 人中 3 人較好，平均好 0.12°，而且曲線平穩、最後一個 epoch 通常就是最佳；論文配方則上下震盪 ±0.5°，最後一個 epoch 常常不是最好的。
- **改良配方沒有提高上限**：最佳 epoch 的平均反而差 0.16°，p03 明顯變差（3.03° → 4.10°）。原因看訓練 loss 就知道：有增強的 run 在第 7 個 epoch 的 train loss 還在 7–8、仍在下降，而無增強的已降到 4，加上 cosine 在 7 個 epoch 內就把 lr 降到 1e-7，模型其實還沒學完。增強配方需要更多 epoch 才划算。
- **與論文的差距**：5 人平均 4.1–4.2°，論文 15 人平均 3.92°。考慮只跑 7 epoch（論文 50）且這 5 人裡包含 zero-shot 最難的 p03、p11，結果在合理範圍內；要真正對齊需要跑滿 15 個 fold。
- 跨人差異（p00 2.4° vs p11 4.7°）仍然大於配方之間的差異，報告裡應以成對比較呈現，不要只看平均。

### 如果要再跑
- 改良配方拉長到 15–20 epoch（或 cosine 底設 1e-6），看上限是否超過論文配方。
- 做消融：只 cosine、只增強，各跑 2–3 個 fold，釐清 p03 變差是哪一項造成。
- 每個 run 都有 `state.pt`，`run_overnight.sh` 會自動跳過完成的 run、從中斷處續跑。

## 推論速度（2026-10-29 重測，取代上面 9/21 的 0.68 s／幀）

腳本：`scripts/bench_speed.py`（純模型、原版流程）、`scripts/bench_speed_optimized.py`（GPU 前處理，三種設定在同一幀輪流跑以排除 GPU 頻率造成的順序偏差）。原始數據在 `results/speed/`。這次沒有降頻：fp32 4096² matmul 3.0 TFLOPS（9/21 鎖頻時 0.9）。論文沒有報告推論速度。

| 完整流程（1280×720、1 張臉、145 幀） | 偵測 | 前處理 | 模型 | 每幀 | FPS |
|---|---|---|---|---|---|
| 原版 `Pipeline` | 41 ms | 38 ms | 13 ms | 92 ms | 10.9 |
| 前處理改在 GPU | 37 ms | 0.9 ms | 11 ms | 48 ms | 20.7 |
| GPU 前處理 + bf16 | 39 ms | 1.0 ms | 11 ms | 51 ms | 19.8 |

- 原版瓶頸是 `prep_input_numpy` 在 CPU 上的 ToTensor 與 Normalize（各約 10 ms）。改在 GPU 做之後輸出角度與原版平均差 0.15°（p99 0.53°）。
- batch 1 時 bf16 沒有幫助（受 kernel 啟動開銷限制）；純模型 batch 4 時 bf16 才有差：144 → 282 臉/秒。CPU 只有 0.8 臉/秒。
- 現在的瓶頸是 RetinaFace 偵測，約佔一幀的 76%。
