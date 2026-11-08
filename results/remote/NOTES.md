# NOTES — 遠端執行紀錄（Nano4）

> 依 REMOTE_PLAN.md 執行。所有偏離計畫之處與臨時決定都記在這裡（時間為 Asia/Taipei）。

## 2026-11-07 15:00 第 1 節：環境偵測
- 機器：**Nano4（NCHC 晶創26）**，Slurm 叢集，不是計畫假設的 TWCC 容器或台灣杉二號。login node `25a-lgn02`（x86_64、無 GPU、每使用者 20 GB cgroup）。
- Compute node：`25a-hgpn*`，**8× NVIDIA H200 141 GB**，driver 580.65.06（CUDA 13.0），每節點 112 CPU、每 GPU 最多配 12 CPU。sm_90 → `--amp` 自動用 **bf16**（和本機一致）。
- 佇列：`dev`（4 h，smoke/前處理用）、`8gpus`（2 天，正式訓練用）。GB200 節點（`25a-ggpn*`，aarch64）不用：x86 venv 不能在那邊跑。
- 帳號：預設帳號 `acd114003` 被拒（`Project acd114003 is not allowed to use NANO4`）→ 全部改用 **`acd110018`**。
- 網路：login node 與 compute node 都可以連外（GitHub、download.pytorch.org）。
- 磁碟：`/home` 100 GB（剩 87 GB）、`/work` 500 GB（剩 498 GB）→ 資料與 run 放 `/work/u8714813`；repo 留在 `~/DMS`。

## 偏離計畫 / 臨時決定
1. **路徑**：`DATA=/work/u8714813/datasets/MPIIFaceGaze`、`RUNS=/work/u8714813/runs/l2cs`（計畫寫 `~/datasets`、`~/runs`），原因是 /work 空間大很多。
2. **Python 環境**：機器沒有 conda/module；下載後執行 Miniforge 安裝腳本被權限擋下 → 改用系統 Python 3.9 建 venv `/work/u8714813/envs/l2cs`，pip 裝 `torch 2.8.0+cu128`、`torchvision 0.23.0+cu128`、`requirements.txt`、`pip install -e L2CS-Net --no-deps`。沒有跑 `setup_env.sh` 本身（它在無 GPU 的 login node 會判斷錯走 conda 分支），改成手動做同樣的步驟；ResNet-50 ImageNet 權重已預先下載到 `~/.cache/torch/hub`。`face_detection` 沒裝（只有推論 demo 用得到）。
3. **Slurm 範本**：新增 `scripts/remote/slurm_folds_nano4.sbatch`（`-A acd110018 -p 8gpus --gres=gpu:H200:8 --cpus-per-task=96 -t 2-00:00:00`），原本的台灣杉二號範本保留不動。EXTRA 裡的空白用 `_` 傳（例如 `EXTRA=--sched+const`）。
4. **tmux**：訓練以 sbatch job 執行（不依附 SSH）；另開 tmux session `l2cs` 放監控與「job 結束就重送續跑」的迴圈。
5. 暫存目錄一律用 `/work/u8714813/tmp`（700），不寫共用 `/tmp`。

## 2026-11-07 15:05–15:20 第 2 節：建置、資料、smoke test
- 資料：login node 下載 zip（942 MB，testzip 無錯），在 `dev` job 720574 解壓 + 前處理。**驗收：Label 15 個、face jpg 37,667 張** ✔（p00 測試 2,927、p11 2,262，和本機一致）。
- `face_detection` 無法安裝（從 GitHub 用 pip 裝被權限擋下）。`l2cs/__init__` 會匯入 `pipeline.py → face_detection`，所以 `train_mpii.py` 新增：**import 失敗時放一個只在被呼叫時報錯的 stub**。訓練從不使用 RetinaFace，loss／模型／配方完全沒動。
- Smoke test（job 720578，1×H200）：`resume OK`、`smoke test OK`，**amp_dtype = bf16，GPU = NVIDIA H200，約 665 img/s** → m ≈ 0.87 min/epoch（只算訓練）。
- **第一次送 A（job 720583）失敗**：8 個 fold 同時在 eval 開 DataLoader 時 `OSError: [Errno 12] Cannot allocate memory`（fork）。原因：compute node `vm.overcommit_memory=2`（CommitLimit ≈ 1.68 TB），每個 worker fork 都會複製母行程的 commit 量；8 個行程 × (8 train + 8 test workers) 超過上限。處理：取消 A，hold B/C 失敗所以一併取消；刪掉這次只跑 3 分鐘的 paper_e50 run 目錄（部分 fold 沒有 state.pt），從頭重跑。
- 8 卡記憶體測試（dev job 720592，`--workers 4`、600 iter × 3 epoch）：沒有錯誤，Committed_AS 峰值 1.415 TB／1.68 TB（84%，而且是 8 個 fold 同步 eval 的最壞情況）→ **正式訓練用 `--workers 4`**（計畫寫 6）。每個行程約 470–515 img/s。

## 第 3 節：排程決策
- 實測（A 正式 run）：每 epoch 約 68–75 s 訓練 + eval。A：ceil(15/8)=2 輪 × 50 epoch ≈ 2.2 h 牆鐘，遠低於 24 h → **A、B、C 全跑 50 epoch**。
- Job：A=720605（8gpus，8×H200）→ B=720606（`afterany:A`）→ C1=720607 / C2=720608（`afterany:B`，各 5×H200，fold 0/3/7/11/14）。tmux session `l2cs` 裡的 `scripts/remote/watchdog_nano4.sh` 每 10 分鐘檢查一次：tag 還沒 `.done` 而 job 不見了就重送（續跑），每個 tag 最多 3 次。
- 注意：`run_folds.sh` 在 status 檔裡寫的 `exit=` 不可靠（第一次失敗時 crash 的 fold 也寫 exit=0）→ 完成與否一律以 `done.json` 判斷。

## 2026-11-07 15:30 早期可重現性檢查（A 跑到 ep7 時）
- fold0 paper @ep7：遠端 **2.80°** vs 本機 `fold0_e15` 2.78° → +0.03° ✔（±0.3° 內）。fold3 2.93 vs 3.03（−0.10）、fold7 4.95 vs 5.04（−0.09）。
- 每 epoch（訓練 + eval）約 1.3 min；預估 A 約 17:40 結束、B 約 20:05、C 約 21:20。

## 2026-11-07 15:55 使用者要求控制 quota 用量
- Nano4 計費：`billing` = 每 GPU 12（按卡時計）；`acd110018` 是多人共用的計畫帳號，Slurm 上沒有 GrpTRESMins 上限（真正的錢包在 iService，CLI 看不到）。
- 原排程估計：A 約 19 GPU-h、B 約 19 GPU-h、C 約 12 GPU-h；測試／smoke／失敗的第一次 A 已用約 0.7 GPU-h。
- 處理：**取消選做的 C（job 720607/720608，還沒開始跑）**，watchdog 也拿掉 C；watchdog 重送上限 3 → 1。A 照跑（必做）；B 是否縮成 5 fold 等使用者決定。
- 使用者決定：**B 照原計畫 15 fold × 50 epoch**（job 720606，afterany:A）。C 維持取消。預估總用量 ≈ A 19 + B 19 + 已用 0.7 ≈ 39 GPU-h。

## 2026-11-07 17:11 A（paper_e50）完成
- job 720605，牆鐘 **1:53:16**（8×H200）→ 約 15.1 GPU-h（計費以 8 卡算）。
- 驗收：15/15 `done.json`、皆到 epoch 50、metrics.csv 無 NaN、log 無 Traceback、amp_dtype=bf16 ✔。每 run 約 0.46 GB。
- B（job 720606）17:11 自動開始，預估約 19:05 完成。

## 2026-11-07 19:58 B（improved_e50）完成
- job 720606，牆鐘 **2:47:00**（8×H200）→ 約 22.3 GPU-h。比 A 慢約 50%，因為增強是在 CPU 的 DataLoader 裡做的（385 vs 586 img/s）。
- 驗收：15/15 `done.json`、皆到 epoch 50、無 NaN、無 Traceback、sched=cosine、aug=true、amp_dtype=bf16 ✔。
- 表 3 修正：本機 `*_improved_e7` 的 cosine 排程是 T=7（ep7 時 lr=1e-7），遠端是 T=50（ep7 時 lr=9.5e-6）→ improved 的 @ep7 不能比，只比 paper 配方。

---

# 第 6 節：最後回報

**機器**：NCHC Nano4（Slurm），`25a-hgpn022`，8× NVIDIA H200 141 GB；torch 2.8.0+cu128；**bf16**。帳號 `acd110018`（預設帳號 acd114003 不能用 Nano4）。

**速度**：paper 1.00 min／epoch（586 img/s／行程）；improved 1.52 min／epoch（385 img/s）。

**完成狀況**
| 實驗 | 狀態 | 牆鐘 | GPU-h |
|---|---|---|---|
| A paper_e50（15 fold × 50） | ✔ 15/15 | 1:53 | 15.1 |
| B improved_e50（15 fold × 50） | ✔ 15/15 | 2:47 | 22.3 |
| C 消融 | ✘ 未執行：使用者要求控制 quota，選做項目取消 | — | — |
| 測試／smoke／記憶體測試／失敗的第一次 A | — | — | 約 0.8 |
| **合計** | | | **約 38 GPU-h** |

**表 2（15-fold 平均，度）**
| | @ep50 | 最佳 epoch（偏樂觀） |
|---|---|---|
| 論文 | 3.92 | — |
| paper 配方 | **4.70 ± 0.96**（+0.78，+20.0%） | 3.53 ± 0.66 |
| improved 配方 | **4.15 ± 1.17**（+0.23，+5.9%） | 3.38 ± 0.69 |
improved − paper @ep50：平均 −0.55°，12/15 人改善，配對 t 檢定 p = 0.017。paper 配方過擬合：15 人平均在 ep5 最低（4.05°）。

**可重現性**：fold0 paper @ep7 遠端 2.80° vs 本機 2.78°（+0.03°）✔。paper 配方 5 組裡 4 組 |Δ| ≤ 0.1°，p11 −0.32°（在 p11 自身逐 epoch 0.6° 的震盪範圍內）。

**偏離計畫與臨時決定**（細節見上方各節）
1. 機器是 Nano4，不是 TWCC 或台灣杉二號；新增 `scripts/remote/slurm_folds_nano4.sbatch` 和 `scripts/remote/watchdog_nano4.sh`（tmux `l2cs` 裡的續跑 watchdog）。
2. 資料和 run 放 `/work/u8714813`（不是 `~`）。
3. 沒有 conda，Miniforge 安裝被權限擋下 → 改用系統 Python 3.9 venv + pip 安裝 torch cu128。
4. `face_detection` 沒裝（從 GitHub 用 pip 裝被擋）→ `train_mpii.py` 在沒有這個套件時放 stub，配方沒動。
5. 每 GPU 4 workers（計畫寫 6），原因是嚴格 overcommit 讓 fork 出現 ENOMEM；第一次 A 只跑 3 分鐘就失敗，已刪掉重跑。
6. 實驗 C 取消（quota）；watchdog 重送上限 1 次。
7. `state.pt` 沒刪（/work 空間足夠，留著最保守）；run 目錄共 17 GB。

**產出**
- 權重：`~/l2cs_mpii_best_weights.tgz`（**2.5 GB**，30 個 `best.pkl` + `best.json`）；完整 run 在 `/work/u8714813/runs/l2cs/`。
- 結果：`results/remote/`（RESULTS_REMOTE.md、表、圖、CSV），branch **`remote-results`**。
- 補跑 C 的指令（約 12 GPU-h）：
  `sbatch -J l2cs-C1 -A acd110018 --gres=gpu:H200:5 --cpus-per-task=60 --export=ALL,TAG=abl_cosine_e50,RECIPE=improved,EPOCHS=50,FOLDS=0/3/7/11/14,GPUS=0/1/2/3/4,EXTRA=--no-aug scripts/remote/slurm_folds_nano4.sbatch`
  `sbatch -J l2cs-C2 -A acd110018 --gres=gpu:H200:5 --cpus-per-task=60 --export=ALL,TAG=abl_aug_e50,RECIPE=improved,EPOCHS=50,FOLDS=0/3/7/11/14,GPUS=0/1/2/3/4,EXTRA=--sched+const scripts/remote/slurm_folds_nano4.sbatch`
