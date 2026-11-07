# REMOTE_PLAN — 在國網（TWCC / 台灣杉二號）上跑完整 L2CS-Net 實驗

> 這份文件是給**遠端機器上的 Claude** 照著執行的完整計畫。本機（WSL2、8 GB 筆電 GPU）一個 epoch 要 8.5 分鐘，
> 論文協定（15 fold × 50 epoch）要 100+ 小時，所以把訓練搬到國網。所有東西都在這個 repo：程式、腳本、
> 先前在本機跑出來的結果（`results/`），以及這份計畫。資料集與權重**不在** repo 裡，照第 2 節取得。
>
> 執行者請：**先完整讀完這份文件再動手**；每個階段結束前對照「驗收」；任何偏離計畫的地方都寫進最後的報告。

---

## 0. 貼給 Claude 的起始指令（複製這段）

```
請讀 REMOTE_PLAN.md 並照它執行。先做第 1 節的環境偵測並把結果告訴我，然後依序做第 2～6 節。
訓練請放在 tmux 裡跑，不要讓 SSH 斷線中斷訓練。每個階段的驗收項目都要實際檢查。
權重檔不要 commit。遇到計畫沒寫到的狀況，選最保守、可續跑的做法，並記在 results/remote/NOTES.md。
```

---

## 1. 背景、目標、成功標準

### 1.1 我們已經有什麼（本機做完的）
| 項目 | 位置 | 說明 |
|---|---|---|
| 訓練腳本 | `scripts/train_mpii.py` | 一次跑一個 leave-one-person-out fold，每個 epoch 在測試者上評估並記錄；支援 `--resume`（`state.pt`）、`--amp`（bf16 或 fp16 自動選）、`--recipe paper|improved` |
| 前處理 | `scripts/prep_mpiifacegaze.py` | MPIIFaceGaze 原始檔 → GazeHub 格式（224×224 正規化臉），15 人 37,667 張 |
| 彙整 | `scripts/make_report.py` | 把所有 run 彙整成 CSV、曲線圖、RESULTS.md |
| 多 GPU 排程 | `scripts/run_folds.sh` | 一個 GPU 一個 process，fold 輪流分配；重跑同一指令 = 續跑 |
| 本機結果 | `results/`、`README.md`「自己訓練 vs 論文」「第二輪」 | 5 個測試者（p00/p03/p07/p11/p14）各 7 epoch：paper @ep7 平均 4.22°，improved @ep7 平均 4.10°；論文同 5 人 4.14°（`results/slides_charts/paper_fig2_per_subject.json`） |
| 論文數字 | — | MPIIFaceGaze 15-fold 平均 **3.92°**（Adam 1e-5、batch 16、50 epoch、無增強）；Gaze360 10.41° |

### 1.2 這次要得到什麼（依優先順序）
1. **實驗 A `paper_e50`**：論文配方、**15 個 fold 全跑、每個 50 epoch**。這是可以直接和 3.92° 比較的數字。
2. **實驗 B `improved_e50`**：改良配方（cosine lr 1e-5→1e-7 + 輕量增強）同樣 15 fold × 50 epoch。本機 7 epoch 時改良配方「還沒學完」，要在同樣預算下才能下結論。
3. **實驗 C 消融（選做）**：只 cosine（`--recipe improved --no-aug`）、只增強（`--recipe improved --sched const`），各跑 5 個 fold（0、3、7、11、14）× 50 epoch，釐清 p03 變差是哪一項造成。
4. **交付**：`results/remote/` 裡的 CSV／圖／`RESULTS_REMOTE.md`，commit 到 branch `remote-results` 並 push；權重留在遠端並打包成 tar 供下載。

### 1.3 成功標準
- A 的 15 個 run 都有 `done.json`，`results/remote/all_runs.csv` 有 15 列 `paper`。
- 報告裡同時給「最後一個 epoch（@ep50）」與「最佳 epoch」兩種平均；@ep50 才是不看測試集就能拿到的誠實數字。
- 有一張 15 人逐人長條圖：我們的 paper @ep50、improved @ep50、論文 Fig. 2。
- 可重現性檢查：遠端 `fold0_paper_e50` 的 @ep7 應該和本機 `fold0_e15` 的 @ep7（2.78°）在 ±0.3° 內；差太多代表環境有問題（dtype、資料、前處理）。

---

## 2. 第一階段：環境偵測與建置（預計 30–60 分鐘）

### 2.1 先弄清楚是哪一種機器
```bash
hostname; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
which sbatch sinfo 2>/dev/null && sinfo -s        # 有 → 台灣杉二號 (Slurm)；沒有 → TWCC 容器 (CCS)
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
df -h ~ /work 2>/dev/null; nproc; free -g
```
| 情況 | 怎麼做 |
|---|---|
| **TWCC 開發型容器（CCS）**，有 GPU、可直接 `python` | 用容器自帶的 torch（NGC pytorch 映像通常已有 CUDA 版 torch）。訓練直接在 tmux 裡跑 `scripts/run_folds.sh`。 |
| **台灣杉二號（Slurm）** | 在 login node 建環境、下載資料（compute node 可能沒有網路）；訓練用 `sbatch scripts/remote/slurm_folds.sbatch`。 |
| GPU 是 **V100**（sm_70） | 沒有原生 bf16；`train_mpii.py --amp` 會自動改用 **fp16 + GradScaler**，config.json 的 `amp_dtype` 會寫 `fp16`。 |
| GPU 是 **A100 / H100 / RTX**（sm_80+） | 自動用 bf16，和本機一致。 |

### 2.2 取得程式
```bash
cd ~  # 或 /work/$USER（看 df -h 哪個空間大，資料與 run 都放那裡）
git clone --recursive https://github.com/YichiaHuang/dms-l2cs.git DMS
cd DMS && git submodule status          # L2CS-Net 必須有 commit hash（不是 "-" 開頭）；若是空的：git submodule update --init
```

### 2.3 建環境
```bash
bash scripts/remote/setup_env.sh
```
- 有 CUDA torch 就只裝 `requirements.txt`；沒有就建 conda env `l2cs`（`TORCH_INDEX` 可改 cu118/cu124/cu128）。
- 會把 ImageNet ResNet-50 權重預先下載到 `~/.cache/torch`（compute node 沒網路時需要）。
- `face_detection` 裝不起來沒關係，只有推論 demo 用得到。

### 2.4 取得資料（二選一）
**(a) 遠端直接下載並前處理（建議，約 15–30 分鐘）**
```bash
DATA_ROOT=~/datasets bash scripts/remote/get_data.sh     # 或 DATA_ROOT=/work/$USER/datasets
```
**(b) 從本機把處理好的資料集傳過去（923 MB、約 11 萬個小檔；先 tar 再傳）**
```bash
# 在本機 WSL：
tar -C ~/datasets -cf - MPIIFaceGaze | ssh <remote> 'mkdir -p ~/datasets && tar -C ~/datasets -xf -'
```
驗收：`ls ~/datasets/MPIIFaceGaze/Label | wc -l` = 15；`find ~/datasets/MPIIFaceGaze/Image -path '*/face/*.jpg' | wc -l` = 37667。

> 如果資料放在網路檔案系統（TWCC HFS、/work）而 smoke test 顯示 GPU 使用率低、img/s 很差，把整個 `MPIIFaceGaze/` 複製到節點本機磁碟（例如 `/tmp` 或 `$SLURM_TMPDIR`）再用 `DATA=` 指過去。

### 2.5 Smoke test（必做）
```bash
export DATA=~/datasets/MPIIFaceGaze RUNS=~/runs/l2cs      # 之後所有指令都用同一組
bash scripts/remote/smoke_test.sh
```
驗收：印出 `resume OK`、`smoke test OK`，並記下 **img/s、amp_dtype、GPU 名稱**。
再用這個數字算每個 epoch 的時間：**epoch 分鐘 ≈ 34,740 / (img/s) / 60**（本機 68 img/s → 8.5 分鐘）。
若 img/s < 120 且 `nvidia-smi` 的 GPU-Util 長期 < 60%，是 dataloader 吃不飽：`--workers` 加到每 GPU 6–8（看 `nproc`），或把資料搬到本機磁碟（2.4 的註）。

---

## 3. 第二階段：依實測速度決定排程

令 `m` = 每 epoch 分鐘、`G` = 可用 GPU 數。一個 15-fold × 50-epoch 實驗的牆鐘時間 ≈ `ceil(15 / G) × 50 × m / 60` 小時。

| GPU | 預期 img/s | m | G=4 時 A 或 B 各需 | 建議 |
|---|---|---|---|---|
| V100 32 GB（fp16） | 200–300 | 2–3 min | 7–10 h | A、B 都跑 50 epoch |
| A100 / H100（bf16） | 400–800 | 0.7–1.5 min | 2.5–5 h | A、B、C 全跑 |
| 單張 GPU | 同上 | 同上 | ×4 | 先跑 A；B 視配額 |

決策規則（照順序）：
1. **A 永遠跑滿 15 fold × 50 epoch**。若 A 預估 > 24 小時牆鐘且配額不夠，才改 25 epoch（tag 改 `paper_e25`），並在報告註明。
2. A 排完之後排 B（同樣 50 epoch）。配額真的不夠時 B 先跑 5 個 fold（`--folds 0,3,7,11,14`），能和本機成對比較。
3. C 只在 A、B 都完成且還有配額時跑。

磁碟：每個 run 約 0.5 GB（`best.pkl`、`last.pkl`、`state.pt`，**不要加 `--save-every-epoch`**，否則 15 × 50 × 92 MB = 69 GB）。A+B+C 共 40 run ≈ 20 GB。先 `df -h` 確認。

---

## 4. 第三階段：執行訓練

### 4.1 TWCC 容器（直接跑）
```bash
tmux new -s l2cs          # 之後用 tmux attach -t l2cs 回來
export DATA=~/datasets/MPIIFaceGaze RUNS=~/runs/l2cs
cd ~/DMS
# 實驗 A
bash scripts/run_folds.sh --tag paper_e50 --recipe paper --epochs 50 --gpus 0,1,2,3 --workers 6
# 實驗 B（A 結束後自動接著跑：把兩行寫在同一個 tmux 視窗，用 ; 或 && 串起來）
bash scripts/run_folds.sh --tag improved_e50 --recipe improved --epochs 50 --gpus 0,1,2,3 --workers 6
# 實驗 C（選做）
bash scripts/run_folds.sh --tag abl_cosine_e50 --recipe improved --epochs 50 --folds 0,3,7,11,14 --gpus 0,1,2,3 --extra "--no-aug"
bash scripts/run_folds.sh --tag abl_aug_e50    --recipe improved --epochs 50 --folds 0,3,7,11,14 --gpus 0,1,2,3 --extra "--sched const"
```
`--gpus` 依實際 GPU 數調整（`nvidia-smi -L`）。

### 4.2 台灣杉二號（Slurm）
先把 `scripts/remote/slurm_folds.sbatch` 的 `-A MST1XXXXX`、`-p`、`--gres`、`-t` 改成帳號實際可用的值（`sacctmgr show assoc user=$USER format=account,partition`、`sinfo`）。
```bash
sbatch --export=ALL,TAG=paper_e50,RECIPE=paper,EPOCHS=50    scripts/remote/slurm_folds.sbatch
sbatch --export=ALL,TAG=improved_e50,RECIPE=improved,EPOCHS=50 scripts/remote/slurm_folds.sbatch
# 消融（fold 清單用 / 分隔，因為 --export 以逗號切欄位；含空白的 EXTRA 傳不過去就直接改 sbatch 檔的預設值）：
sbatch --export=ALL,TAG=abl_cosine_e50,RECIPE=improved,EPOCHS=50,FOLDS=0/3/7/11/14,EXTRA="--no-aug"     scripts/remote/slurm_folds.sbatch
sbatch --export=ALL,TAG=abl_aug_e50,RECIPE=improved,EPOCHS=50,FOLDS=0/3/7/11/14,EXTRA="--sched const" scripts/remote/slurm_folds.sbatch
```
牆鐘上限不夠一次跑完時，**重送同一個 job 即可續跑**（done.json 跳過、state.pt 續跑）。

### 4.3 run 命名與輸出
- run 目錄：`$RUNS/fold<N>_<tag>/`，內含 `config.json`、`metrics.csv`（每 epoch）、`iters.csv`（每 10 iter）、`predictions/epoch_N.csv`、`best.json`、`done.json`、`best.pkl`、`last.pkl`、`state.pt`。
- log：`$RUNS/fold<N>_<tag>.log`；排程狀態：`$RUNS/queue/<tag>_status.txt`；全部完成會有 `$RUNS/queue/<tag>.done`。

### 4.4 監控（每隔幾小時看一次）
```bash
tail -n 20 ~/runs/l2cs/queue/paper_e50_status.txt
for f in ~/runs/l2cs/fold*_paper_e50/metrics.csv; do echo "$f: $(tail -n1 $f | cut -d, -f1,7,15,16)"; done   # epoch, test_err, 秒/epoch, img/s
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
grep -il "error\|traceback\|nan" ~/runs/l2cs/*.log
```
正常的樣子：
- epoch 0（未訓練）誤差很大是正常的；epoch 1 之後 paper 配方多數人就落在 3–5°，之後在 ±0.5° 內震盪；improved 配方應該單調下降、曲線平滑。
- 訓練 loss：paper 配方第 7 epoch 約 4，improved 約 7–8 並持續下降，到 50 epoch 應該明顯更低。
- fp16 時 loss 出現 NaN：GradScaler 會自動跳過該步並縮小 scale；若 `metrics.csv` 整個 epoch 都是 NaN，改 `--amp-dtype bf16`（若 GPU 支援）或 `--no-amp` 重跑該 fold（刪掉該 run 目錄）。

異常處理：
- 某個 fold crash：看 `$RUNS/fold<N>_<tag>.log` 最後 50 行；修好後**重跑同一個 run_folds.sh 指令**即可續跑。
- 容器／job 被中止：一樣重跑同一個指令。
- 永遠不要用 `pkill -f train_mpii`（會殺到 Claude 自己的 shell），要用 `pkill -f '[t]rain_mpii'`。

---

## 5. 第四階段：彙整與報告

全部（至少 A）完成後：
```bash
cd ~/DMS
python scripts/make_report.py --runs ~/runs/l2cs --out results/remote      # 產生 all_runs.csv / all_epochs.csv / all_iters.csv / curves_*.png / RESULTS.md
rm -rf results/remote/smoke 2>/dev/null
```
然後**自己寫** `results/remote/RESULTS_REMOTE.md`（zh-TW，給課程報告用），至少包含：

1. **硬體與設定**：GPU 型號／數量、torch 版本、amp_dtype、每 epoch 分鐘、img/s、總 GPU 小時。
2. **表 1：15 人逐人表**，欄位：測試者、paper @ep50、paper 最佳(epoch)、improved @ep50、improved 最佳(epoch)、論文 Fig. 2（讀 `results/slides_charts/paper_fig2_per_subject.json`）；最後一列 **平均 ± 標準差**。
3. **表 2：和論文的比較**：論文 3.92° vs 我們 paper @ep50 平均 vs paper 最佳 epoch 平均 vs improved 兩種；差距以度與百分比表示。說明「最佳 epoch 是用測試者本人挑的，偏樂觀」。
4. **表 3：可重現性**：本機 5 人 @ep7（`results/all_runs.csv` 的 `err_at_ep7`）vs 遠端同配方 @ep7（`results/remote/all_runs.csv`）。差異 > 0.3° 的要解釋（dtype、GPU、資料載入順序等）。
5. **圖**：(a) 15 人長條圖（paper @ep50、improved @ep50、論文 Fig. 2 三組）；(b) `curves_all.png`／逐人曲線（make_report 產生）；(c) paper vs improved 成對差異圖（improved − paper，每人一根，標平均）。畫圖腳本放 `scripts/make_remote_charts.py`，用 matplotlib，不要互動視窗（`matplotlib.use("Agg")`）。
6. **消融（若有做）**：5 人 × {paper, cosine-only, aug-only, improved} @ep50 表。
7. **與論文不同之處**（照實列）：bf16/fp16、前處理是自己移植的 GazeHub、layer1-4 BN 固定 eval、有做 @ep50 與最佳 epoch 兩種報告、其他任何偏離。
8. **結論三句話**：我們 15-fold 的數字是多少、和 3.92 差多少、改良配方在 50 epoch 下有沒有贏。

### 5.1 Commit 與 push（不含權重）
```bash
cd ~/DMS
git config user.name "Yichia Huang"; git config user.email "yichiahuang1@gmail.com"
git checkout -b remote-results
git add results/remote scripts/make_remote_charts.py
git status           # 確認沒有 .pkl / .pt / 資料集；.gitignore 已擋，但要看一眼
git commit -m "Add TWCC 15-fold results (paper_e50, improved_e50)"
git push -u origin remote-results
```
push 需要遠端能寫這個 repo：用 `ssh-keygen -t ed25519` 產生金鑰、把公鑰加到 GitHub（Settings → SSH keys，或 repo 的 Deploy key 勾 write），remote 改成 `git@github.com:YichiaHuang/dms-l2cs.git`。沒辦法 push 時，把 `results/remote` 打包成 tar 放在家目錄並在報告裡說明。

### 5.2 權重打包（留在遠端，使用者之後 scp）
```bash
mkdir -p ~/runs/l2cs/export
for d in ~/runs/l2cs/fold*_paper_e50 ~/runs/l2cs/fold*_improved_e50; do n=$(basename $d); cp $d/best.pkl ~/runs/l2cs/export/${n}_best.pkl; cp $d/best.json ~/runs/l2cs/export/${n}_best.json; done
tar -C ~/runs/l2cs -czf ~/l2cs_mpii_best_weights.tgz export     # 約 30 × 92 MB ≈ 2.8 GB
ls -lh ~/l2cs_mpii_best_weights.tgz
```
`state.pt` 在 run 都完成後可以刪掉省空間（`rm ~/runs/l2cs/fold*_e50/state.pt`），`best.pkl`／`last.pkl` 保留。

---

## 6. 最後回報格式（寫在對話裡，也存成 `results/remote/NOTES.md`）
- 機器類型、GPU、dtype、每 epoch 分鐘、每個實驗的牆鐘時間與 GPU 小時。
- 哪些實驗完成、哪些沒完成及原因。
- 表 2 的數字（15-fold 平均 @ep50 與最佳）。
- 可重現性檢查結果。
- 所有偏離計畫的地方與臨時決定。
- 權重 tar 的路徑與大小、push 的 branch 名稱。

---

## 附錄 A：規則
- 不改訓練配方與模型（`train_mpii.py` 的 loss、bins、lr、batch、BN 行為都不要動）；要改只能加旗標並預設維持原行為，並在報告裡說明。
- 不 commit `*.pkl`、`*.pt`、`*.safetensors`、資料集、`runs/`。
- 所有長時間程序都在 tmux 或 Slurm 裡；每次 SSH 回來先 `tmux attach -t l2cs`。
- 執行 Python 用 `python -I`（避免從目前目錄載入同名模組）。
- 每做完一個階段就把狀態寫進 `results/remote/NOTES.md`（時間戳、做了什麼、數字）。

## 附錄 B：檔案導覽
| 檔案 | 用途 |
|---|---|
| `scripts/train_mpii.py` | 訓練＋每 epoch 評估（單 fold） |
| `scripts/run_folds.sh` | 多 GPU fold 排程、續跑 |
| `scripts/remote/setup_env.sh` | 環境建置（conda 或容器自帶 torch） |
| `scripts/remote/get_data.sh` | 下載 + 前處理 MPIIFaceGaze |
| `scripts/remote/smoke_test.sh` | 5 分鐘 end-to-end 檢查、量 img/s |
| `scripts/remote/slurm_folds.sbatch` | 台灣杉二號 job 範本 |
| `scripts/make_report.py` | 彙整所有 run → CSV／圖／RESULTS.md |
| `scripts/eval_mpii.py` | 用任一權重在 MPII 上評估（含 Gaze360 權重 zero-shot） |
| `scripts/prep_mpiifacegaze.py` | GazeHub 前處理移植 |
| `results/` | 本機先前的結果（勿覆寫；新結果放 `results/remote/`） |
| `PLAN.md`、`README.md` | 本機兩輪實驗的計畫與紀錄，背景知識 |
| `L2CS-Net/` | 官方 repo（git submodule，commit a4d8f7f），`l2cs` 套件由 `train_mpii.py` 以 sys.path 匯入 |

## 附錄 C：已知的坑
- 官方 `train.py`／`test.py` 的 mpiigaze 分支跑不起來，所以才有 `train_mpii.py`；不要嘗試用官方的。
- repo 裡 `pitch`／`yaw` 命名相反：label[0]（叫 pitch）其實是水平角。增強的水平翻轉就是對它取負。
- 官方 Gaze360 權重的 Google Drive 連結已失效；`run_l2cs.py` 會從 HuggingFace `py-feat/l2cs` 的 safetensors 轉檔。訓練用不到這份權重。
- `torch.hub.load_state_dict_from_url` 需要網路，compute node 沒網路時先在 login node 跑 `setup_env.sh` 預下載。
- unzip 可能沒裝；`get_data.sh` 用 python zipfile 解壓。
- OneDrive 路徑的問題只存在本機，遠端沒有。
