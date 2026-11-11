# DMS 視線估計專題（L2CS-Net）— 進度總覽

> 題目：智慧交通 #4 Driver Monitoring Systems（DMS）for ADAS。
> 目標：讓車上的小晶片也能判斷「駕駛在看哪裡」。
> 完整論文式報告：[`results/final/RESULTS_FINAL.md`](results/final/RESULTS_FINAL.md)。早期在筆電上的紀錄搬到 [`docs/LOCAL_LOG.md`](docs/LOCAL_LOG.md)。
> 最後更新：2026-11-11

---

## 一句話版本

我們重現了 L2CS-Net，並把它改成比原本小 19 倍、可以放上車用晶片的版本，準確度不變（4.15°）。但只要換一個資料集，誤差就會變成好幾倍。最新實驗發現：只要給模型看新環境每人 10 張照片（約 110 張），誤差就能從約 29° 降到約 13°，但會忘掉原本學會的東西。每人約 100 張、再混入 25% 原本的訓練資料，就能做到 12.5° 而且不忘。

---

## 1. 目前做到哪裡

| # | 做了什麼 | 結果（誤差越小越好，單位：度） | 狀態 |
|---|---|---|---|
| 1 | 照論文設定重現 L2CS-Net（MPIIFaceGaze，15 個人輪流當測試者） | **4.70°**（論文寫 3.92°） | ✅ |
| 2 | 改良訓練方式（資料增強 + 學習率慢慢降） | **4.15°**，比論文配方好，且有統計顯著性 | ✅ |
| 3 | 拆解改良到底是哪一招有用 | 資料增強有用；只調學習率反而變差 | ✅ |
| 4 | 換成小模型（MobileNetV3、EfficientNet）+ 知識蒸餾 | MobileNetV3 + 蒸餾 **4.01°**，運算量只有原本的 1/19 | ✅ |
| 5 | 量化成 int8（車用晶片只跑得動 int8） | MobileNetV3 **4.15°**，沒有變差；EfficientNet 掉到 7.7° | ✅ |
| 6 | 速度與車用晶片可行性 | 0.85 GMACs，2 TOPS 等級的晶片理論上 25 fps 以上 | ✅（未上板實測） |
| 7 | 在 Gaze360（大角度資料集）上重現 | **11.04°**（論文 10.41°） | ✅ |
| 8 | 跨資料集測試（不重新訓練） | 只用 MPII 訓練的模型：RT-GENE 15°、Gaze360 30° | ✅ |
| 9 | 「用大角度資料訓練小模型」能不能解決第 8 項 | **不行**：RT-GENE 反而變成 29°；事先定的 3 個標準都沒達成 | ✅（負面結果） |
| 10 | 用少量目標資料微調（RT-GENE 篩選，見第 4 節） | 3 組受試者平均：每人 10 張從 28.6° 降到 12.6°（比現在的車用模型好 2°），但會忘記大角度；混舊資料就不會忘，但只到 14.6° | ✅（12° 標準沒過） |
| 11 | 每人要收多少張、要混多少舊資料（R2，見第 4 節） | **每人 100 張 + 混 25% 舊資料：RT-GENE 12.5°，Gaze360／MPII 都沒變差**，事先定的標準全過（餘裕很小） | ✅ |

GPU 用量：累計約 96 GPU-h（NCHC Nano4，H200）。

---

## 2. 我們優化了什麼？為什麼這樣做？

以下每一項都附參考資料，方便寫報告時引用。

### 2.1 評估方式：只報「誠實數字」
- **做法**：一律用訓練結束時的模型來報告。不從測試集挑「最好的那個 epoch」。
- **原因**：論文配方會過擬合。從測試集挑最佳 epoch 會得到 3.53°，看起來比論文還好，但這是偷看答案。誠實數字是 4.70°。
- **參考**：評估協定與資料前處理照 GazeHub 的整理。
  - Cheng et al., *Appearance-based Gaze Estimation With Deep Learning: A Review and Benchmark*, IEEE TPAMI 2024（arXiv:2104.12668）。

### 2.2 訓練配方：資料增強 + cosine 學習率
- **做法**：
  - 色彩抖動與小幅平移縮放。
  - 左右翻轉，同時把水平視線角取負號。
  - 學習率從 1e-5 用 cosine 曲線降到 1e-7。
- **結果**：4.70° → 4.15°（p = 0.017）。消融實驗顯示功勞主要在資料增強；只用 cosine 反而讓後期過擬合更嚴重。
- **參考**：
  - Loshchilov & Hutter, *SGDR: Stochastic Gradient Descent with Warm Restarts*, ICLR 2017（cosine 學習率）。
  - 翻轉時把水平角取負，是視線估計常見的增強方式（同 GazeHub）。

### 2.3 輕量化：換小骨幹
- **做法**：把 ResNet-50 換成 MobileNetV3-Large 或 EfficientNet-B0，兩個預測頭（水平、垂直）不變。
- **結果**：MobileNetV3-L 為 0.85 GMACs，ResNet-50 為 16.35 GMACs，運算量是 1/19。
- **參考**：
  - Howard et al., *Searching for MobileNetV3*, ICCV 2019。
  - Tan & Le, *EfficientNet: Rethinking Model Scaling for CNNs*, ICML 2019。
  - He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016（原本的 ResNet-50）。

### 2.4 知識蒸餾（KD）：大模型教小模型
- **做法**：讓小模型除了學正確答案，也學 ResNet-50 輸出的機率分布（溫度 T = 2）。
- **結果**：MobileNetV3 從沒蒸餾的版本進步到 4.01°，是所有模型裡最好的。T = 4 沒有比較好。
- **參考**：Hinton, Vinyals & Dean, *Distilling the Knowledge in a Neural Network*, arXiv:1503.02531, 2015。

### 2.5 int8 量化：讓車用晶片跑得動
- **做法**：
  - PTQ：訓練完直接量化。
  - QAT：在模擬量化的情況下再微調 2 個 epoch，第 2 個 epoch 凍結量化參數與 BatchNorm。
- **結果**：
  - MobileNetV3 + 蒸餾經 QAT 為 4.15°，和 ResNet-50 fp32 一樣（p = 0.98）。
  - EfficientNet-B0 的 fp32 一樣好，但量化後壞掉（7.7°）。
  - **教訓：要用量化後的準確度選模型。**
- **參考**：
  - Jacob et al., *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*, CVPR 2018。
  - Krishnamoorthi, *Quantizing Deep Convolutional Networks for Efficient Inference: A Whitepaper*, arXiv:1806.08342, 2018（凍結 BN 與 observer 的做法）。

### 2.6 翻轉 TTA（測試時左右翻轉取平均）
- **結果**：多 0.04–0.05°，但推論時間變兩倍，車上不划算，只放在報告裡當參考。

### 2.7 推論管線加速（筆電）
- **做法**：前處理搬到 GPU。
- **結果**：一幀從 92 ms 降到 48 ms（10.9 → 20.7 fps）。現在的瓶頸是人臉偵測（RetinaFace），約佔 3/4 的時間。

### 2.8 跨資料集與大角度訓練
- **跨資料集**：用 Gaze360 與 RT-GENE 測泛化能力。
  - 先用官方權重確認我們的前處理正確（10.56°，論文 10.41°）。
  - 正負號與軸的方向每個資料集只決定一次，不會針對不同模型調整。
- **大角度訓練（G2／J1）**：用 Gaze360 加 MPII 一起訓練小模型。
  - 一個小模型同時涵蓋大角度（11.6°）與正面（5.3°）。
  - 沒有解決 RT-GENE（約 29°）。
- **參考**：
  - Kellnhofer et al., *Gaze360: Physically Unconstrained Gaze Estimation in the Wild*, ICCV 2019。
  - Fischer, Chang & Demiris, *RT-GENE: Real-Time Eye Gaze Estimation in Natural Environments*, ECCV 2018。
  - Zhang et al., *It's Written All Over Your Face: Full-Face Appearance-Based Gaze Estimation*, CVPRW 2017（MPIIFaceGaze）。
  - Abdelrahman et al., *L2CS-Net: Fine-Grained Gaze Estimation in Unconstrained Environments*, arXiv:2203.03339（本專題的基礎模型）。

---

## 3. 主要數字一覽

| 模型 | MPII（同資料集） | Gaze360 | RT-GENE | 運算量 | 可上車？ |
|---|---|---|---|---|---|
| L2CS-Net 論文 | 3.92 | 10.41 | — | 16.35 GMACs | 太大 |
| ResNet-50 改良配方 | 4.15 | 29.3（沒訓練過） | 15.0 | 16.35 | 只有 8 TOPS 以上晶片 |
| **MobileNetV3 + 蒸餾 + int8（目前推薦）** | **4.15** | 約 31（沒訓練過） | 約 15 | **0.85** | ✅ |
| J1：MobileNetV3，Gaze360 + MPII 一起訓練 | 5.27（留出 5 人） | **11.57** | 28.9 | 0.85 | ✅（int8 再 +0.3°） |
| **J1 + 每人 100 張 RT-GENE + 25% replay（3 組平均）** | 5.50（留出 5 人） | 12.04 | **12.5** | 0.85 | ✅（int8 未測） |

詳細表格、信賴區間、逐人結果和統計檢定都在 `results/final/RESULTS_FINAL.md`。

---

## 4. 最新實驗：給模型看一點點「新環境」的資料，夠不夠？

### 為什麼做這個
- 第 9 項證明「看更多角度」沒用，所以 RT-GENE 的問題出在**影像長得不一樣**：遠距離、解析度低、眼鏡被修圖移除。
- 真正上車時也一樣：車內是近紅外線攝影機，臉的樣子跟所有公開資料集都不同。
- 所以要回答一個很實際的問題：**如果去車上收一小批資料，要收多少才夠？會不會把原本學好的東西忘掉？**
- 我們用 RT-GENE 模擬「新環境」：一部分人當作「收來的資料」，另一部分人當作「之後遇到的新駕駛」。

### 怎麼做（訓練前就固定下來，不看結果再改）
- **起點模型**：J1（MobileNetV3，已能處理大角度與正面）。
- **切人**：RT-GENE 17 人依編號排序，第 i 人分到第 i % 3 組。
  - 每次拿 1 組當測試（從沒看過的新人），其他 2 組當可以拿來微調的資料。
  - 篩選階段只跑第 0 組：測試 6 人（s000 s003 s006 s009 s012 s015，34,739 張），微調可用 11 人。
- **資料量**：每人 10 張（共 110 張）、每人 100 張（1,100 張）、全部（79,819 張）。
- **防遺忘**：多一個設定，每個 batch 一半放原本的訓練資料（Gaze360 + MPII），叫做 replay。
- **訓練**：每個設定都是 1,500 個 iteration；學習率 3e-5，cosine 降到 1/100；損失函數和資料增強與原本訓練相同。
- **每 500 個 iteration 測一次三樣東西**：
  1. RT-GENE 新的人（主要目標）。
  2. MPII 留出的 5 人（有沒有忘記正面情境）。
  3. Gaze360 test（有沒有忘記大角度）。

### 成功標準（事先定好）
1. **用每人 100 張以內，RT-GENE 新的人降到 12° 以下**（目前約 29°；只用 MPII 訓練的模型約 15°）。
2. **不遺忘**：MPII 留出 5 人與 Gaze360 各自不比微調前差超過 0.5°（不加 replay 時預期會忘，所以這條主要看 replay 版）。
3. 第 1 條達成後，才做全部 3 組，並補做 int8 QAT，確認量化後一樣好。

參考點：RT-GENE 原論文在同資料集內訓練與測試，大約 8° 上下（用了眼睛影像與多模型集成，條件不同，不能直接比）。

### 成本
- 篩選：4 個設定 × 約 10–15 分鐘，約 **1 GPU-h**（`/work/<user>/runs/rtgene_ft/screen.sh`）。
- 全部 3 組加 int8：再約 2–3 GPU-h。只有篩選結果好才做。

### 篩選結果（2026-11-11，job 751859，0.6 GPU-h）

測試對象是 6 個從沒看過的 RT-GENE 受試者。所有數字都是第 1500 個 iteration（訓練結束）的值，單位是度。

| 設定 | 用了幾張 RT-GENE | RT-GENE 新的人（全部影像平均／逐人平均） | MPII 留出 5 人 | Gaze360 |
|---|---|---|---|---|
| 微調前的 J1 | 0 | 29.71／26.10 | 5.27 | 11.57 |
| 對照：現在的車用模型（只用 MPII 訓練） | 0 | 15.71／14.47 | — | 約 31 |
| 每人 10 張 | 110 | 13.78／10.83 | 5.12 | **21.70** |
| 每人 100 張 | 1,100 | 13.67／10.31 | 6.39 | **22.81** |
| 全部 | 79,819 | **13.39／10.20** | 6.22 | **24.03** |
| 全部 + 一半舊資料（replay） | 79,819 | 14.55／11.58 | 5.41 | 11.92 |

**白話結論：**
1. **一點點資料就有大效果。** 只看 110 張照片，誤差就從 29.7° 降到 13.8°。用 8 萬張也只多進步 0.4°，所以「收多少」不是重點，「有沒有收」才是。
2. **微調後比現在的車用模型好。** 在同一批新的人上，13.4–13.8° 對 15.7°（逐人平均 10.2–10.8° 對 14.5°）。
3. **但會忘記大角度。** 不混舊資料的話，Gaze360 從 11.6° 退到 22–24°。混一半舊資料後幾乎不忘（Gaze360 +0.3°、MPII +0.1°），代價是 RT-GENE 多 1.2°。
4. **事先定的標準：**
   - 「RT-GENE 降到 12° 以下」：用之前一直使用的全部影像平均是 13.7°，**沒有達成**；改用逐人平均是 10.3°，有達成。我們以事先用的全部影像平均為準。
   - 「不遺忘」：有 replay 時達成，沒有 replay 時嚴重遺忘。
   - 因為第 1 條沒有達成，依規定不自動進入「3 組 + int8」。

**建議的下一步（已執行，見下方追加實驗）：**
- 「每人 10 張 + replay」：最接近實際上車的做法，收很少的資料，又不會忘記。
- 換第 1、2 組受試者再跑一次，確認結果不是剛好挑到容易的人。

### 追加實驗：換受試者、加 replay（2026-11-11，job 752448，0.8 GPU-h）

每一組各有 5–6 個從沒看過的受試者。RT-GENE 是全部影像平均，單位是度。

| 設定（每人 10 張 = 110–121 張） | 第 0 組 | 第 1 組 | 第 2 組 | **3 組平均** |
|---|---|---|---|---|
| RT-GENE：微調前的 J1 | 29.71 | 29.86 | 26.21 | 28.59 |
| RT-GENE：現在的車用模型（只用 MPII） | 15.71 | 15.51 | 13.06 | 14.76 |
| RT-GENE：每人 10 張 | 13.78 | 12.84 | 11.29 | **12.64** |
| RT-GENE：每人 10 張 + replay | 15.61 | 14.72 | 13.44 | 14.59 |
| Gaze360：每人 10 張（微調前 11.57） | 21.70 | 23.09 | 22.75 | 22.51 |
| Gaze360：每人 10 張 + replay | 11.79 | 11.70 | 11.72 | **11.74** |
| MPII 留出 5 人：每人 10 張（微調前 5.27） | 5.12 | 5.82 | 6.14 | 5.69 |
| MPII 留出 5 人：每人 10 張 + replay | 5.37 | 5.23 | 5.68 | **5.43** |

**白話結論：**
1. **結果在 3 組受試者上都成立。** 每人 10 張平均讓 RT-GENE 從 28.6° 降到 12.6°，比現在的車用模型（14.8°）好 2°，其中第 2 組已低於 12°。
2. **但只用 10 張微調一定會忘記大角度**（Gaze360 退到 22.5°）。
3. **10 張加 replay 完全不忘**（Gaze360 +0.2°、MPII +0.2°），但 RT-GENE 只到 14.6°，和現在的車用模型差不多。照片太少時，一半舊資料把新資料的效果稀釋掉了。
4. **目前最平衡的是「10 張 + replay」**：一個 0.85 GMACs 的小模型，Gaze360 11.7°、MPII 5.4°、RT-GENE 14.6°，三種情境都不會壞掉。現在的車用模型在 Gaze360 是 31°。
5. 這兩個方向已在下一節的 R2 測試。

### R2：每人收多少張 × 混多少舊資料（2026-11-11，job 753691／753961／753693，2.6 GPU-h，含慢節點浪費的 1.3）

標準沿用上面：RT-GENE 3 組平均 ≤ 13°，Gaze360 與 MPII 都不比微調前差超過 0.5°。數字是 3 組平均，單位為度。

| 設定 | RT-GENE | Gaze360（前 11.57） | MPII 留出 5 人（前 5.27） | 達標？ |
|---|---|---|---|---|
| 每人 10 張，不混 | 12.64 | 22.52 | 5.69 | ✗ 會忘 |
| 每人 10 張 + 50% | 14.59 | 11.74 | 5.43 | ✗ |
| 每人 10 張 + 25% | 13.80 | 12.04 | 5.75 | ✗ |
| 每人 100 張 + 50% | 13.16 | 11.74 | 5.36 | ✗（差 0.16°） |
| **每人 100 張 + 25%** | **12.54** | 12.04 | 5.50 | **✓** |

**白話結論：**
1. **每人 100 張、混 25% 舊資料**：RT-GENE 和「不混」一樣好（12.5° vs 12.6°），而且不會忘記大角度。
2. 多收照片和少混舊資料都讓 RT-GENE 變好；少混舊資料的代價是 Gaze360 多 0.3°。
3. **餘裕很小**：Gaze360 只比上限低 0.03°，而且只有一個 seed。第 0 組單獨看是 14.5°，但三組都比現在的車用模型好。
4. 下一步（需要同意才會跑）：對這個設定做 int8 QAT；有車內資料後，用相同流程驗證。

### 相關論文（解釋與延伸用）
- Park et al., *Few-Shot Adaptive Gaze Estimation*（FAZE）, ICCV 2019：每人少量樣本做個人化校正，與「每人 10 張」的設定直接相關。
- Liu et al., *Generalizing Gaze Estimation with Outlier-guided Collaborative Adaptation*（PnP-GA）, ICCV 2021：不用標註的領域適應。若車內資料沒有標註可以考慮。
- Cheng, Bao & Lu, *PureGaze: Purifying Gaze Feature for Generalizable Gaze Estimation*, AAAI 2022：訓練時就讓特徵不受外觀影響（領域泛化）。
- Zhang et al., *ETH-XGaze: A Large Scale Dataset for Gaze Estimation under Extreme Head Pose and Gaze Variation*, ECCV 2020：另一個大角度資料集（需要申請）。
- Ortega et al., *DMD: A Large-Scale Multi-Modal Driver Monitoring Dataset for Attention and Alertness Analysis*, ECCV Workshops 2020：真實車內資料集。有注視區域標籤，但沒有連續的視線角度，適合最後驗證「注視區域」判斷。

### 預想的結果與之後怎麼走

| 結果 | 代表什麼 | 下一步 |
|---|---|---|
| 每人 10–100 張就降到 12° 以下 | 上車時只要收一小批車內資料就行 | 規劃車內資料收集（每位駕駛幾十張），做 3 組 + int8 |
| 要用全部資料才有效 | 需要大量目標資料 | 改試不需標註的領域適應（PnP-GA 類） |
| 加 replay 才不會忘 | 微調時要混入舊資料 | 部署流程固定使用 replay |
| 怎樣都降不下來 | 可能是 RT-GENE 修圖後的影像本身就難 | 改用 DMD 等真實車內資料驗證 |

---

## 5. 檔案在哪裡

| 內容 | 位置 |
|---|---|
| 完整報告（論文格式、所有表格） | `results/final/RESULTS_FINAL.md` |
| 15 fold 主實驗紀錄 | `results/remote/`（`NOTES.md`） |
| 輕量化、蒸餾、量化紀錄 | `results/lightweight/`（`NOTES.md`） |
| 跨資料集、Gaze360、G2／J1 | `results/crossdata/`（`NOTES.md`、各 JSON） |
| 訓練與評估程式 | `scripts/`（`train_mpii.py`、`train_gaze360.py`、`qat_finetune.py`、`qat_gaze360.py`、`eval_crossdata.py`、`finetune_rtgene.py`…） |
| 國網執行計畫 | `REMOTE_PLAN.md` |
| 權重、資料集、預測檔 | **不在 repo 裡**，放在國網 `/work/<user>/runs`、`/work/<user>/datasets` |

### 環境
- 國網 Nano4：venv `/work/<user>/envs/l2cs`，所有訓練都用 Slurm 送出（例如 `sbatch` 對應的 `.sh`）。
- 本機：conda env `l2cs`，安裝方式見 `docs/LOCAL_LOG.md`。
- `L2CS-Net/` 是官方 repo 的 git submodule，clone 時要加 `--recursive`。
