# Tello + YOLO 影像辨識工作坊

把 **Tello EDU** 的影像接到桌電，用 **YOLO** 做物件辨識，結果存成本機檔案與可查的
索引。CLI 操作、單執行緒、無 GUI 框架。

| | 本教材用什麼 |
|---|---|
| 無人機 | Tello EDU（`djitellopy`） |
| 模型 | YOLO（`ultralytics`，只做推論，不訓練） |
| 介面 | CLI + `cv2.imshow` 預覽，**沒有 GUI 框架** |
| 併發 | **一條主迴圈，零執行緒** |
| 儲存 | **本機檔案 + `index.jsonl`**，零金鑰、零網路呼叫 |

**兩節實作課，共約 100 分鐘**（每節 50 分鐘）。
環境必須課前裝好 —— 實測完整環境即使不用下載、純解壓也要 2 分 27 秒。
分鐘級時間表見 [docs/00-課程地圖.md](docs/00-課程地圖.md)。

---

## 這份教材的三個規矩

**1. 每支 lab 一檔到底。** 沒有共用套件、沒有 `utils.py`。四支程式各自完整，你從
第一行讀到最後一行就讀完了，不用跟著 `import` 跳檔。重複的程式碼（載模型、畫框、
存檔）在四支檔案裡各寫一份 —— 這是刻意付出的代價，換來的是可讀性。

四支檔案的段落順序完全一樣，所以要看每章加了什麼，直接 diff：

```bat
fc labs\lab2_loop.py labs\lab3_tello.py
```

段落固定是這七段：

```
1. 參數    2. 載入模型    3. 影像來源    4. 推論
5. 畫框    6. 存檔        7. 主迴圈
```

**2. 不用 thread。** 指的是**你寫的程式碼**：不 `import threading`、不做佇列、
只有一條 `while`。djitellopy 內部會自己開一條 H.264 解碼執行緒，那是它的事，
我們不寫也不管 —— 而它剛好讓我們不必自己處理「丟舊幀」（見
[docs/90-設計決策.md](docs/90-設計決策.md) 第 3 條）。

**3. 不用 GUI 框架。** 沒有 PySide6、pygame、tkinter。預覽用 `cv2.imshow` +
`cv2.waitKey`：單執行緒、OpenCV 內建、沒有事件迴圈也沒有訊號槽。每支 lab 都有
`--headless`，完全不開窗也能跑（自動測試、遠端機器用）。

> 為什麼還是要有畫面？飛行時看不到畫面等於盲飛。`cv2.imshow` 是「一個視窗」，
> 但不是「一套 GUI 框架」—— 這兩件事的差別本身就是一課。

---

## 步驟 0：需求

| 需要 | 版本 | 說明 |
|---|---|---|
| Windows | 10 / 11 | 指令都以 `.bat` 風格寫；Linux/macOS 把 `\` 換成 `/` 即可 |
| Python | 3.11 / 3.12 | 安裝時要勾 **Add Python to PATH** |
| Tello EDU | — | **第一節課必備。第二節課可以用筆電鏡頭或內附素材代替** |

```bat
python --version
```

---

## 步驟 1：安裝套件（分兩段）

完整環境是 **1.9 GB**（PyTorch 一個就占 1.2 GB），三十個人同時在教室裝會全部耗在
進度條上。所以拆成兩段：

### 1-A　第一節課用的最小環境（265 MB）

```bat
git clone https://github.com/penguin789456/tello-yolo-workshop.git
cd tello-yolo-workshop
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-step1.txt
```

只有 `opencv-python` + `numpy` + `djitellopy`。**這樣就足以把無人機影像接到桌電**
—— 因為 `lab3_tello.py --preview-only` 完全不碰模型（`from ultralytics import YOLO`
寫在函式裡，預覽模式不會呼叫到）。

### 1-B　第二節課之前再裝完整環境（1.9 GB）

```bat
.venv\Scripts\activate
pip install -r requirements.txt
```

這段會下載 PyTorch，要幾分鐘。**請在第二節課之前自己裝好**，不要留到課堂上。

---

## 步驟 2：取得 YOLO 權重

```bat
.venv\Scripts\activate
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
move yolo11n.pt models\
```

第一次建立 `YOLO('yolo11n.pt')` 會自動從網路下載（約 5 MB）到目前目錄，搬進
`models\` 就好。之後所有 lab 的 `--weights models/yolo11n.pt` 都指這個檔。

> ⚠️ **這是 COCO 預訓練權重，只認得 80 類。**
>
> **認得：** `person` `bottle` `cup` `book` `scissors` `cell phone` `laptop`
> `mouse` `keyboard` `chair`…
> **不認得：** `pencil`、`pen`（COCO 裡沒有鉛筆這個類別）。
>
> 而且它不是安靜地什麼都不找 —— 內附的 `data\samples\` 是合成的幾何圖形，跑下去
> 它會把圓形標成 `sports ball`、三角形標成 `kite`、方塊標成 `tv`，最高分 0.32。
> **分類模型永遠會給你一個答案，還附一個看起來有把握的分數。**
> 教材刻意留著這個現象當教材（見
> [docs/90-設計決策.md](docs/90-設計決策.md) 的「誠實的限制」）。
>
> 要辨識你真正要的東西，換一顆權重就好，**四支程式一行都不用改**：
>
> ```bat
> python labs\lab4_patrol.py --source image --path data\samples --weights models\best.pt
> ```
>
> 訓練不在這個工作坊的範圍。

---

## 步驟 3：驗證安裝

**只裝了最小環境（1-A）時**，用 `--dry-run` 確認流程走得通 —— 它不載模型、不寫檔：

```bat
python labs\lab4_patrol.py --source image --path data\samples --dry-run --headless
```

最後一行印出「`--dry-run：沒有寫入任何檔案`」就對了。

**裝完完整環境（1-B）之後**，跑真的推論。不需要無人機，`data\samples\` 已經放好
8 張合成測試圖與一支 40 幀的測試影片：

```bat
python labs\lab1_detect.py --source data\samples --headless
type runs\<剛剛產生的時間戳>\index.jsonl
```

看到 8 行 JSON、`runs\` 底下有 8 張 jpg，就成功了。

---

## 三節課 ↔ 四支程式

| 節 | 主題 | 用哪支 | 講義 | 需要無人機 |
|---|---|---|---|---|
| **1** | 無人機影像接到桌電、下載權重 | `lab3_tello.py --preview-only` | [03-lab3.md](docs/03-lab3.md) | ✓ |
| **2** | 影像辨識：瓶子、杯子、書… | `lab1_detect.py` → `lab2_loop.py` | [01](docs/01-lab1.md)、[02](docs/02-lab2.md) | 選用 |
| **3** | 競賽作品介紹（老師簡報） | — | — | — |
| 選讀 | 飛行操控、三來源整合 | `lab3` 完整、`lab4_patrol.py` | [03](docs/03-lab3.md)、[04](docs/04-lab4.md) | ✓ / 選用 |

> **檔名編號 ≠ 上課順序。** 編號是**程式碼的累積順序**：`lab2` = `lab1` + 連續迴圈、
> `lab3` = `lab2` + 真無人機、`lab4` = `lab3` + 三來源整合。這個累積關係讓後面的檔案
> 可以直接 diff 出「多了什麼」，所以檔名不跟著上課順序改。
>
> 第一節課雖然用 `lab3`（422 行），但 `--preview-only` 只會走到**段落 1、3、7** ——
> 段落 2（載入模型）、4（推論）、5（畫框）在預覽模式下完全不會被呼叫，可以整段跳過，
> 第二節課再回頭補。每份講義開頭都標了「這節要讀哪幾段」。

完整安排見 [docs/00-課程地圖.md](docs/00-課程地圖.md)。

### 最短的四道指令

```bat
python labs\lab1_detect.py --source data\samples --headless
python labs\lab2_loop.py   --source data\samples\test.mp4 --every 5 --loop --headless --max-frames 60
python labs\lab3_tello.py  --preview-only
python labs\lab4_patrol.py --source video --path data\samples\test.mp4 --loop --headless --max-frames 50
```

---

## 操作

### 看無人機畫面：`lab3_tello.py`

1. 裝好電池、開 Tello 電源，等機頭燈開始閃黃燈
2. 筆電 Wi-Fi 連到 `TELLO-XXXXXX`（**無密碼**；Windows 顯示「無網際網路」是正常的）
3. 跑：

```bat
python labs\lab3_tello.py --preview-only
```

會跳出視窗顯示即時畫面，主控台印出：

```
已連線　電量 87%　溫度 52C
#1  待命  電量 87%　高度 0 cm  僅顯示 0 處 最高 0.00  命中 0
```

按 `P` 拍一張、`Q` 離開。要邊看邊辨識（需要 1-B 環境）：`--every 10`。

| 參數 | 預設 | 說明 |
|---|---|---|
| `--preview-only` | 關 | **完全不載模型**，只看畫面。飛行時用這個 |
| `--every N` | 10 | 每 N 幀推論一次 |
| `--conf` | 0.25 | 模型回報的信心度門檻 |
| `--threshold` | 0.8 | 自動存檔門檻（比 `--conf` 嚴） |
| `--cooldown` | 3.0 | 兩次自動存檔至少間隔幾秒 |
| `--speed` | 40 | rc 速度 0~100。第一次實飛用 20 |
| `--weights` | `models/yolo11n.pt` | 權重路徑 |
| `--out` | `runs` | 輸出根目錄 |
| `--no-land-on-exit` | 關 | 離開時不自動降落（你要自己接手） |
| `--allow-emergency` | 關 | 啟用 `X` 鍵＝切斷馬達（機體會墜落） |

**⚠️ 一台 Tello 同時只能被一台筆電連線。**

### 辨識照片：`lab1_detect.py`

```bat
python labs\lab1_detect.py --source data\samples
python labs\lab1_detect.py --source data\samples\sample_01.jpg --conf 0.3
python labs\lab1_detect.py --source data\samples --headless
```

不加 `--headless` 會開視窗，**按任意鍵看下一張、`q` 離開**。

**語意：一張不漏。** 每張都推論、每張都存，不看門檻、沒有冷卻。讀不進來的檔案印一行
警告跳過，不中斷整批。

| 參數 | 預設 | 說明 |
|---|---|---|
| `--source` | （必填） | 圖片檔或資料夾 |
| `--conf` | 0.25 | 信心度門檻 |
| `--headless` | 關 | 不開視窗 |
| `--dry-run` | 關 | 不載模型、不寫檔 |
| `--reload-each` | 關 | 每張都重載模型，用來量載入成本 |

### 連續影像：`lab2_loop.py`

```bat
python labs\lab2_loop.py --source 0 --every 3
python labs\lab2_loop.py --source data\samples\test.mp4 --every 5 --loop
python labs\lab2_loop.py --source data\samples\test.mp4 --loop --headless --max-frames 60
```

**鍵盤：`q` 離開、`p` 拍一張**（不看門檻、不受冷卻限制；那一幀若還沒推論過，會單獨
跑一次再存）。

**語意：丟得起就丟。** 處理不完的舊畫面直接丟掉，永遠分析最新的一格。摘要裡
「跳過 48 幀」不是缺陷，那是設計。

| 參數 | 預設 | 說明 |
|---|---|---|
| `--source` | （必填） | 影片檔，或鏡頭編號（`0`、`1`…） |
| `--every N` | 5 | 每 N 幀推論一次；其餘幀只顯示 |
| `--threshold` | 0.8 | 自動存檔門檻 |
| `--cooldown` | 3.0 | 自動存檔冷卻秒數 |
| `--loop` | 關 | 影片播完從頭再來 |
| `--max-frames` | 0（不限） | 讀滿幾幀就停 |

> `data\samples\test.mp4` **只有 40 幀**（15 fps、約 2.7 秒）。要跑超過 40 幀請加
> `--loop`。每一幀右下角都印了幀號，所以開窗跑的時候，`--every N` 跳過哪幾幀看得見。

### 整合版：`lab4_patrol.py`

一支工具吃三種來源，**兩種相反的語意跟著 `--source` 自動切換**：

```bat
python labs\lab4_patrol.py --source image --path data\samples --headless
python labs\lab4_patrol.py --source video --path data\samples\test.mp4 --loop --max-frames 50
python labs\lab4_patrol.py --source tello --preview-only
```

啟動時會印出生效的語意，不用猜：

```
有限來源（image）：每一張都推論、每一張都存，一張不漏
即時來源（video）：每 5 幀推論一次，門檻 0.8，冷卻 3.0 秒，來不及的幀會丟掉
```

結束印一份摘要：

```
=== 摘要 ===
來源 video　讀 50 幀　推論 10 幀　跳過 40 幀
平均推論 37 ms　整體 85.2 fps　耗時 0.6 s
存檔 0 張　手動拍 0 張
輸出：runs\20261002_150210
```

`--source` 以外的參數同上，另有 `--save-all`（每張推論過的都存，不看門檻）。

---

## 鍵盤

| 鍵 | 動作 | 哪幾支 |
|---|---|---|
| `Q` / `Esc` | 離開 | 全部 |
| 任意鍵 | 下一張 | lab1（開視窗時） |
| `P` | 拍一張（不看門檻、不受冷卻限制） | lab2 / lab3 / lab4 |
| `T` / `L` | 起飛 / 降落 | lab3 / lab4（Tello） |
| 空白鍵 | 四軸歸零（懸停） | lab3 / lab4（Tello） |
| ↑ ↓ ← → 或 `8 2 4 6` | 前進 / 後退 / 左移 / 右移 | lab3 / lab4（Tello） |
| `W` / `S` | 上升 / 下降 | lab3 / lab4（Tello） |
| `A` / `D` | 左轉 / 右轉 | lab3 / lab4（Tello） |
| `X` | 切斷馬達（需 `--allow-emergency`） | lab3 / lab4（Tello） |

**按鍵要先用滑鼠點一下影像視窗取得焦點才有效。**

> ⚠️ **方向鍵是「按一下持續移動」，放開不會停。** `cv2.waitKey` 沒有「放開」事件，
> 速度會 latch 住 —— **要停請按空白鍵或反向鍵**。方向鍵回傳值不跨平台，
> 非 Windows 請用 `8 2 4 6`。

---

## 輸出格式

`index.jsonl` 每一行：

```json
{"file": "20261002_160644_123456_sample_01.jpg", "label": "sample_01",
 "time": "2026-10-02T16:06:44", "count": 1, "best_score": 0.3219,
 "classes": ["sports ball"], "latency_ms": 36.2}
```

| 欄位 | 意思 |
|---|---|
| `file` | 同資料夾下的影像檔名 |
| `label` | 來源標籤（照片檔名、`frame000123`、`snap000456`） |
| `count` | 這一幀偵測到幾個物件 |
| `best_score` | 最高信心度（沒偵測到是 `0.0`） |
| `classes` | 偵測到的類別名（去重排序） |
| `latency_ms` | 這一幀的推論耗時 |

用 jsonl（一行一筆、append 寫入）而不是整包 json：程式被 `Ctrl-C` 或斷電，已經寫進去
的行仍然是合法內容。

---

## 情境速查

| 我想… | 怎麼做 |
|---|---|
| 沒有無人機也要練 | `--source data\samples`（照片）或 `--source data\samples\test.mp4 --loop` |
| 只看無人機畫面，不跑模型 | `lab3_tello.py --preview-only`（只需 1-A 環境） |
| 確認環境裝好了但還沒權重 | `lab4_patrol.py --source image --path data\samples --dry-run --headless` |
| 無人值守跑完、自動結束 | 加 `--headless --max-frames 60` |
| 畫面卡卡、CPU 燒滿 | 調大 `--every`（例如 `--every 20`） |
| 一張都沒存 | 調低 `--threshold`，或加 `--save-all` |
| 存太多幾乎一樣的圖 | 調高 `--cooldown` |
| 換成自己訓練的權重 | `--weights models\best.pt`（程式一行都不用改） |

---

## 排錯速查

| 現象 | 原因 |
|---|---|
| `ModuleNotFoundError: No module named 'ultralytics'` | 忘了 `.venv\Scripts\activate`，或只裝了 1-A |
| `UnicodeDecodeError: 'cp950' codec...`（裝套件時） | 舊版 pip 用系統編碼讀 requirements。先 `python -m pip install --upgrade pip` |
| `OSError: [WinError 1114] ... c10.dll` | PyTorch 版本不合。`pip install torch==2.7.0 torchvision==0.22.0` |
| 一直卡在 `Downloading yolo11n.pt` | 權重沒放進 `models\`，見步驟 2 |
| `連不上 Tello`（等約 28 秒才報錯） | Wi-Fi 沒連到熱點，或上一個 python 還開著占住 UDP 8889。**那 28 秒是重試，不是當掉** |
| 10 秒內沒有畫面 | UDP 11111 還被占。關掉所有 python，等五秒再試 |
| 畫面顏色很怪、天空是橘的 | RGB/BGR 沒轉（程式已處理；你改過才會這樣） |
| 按 `T` 印「取消起飛」 | 設計如此：還沒看到畫面不准起飛 |
| 飛機停不下來 | **按空白鍵。** 方向鍵是 latch 式的 |
| 按鍵完全沒反應 | 焦點不在影像視窗上，用滑鼠點一下 |
| 中文變亂碼 | `chcp 65001` 之後再跑 |

完整版見 [docs/91-排錯手冊.md](docs/91-排錯手冊.md)。

用到的 UDP 埠：**8889** 指令、**8890** 狀態封包、**11111** 影像串流。埠被占住時：

```bat
netstat -ano -p UDP | findstr 8889
taskkill /PID <查到的 PID> /F
```

---

## 把開發流程變成交付物：每節課一個 commit

這門課教的是**流程**，不只是程式。每節課結束就 commit 一次你自己的改動，最後
`git log` 就是你的開發紀錄：

```bat
git init
git add .
git commit -m "chore: 工作坊骨架與樣本素材"

:: 第一節課後
git commit -am "feat: 接通 Tello 影像，確認 --preview-only 不需要載模型"

:: 第二節課後
git commit -am "feat: 靜態與即時辨識，門檻與冷卻調成教室環境的值"
```

commit 訊息寫「**為什麼**」而不是「改了什麼」—— 改了什麼 diff 會說，為什麼只有
你知道。

---

## 專案結構

```
README.md              你正在讀的這份
requirements-step1.txt 第一節課的最小依賴（265 MB）
requirements.txt       完整依賴（1.9 GB，含 PyTorch）
labs\                  四支各自完整的程式
docs\                  講義與設計決策
data\samples\          8 張合成測試圖 + test.mp4（40 幀 / 15 fps，不需無人機）
models\                yolo11n.pt 放這裡（不進版控）
runs\                  每次執行的輸出（不進版控）
```

輸出長這樣：

```
runs\20261002_160643\
  20261002_160644_123456_sample_01.jpg   標註後的圖
  ...
  index.jsonl                            一行一筆：檔名、時間、最高分、類別、延遲
```

---

## 已知限制

- **COCO 預訓練權重只認得 80 類**，裡面沒有鉛筆，也沒有內附素材的幾何圖形。
  而且它不會說「我不知道」，會硬挑一個最像的給你。見步驟 2。
- **lab3 沒有在真機上驗證過。** 開發環境沒有 Tello。鍵盤對應、安全規則與收尾
  流程都有寫，但真機的起飛／降落反應沒實測。第一次實飛請照
  [docs/03-lab3.md](docs/03-lab3.md) 的首飛檢查表：空曠處、`--speed 20`、
  隨時準備按 `L`。
- **鍵盤是「按一下持續移動」**，不是「按住才動」。`cv2.waitKey` 沒有「放開」
  事件。要停下來按**空白鍵**。飛之前先記住這件事。
- `ultralytics` 授權是 **AGPL-3.0**。課堂使用沒問題，作品要散布前請看授權條款。

---

## 文件

- [docs/00-課程地圖.md](docs/00-課程地圖.md) — 四節課的時間表與先修知識
- [docs/01-lab1.md](docs/01-lab1.md) ~ [docs/04-lab4.md](docs/04-lab4.md) — 各章講義
- [docs/89-逐章差異.md](docs/89-逐章差異.md) — 每章相對前章改了哪幾段
- [docs/90-設計決策.md](docs/90-設計決策.md) — **七個「為什麼」，這份最重要**
- [docs/91-排錯手冊.md](docs/91-排錯手冊.md) — 錯誤訊息對照表
- [docs/92-作業與評分.md](docs/92-作業與評分.md) — 四份作業與評分表
