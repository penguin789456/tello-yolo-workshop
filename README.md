# Tello + YOLO 影像辨識工作坊

這是一份給大學生實作的兩節課教材。你會先把 **Tello EDU** 的影像接到電腦、拍照存檔，
再用 **YOLO** 做物件辨識，最後把「拍照」和「辨識」整合成一個可以自己跑的工具。

整套教材刻意保持簡單：用命令列操作、用 OpenCV 開預覽視窗、不做 GUI、不寫多執行緒。
重點不是把程式寫得很花，而是看懂一條影像處理流程怎麼從「能跑」變成「可用」。

| 項目 | 本教材使用 |
|---|---|
| 無人機 | Tello EDU（`djitellopy`） |
| 影像辨識 | YOLO（`ultralytics`，只做推論，不訓練） |
| 顯示畫面 | `cv2.imshow` |
| 操作方式 | CLI 指令 + 鍵盤 |
| 輸出 | `runs\<時間戳>\` 裡的圖片與 `index.jsonl` |

---

## 兩節課會做什麼

| 節次 | 主題 | 你會完成什麼 | 主要檔案 |
|---|---|---|---|
| 第一節 | 操控與影像，分開練 | 起飛前進 100 cm 再降落；看到即時畫面並按 `P` 拍照 | `lab1_flight.py`、`lab2_capture.py` |
| 第二節 | 整合 + 辨識 | 邊操控邊看畫面，按 `P` 拍照並對那一張做 YOLO 辨識 | `lab3_detect.py` |

建議閱讀順序：

1. [docs/00-課程地圖.md](docs/00-課程地圖.md)
2. [docs/01-lab1.md](docs/01-lab1.md)：Tello 操控與路徑控制
3. [docs/02-lab2.md](docs/02-lab2.md)：Tello 影像與拍照
4. [docs/03-lab3.md](docs/03-lab3.md)：整合 + 拍照後辨識

---

## 課前準備

### 0. 需求

| 需要 | 說明 |
|---|---|
| Windows 10 / 11 | 指令以 Windows 寫法為主 |
| **Python 3.12**（建議） | 可用範圍 **3.10 ~ 3.13**。安裝時勾選 **Add Python to PATH** |
| Tello EDU | 第一節會用來連線與拍照 |
| 網路 | 安裝套件與下載權重時需要 |

先確認 Python：

```bat
python --version
```

要看到 `Python 3.12.x`（或 3.10 ~ 3.13 之間的版本）。

**為什麼有上下限？** 不是我們挑的，是套件決定的：

| 界線 | 誰決定的 |
|---|---|
| 最低 3.10 | `pillow`（ultralytics 的相依套件）要求 `>=3.10` |
| 最高 3.13 | **`torch==2.7.0` 只發布到 Python 3.13 的 wheel** |
| 實測版本 | 這份教材在 **3.12.4** 上跑完全部驗收 |

**Python 3.14 或更新的版本會裝不起來**，`pip` 會說找不到符合的 torch：

```
ERROR: Could not find a version that satisfies the requirement torch==2.7.0
```

那不是你打錯字，是真的沒有那個 wheel。請改用 3.12。

> 已經裝了別的版本也不用重裝整個 Python —— 從
> [python.org](https://www.python.org/downloads/) 另外裝一個 3.12，
> 建立虛擬環境時指定它就好：
>
> ```bat
> py -3.12 -m venv .venv
> ```
>
> （`py` 是 Windows 的 Python 啟動器，`py -0` 可以列出你電腦上所有版本。）

### 1. 建立環境

安裝所有課程會用到的套件：

```bat
git clone https://github.com/penguin789456/tello-yolo-workshop.git
cd tello-yolo-workshop
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

套件下載比較久，請不要留到上課才裝。Tello 連線後電腦會切到 `TELLO-XXXXXX` 熱點，
那時候沒有網際網路，不能一邊連無人機一邊下載套件。

### 2. 下載 YOLO 權重

```bat
python -c "from ultralytics import YOLO; YOLO('yolo26n.pt')"
move yolo26n.pt models\
```

`yolo26n.pt` 是 COCO 預訓練權重，只認得 80 類，例如 `person`、`bottle`、`cup`、
`book`、`cell phone`、`laptop`。它不認得 `pencil` 或 `pen`。

---

## 第一節：操控與影像，分開練

### A. 先確認環境裝好了（不需要無人機）

```bat
.venv\Scriptsctivate
python labs\lab3_detect.py --source video --path data\samples	est.mp4 --headless --max-frames 5 --no-model
```

看到「讀 5 幀」就代表環境沒問題。`--no-model` 讓它不載 YOLO，所以這一步很快。

### B. 操控：起飛、前進 100 公分、降落

1. 打開 Tello 電源，等機頭燈閃黃燈
2. 電腦 Wi-Fi 連到 `TELLO-XXXXXX`（沒有密碼）
3. 先確認連得上（**這一步不會飛**）：

```bat
python labs\lab1_flight.py
```

會印出電量、高度、溫度，然後告訴你「沒有加 --path 或 --manual，所以不起飛」。

確認場地空曠、前方至少 2 公尺，再跑路徑控制：

```bat
python labs\lab1_flight.py --path
```

起飛 → 前進 100 公分 → 降落。想改距離用 `--distance 150`（範圍 20~500）。

想用鍵盤即時操控：

```bat
python labs\lab1_flight.py --manual --speed 20
```

> ⚠️ **方向鍵是「按一下持續移動」，放開不會停。要停請按空白鍵。**
> 第一次實飛請用 `--speed 20`。

詳細說明與首飛檢查表：[docs/01-lab1.md](docs/01-lab1.md)

### C. 影像：看畫面、按 P 拍照

這一支**不會飛**，無人機放桌上開機就好：

```bat
python labs\lab2_capture.py
```

看到畫面後：

- 按 `P`：拍一張，存到 `runs\<時間戳>\`
- 按 `Q`：離開

接著檢查輸出：

```bat
dir runs
type runs\<剛剛產生的時間戳>\index.jsonl
```

沒有無人機也能練：

```bat
python labs\lab2_capture.py --source video --path data\samples	est.mp4
```

詳細說明：[docs/02-lab2.md](docs/02-lab2.md)

---

## 第二節：整合 + 辨識

把第一節的操控和影像接起來，再加上 YOLO：

```bat
python labs\lab3_detect.py
```

- 畫面以原本的速度跑，**模型完全不介入**
- 按 `P`：拍照 → 對那一張做 YOLO 辨識 → 存標註圖與結果
- `T` 起飛、`L` 降落、空白鍵懸停、方向鍵移動、`Q` 離開

按一次 `P` 會印一行：

```
拍一張 -> 20261002_203022_211540_snap000042.jpg　1 處　最高 0.44　52 ms　bottle
```

**辨識只在按 `P` 的時候做，不做連續辨識。** 原因是實測的數字：CPU 上一張要
50~60 ms，連續跑會把畫面壓到個位數 fps，那個更新率沒辦法操控無人機。

沒有無人機也能練：

```bat
python labs\lab3_detect.py --source video --path data\samples	est.mp4
```

詳細說明：[docs/03-lab3.md](docs/03-lab3.md)

---

## 常用按鍵

| 鍵 | 動作 | 哪幾支 |
|---|---|---|
| `Q` / `Esc` | 離開（飛行中會先降落） | 三支都有 |
| `P` | 拍一張並存檔 | lab2（只拍照）、lab3（拍照＋辨識） |
| `T` / `L` | 起飛 / 降落 | lab1 `--manual`、lab3 |
| 空白鍵 | 四軸速度歸零，讓無人機懸停 | lab1 `--manual`、lab3 |
| 方向鍵 / `8 2 4 6` | 前進 / 後退 / 左移 / 右移 | lab1 `--manual`、lab3 |
| `W` / `S` | 上升 / 下降 | lab1 `--manual`、lab3 |
| `A` / `D` | 左轉 / 右轉 | lab1 `--manual`、lab3 |
| `X` | 切斷馬達（需 `--allow-emergency`） | lab1 `--manual`、lab3 |

**lab2 沒有任何飛行鍵** —— 按 `T` 會印一行提示叫你去用 lab1。

lab2 / lab3 的按鍵要先用滑鼠點一下影像視窗取得焦點才有效（它們走 `cv2.waitKey`）。
lab1 沒有視窗，用 `msvcrt` 直接讀主控台 —— 那是 Windows 專用的。

飛行前一定要記住：方向鍵是「按一下後持續移動」，放開不會自動停。要停請按空白鍵。

---

## 輸出格式

每次執行會建立一個資料夾：

```text
runs\20261002_160643\
  20261002_160644_123456_sample_01.jpg
  index.jsonl
```

`index.jsonl` 一行代表一張圖：

```json
{"file": "20261002_195929_123456_sample_01.jpg", "label": "sample_01",
 "time": "2026-10-02T19:59:29", "count": 1, "best_score": 0.1721,
 "classes": ["orange"], "latency_ms": 52.4}
```

你可以用它回查每張圖的最高分、辨識到的類別與推論耗時。

---

## 排錯速查

| 現象 | 做法 |
|---|---|
| `ModuleNotFoundError: No module named 'ultralytics'` | 先 `.venv\Scripts\activate`，再確認已安裝 `requirements.txt` |
| 一直下載 `yolo26n.pt` | 權重沒有放到 `models\` |
| 連不上 Tello | Wi-Fi 要連到 `TELLO-XXXXXX`，而且上一個 Python 程式不能還開著 |
| 10 秒內沒有畫面 | 關掉所有 Python，等五秒再重跑 |
| 按鍵沒反應 | 用滑鼠點一下影像視窗，讓視窗取得焦點 |
| 飛機停不下來 | 按空白鍵 |
| 中文亂碼 | 先跑 `chcp 65001` |

更多說明見 [docs/91-排錯手冊.md](docs/91-排錯手冊.md)。

---

## 文件

- [docs/00-課程地圖.md](docs/00-課程地圖.md)：兩節課怎麼走
- [docs/01-lab1.md](docs/01-lab1.md)：Tello 操控與路徑控制（不碰影像）
- [docs/02-lab2.md](docs/02-lab2.md)：Tello 影像與拍照（不碰飛行）
- [docs/03-lab3.md](docs/03-lab3.md)：整合 + 按 P 辨識
- [docs/89-逐章差異.md](docs/89-逐章差異.md)：三支程式差在哪
- [docs/90-設計決策.md](docs/90-設計決策.md)：為什麼這樣設計
- [docs/91-排錯手冊.md](docs/91-排錯手冊.md)：常見錯誤
- [docs/92-課後練習.md](docs/92-課後練習.md)：練習任務
