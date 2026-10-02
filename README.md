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
| 第一節 | 測試 Tello 連線、拍照、YOLO 辨識 | 連上無人機、按 `P` 拍照、用內附圖片測試 YOLO | `lab1_tello.py`、`lab2_detect.py` |
| 第二節 | 整合 Tello 拍照與辨識 | 把即時畫面、拍照、辨識、存檔整理成同一個流程 | `lab3_patrol.py` |

建議閱讀順序：

1. [docs/00-課程地圖.md](docs/00-課程地圖.md)
2. [docs/01-lab1.md](docs/01-lab1.md)：Tello 連線範例
3. [docs/02-lab2.md](docs/02-lab2.md)：YOLO 辨識範例
4. [docs/03-lab3.md](docs/03-lab3.md)：整合測試

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

## 第一節：Tello 連線、拍照、YOLO 測試

### A. 先確認基本流程能跑

```bat
.venv\Scripts\activate
python labs\lab3_patrol.py --source image --path data\samples --dry-run --headless
```

看到 `--dry-run：沒有寫入任何檔案` 就代表程式流程可以啟動。

### B. 連上 Tello 並拍照

1. 打開 Tello 電源，等機頭燈閃黃燈
2. 電腦 Wi-Fi 連到 `TELLO-XXXXXX`
3. 執行：

```bat
python labs\lab1_tello.py --preview-only
```

看到畫面後：

- 按 `P`：拍一張，存到 `runs\<時間戳>\`
- 按 `Q`：離開

接著檢查輸出：

```bat
dir runs
type runs\<剛剛產生的時間戳>\index.jsonl
```

### C. 測試 YOLO 辨識

先用內附圖片測試，不需要無人機：

```bat
python labs\lab2_detect.py --source data\samples --headless
```

如果你有自己拍的照片，也可以放進 `data\mine\`：

```bat
python labs\lab2_detect.py --source data\mine
```

建議拍一張同時有瓶子和鉛筆的照片。YOLO 通常會認出瓶子，但認不出鉛筆，這可以幫你理解
「模型只會認得它訓練過的類別」。

---

## 第二節：整合拍照與辨識

第二節會把第一節的兩件事接起來：

- Tello 或攝影機提供即時影像
- 每隔幾幀做一次 YOLO 辨識
- 按 `P` 可以手動拍照
- 超過門檻的結果會自動存檔
- 每次執行都會產生圖片與 `index.jsonl`

整合版可以吃三種來源，先用測試影片練習：

```bat
python labs\lab3_patrol.py --source image --path data\samples --headless
python labs\lab3_patrol.py --source video --path data\samples\test.mp4 --loop --headless --max-frames 50
python labs\lab3_patrol.py --source tello --preview-only
```

要讓 Tello 畫面也做辨識，確認 `models\yolo26n.pt` 已經準備好，再執行：

```bat
python labs\lab1_tello.py --every 10
```

---

## 常用按鍵

| 鍵 | 動作 |
|---|---|
| `Q` / `Esc` | 離開 |
| `P` | 拍一張並存檔 |
| `T` | 起飛 |
| `L` | 降落 |
| 空白鍵 | 停止水平與旋轉速度，讓無人機懸停 |
| 方向鍵 / `8 2 4 6` | 前後左右 |
| `W` / `S` | 上升 / 下降 |
| `A` / `D` | 左轉 / 右轉 |

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
- [docs/01-lab1.md](docs/01-lab1.md)：Tello 連線、預覽與拍照
- [docs/02-lab2.md](docs/02-lab2.md)：YOLO 靜態圖片辨識
- [docs/03-lab3.md](docs/03-lab3.md)：整合測試（三種來源）
- [docs/89-逐章差異.md](docs/89-逐章差異.md)：三支程式差在哪
- [docs/90-設計決策.md](docs/90-設計決策.md)：為什麼這樣設計
- [docs/91-排錯手冊.md](docs/91-排錯手冊.md)：常見錯誤
- [docs/92-課後練習.md](docs/92-課後練習.md)：練習任務
