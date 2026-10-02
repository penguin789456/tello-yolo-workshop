"""Lab 1：靜態推論 —— 把一批照片跑過 YOLO，畫框、存檔、寫索引。

    python labs\\lab1_detect.py --source data\\samples --headless
    python labs\\lab1_detect.py --source data\\samples\\sample_01.jpg --conf 0.3
    python labs\\lab1_detect.py --source data\\samples --reload-each

這一章的語意是「圖片模式」：選幾張就處理幾張，一張不漏。
不丟幀、沒有冷卻、也不看門檻 —— 使用者挑的每一張都要有結果。讀不進來的檔案
印一行警告跳過，不讓一張壞圖中斷整批。

四支 lab 的段落順序完全一樣，方便你用 diff 看每章加了什麼：

    1. 參數   2. 載入模型   3. 影像來源   4. 推論
    5. 畫框   6. 存檔      7. 主迴圈

沒有 import threading。整支程式只有一條迴圈。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import namedtuple
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Windows 主控台預設不是 UTF-8。統一成 UTF-8 並容錯，否則 print 中文時可能
# 直接丟 UnicodeEncodeError，或把輸出導向檔案時變成亂碼。
# stderr 也要處理：SystemExit 的訊息是印到 stderr 的，那些正是最需要讀懂的字。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

#: 一筆偵測結果。四支 lab 都用同一個形狀，lab4 看到的還是它。
Detection = namedtuple("Detection", "x1 y1 x2 y2 score name")

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


# --- 1. 參數 ---------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Lab 1：對一批照片做 YOLO 推論")
    p.add_argument("--source", required=True, help="圖片檔或資料夾")
    p.add_argument("--weights", default="models/yolo11n.pt", help="YOLO 權重")
    p.add_argument("--conf", type=float, default=0.25, help="YOLO 信心度門檻")
    p.add_argument("--out", default="runs", help="輸出根目錄")
    p.add_argument("--headless", action="store_true", help="不開視窗，只存檔")
    p.add_argument("--dry-run", action="store_true",
                   help="不載模型、不寫檔，只走完流程（還沒有權重時用）")
    p.add_argument("--reload-each", action="store_true",
                   help="每張圖都重新載入模型，用來量載入成本")
    return p.parse_args(argv)


# --- 2. 載入模型 -----------------------------------------------------------

def load_model(weights):
    """載入 YOLO 並跑一張假圖 warmup，回傳 (model, 耗時 ms)。

    warmup 的理由：第一次推論要付 lazy 初始化的成本（配置記憶體、建 kernel）。
    把它挪到啟動階段，之後每張圖的延遲才穩定，量出來的數字才有意義。

    import 放在函式裡，--dry-run 就完全不會碰到 ultralytics / torch。
    """
    from ultralytics import YOLO

    t0 = time.perf_counter()
    model = YOLO(weights)
    model.predict(np.zeros((640, 640, 3), np.uint8), verbose=False)
    return model, (time.perf_counter() - t0) * 1000.0


# --- 3. 影像來源 -----------------------------------------------------------

def collect_images(source):
    """把 --source 展開成一串路徑。是檔案就一張，是資料夾就整批（排序後）。"""
    path = Path(source)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise SystemExit(f"找不到來源：{source}")
    files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise SystemExit(f"{source} 裡沒有圖片")
    return files


# --- 4. 推論 ---------------------------------------------------------------

def detect(model, frame, conf):
    """跑一張圖，回傳 (detections, 延遲 ms)。"""
    t0 = time.perf_counter()
    result = model.predict(frame, conf=conf, verbose=False)[0]
    latency_ms = (time.perf_counter() - t0) * 1000.0

    dets = []
    for box in result.boxes:
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
        dets.append(Detection(x1, y1, x2, y2,
                              float(box.conf[0]), result.names[int(box.cls[0])]))
    return dets, latency_ms


# --- 5. 畫框 ---------------------------------------------------------------

COLORS = [(66, 135, 245), (80, 220, 100), (60, 76, 231), (200, 160, 40)]  # BGR
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _color(name):
    # 用字串的位元組和挑顏色，不用 hash()：hash 每次啟動都不同，顏色會跳。
    return COLORS[sum(name.encode()) % len(COLORS)]


def draw(frame, dets):
    """回傳畫好標註的新影像；不修改輸入。

    只用 cv2，不拉繪圖套件。像 matplotlib 這種通用繪圖庫每張圖要多花
    100~300 ms 在建 figure 上；純 cv2 畫一張 1080p 約 2~5 ms。
    """
    out = frame.copy()
    for d in dets:
        x1, y1, x2, y2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)
        color = _color(d.name)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        text = f"{d.name} {d.score * 100:.0f}%"
        (tw, th), base = cv2.getTextSize(text, FONT, 0.5, 1)
        ty = max(y1, th + base + 2)          # 框貼在畫面頂端時把字壓進畫面內
        cv2.rectangle(out, (x1, ty - th - base - 2), (x1 + tw + 4, ty), color, -1)
        cv2.putText(out, text, (x1 + 2, ty - base), FONT, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return out


# --- 6. 存檔 ---------------------------------------------------------------

def open_run(out_root):
    """每次執行開一個 runs 底下的時間戳資料夾。"""
    run_dir = Path(out_root) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save(run_dir, image, dets, label, latency_ms):
    """存一張標註圖，並在 index.jsonl 追加一行紀錄。回傳檔案路徑。

    先落地本機、不碰網路：存檔要能在沒網路、沒金鑰的情況下成功。把上傳或推播
    放進偵測迴圈，網路一慢整條管線就跟著卡住。

    索引用 jsonl（一行一筆）而不是一整包 json：程式中途被 Ctrl-C，已經寫進去的
    行仍然是合法內容，不會整個檔案壞掉。
    """
    safe = "".join(c for c in label if c.isalnum() or c in "-_")[:40]
    name = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + safe + ".jpg"
    cv2.imwrite(str(run_dir / name), image)

    record = {
        "file": name,
        "label": label,
        "time": datetime.now().isoformat(timespec="seconds"),
        "count": len(dets),
        "best_score": round(max((d.score for d in dets), default=0.0), 4),
        "classes": sorted({d.name for d in dets}),
        "latency_ms": round(latency_ms, 1),
    }
    with (run_dir / "index.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return run_dir / name


# --- 7. 主迴圈 -------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    files = collect_images(args.source)
    print(f"來源 {args.source}：{len(files)} 張")

    model, load_ms = None, 0.0
    if not args.dry_run and not args.reload_each:
        model, load_ms = load_model(args.weights)
        print(f"模型載入 + warmup：{load_ms:.0f} ms（整批只付一次）")

    run_dir = None if args.dry_run else open_run(args.out)
    done, skipped, latencies, loads = 0, 0, [], []

    for path in files:
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"  略過讀不進來的檔案：{path.name}")   # 一張壞圖不該中斷整批
            skipped += 1
            continue

        if args.reload_each:
            # 刻意每張重建一次，用來量出「沒做單例」的代價。
            model, load_ms = load_model(args.weights)
            loads.append(load_ms)

        dets, latency = ([], 0.0) if args.dry_run else detect(model, frame, args.conf)
        latencies.append(latency)
        overlay = draw(frame, dets)
        done += 1

        best = max((d.score for d in dets), default=0.0)
        line = (f"[{done}/{len(files)}] {path.name}  {len(dets)} 處"
                f"  最高 {best:.2f}  推論 {latency:.0f} ms")
        if args.reload_each:
            line += f"  載入 {load_ms:.0f} ms"
        print(line)

        # 圖片模式一張不漏：每一張都存，不看門檻、沒有冷卻。
        if run_dir is not None:
            save(run_dir, overlay, dets, path.stem, latency)

        if not args.headless:
            cv2.imshow("lab1 - any key = next, q = quit", overlay)
            if (cv2.waitKey(0) & 0xFF) in (ord("q"), 27):
                print("使用者中斷")
                break

    if not args.headless:
        cv2.destroyAllWindows()

    # 結束摘要。量測是交付物的一部分，不是只有 debug 時才看的東西。
    avg = sum(latencies) / len(latencies) if latencies else 0.0
    print(f"\n處理 {done} 張、略過 {skipped} 張，平均推論 {avg:.0f} ms")
    if loads:
        print(f"每張重建模型共花 {sum(loads) / 1000:.1f} s"
              f"（平均 {sum(loads) / len(loads):.0f} ms/張），這就是不做單例的代價")
    print(f"輸出：{run_dir}" if run_dir else "--dry-run：沒有寫入任何檔案")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
