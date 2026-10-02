"""Lab 2：連續影像 —— 一條主迴圈，每 N 幀推論一次，超過門檻才存檔。

    python labs\\lab2_loop.py --source data\\samples\\test.mp4 --every 5
    python labs\\lab2_loop.py --source data\\samples\\test.mp4 --headless --max-frames 60
    python labs\\lab2_loop.py --source 0 --every 3          # 0 = 筆電內建鏡頭

和 lab1 的差別（段落編號一樣，可以直接 diff）：

  3. 影像來源   改成 cv2.VideoCapture，會一直吐影格；讀完可以 --loop 循環
  6. 存檔       多了門檻與冷卻：不是每一幀都存，不然幾秒就塞滿硬碟
  7. 主迴圈     從 for 變成 while，多了 --every N、鍵盤、狀態列

這一章的語意是「連續模式」：**處理不完的舊畫面就丟掉**。
和 lab1 的「一張不漏」剛好相反 —— 這兩種語意不能混在同一個容易誤按的開關
裡，所以做成兩支不同的 lab。

為什麼是「每 N 幀推論一次」而不是「把每一幀都排進佇列」：推論比抓幀慢，排隊
只會讓延遲一路累積，看到的結果離現場越來越遠。畫面要跟得上現實，就必須丟掉
來不及處理的幀。

沒有 import threading。整支程式只有一條 while。
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

Detection = namedtuple("Detection", "x1 y1 x2 y2 score name")


# --- 1. 參數 ---------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Lab 2：對連續影像做 YOLO 推論")
    p.add_argument("--source", required=True, help="影片檔，或鏡頭編號（0、1…）")
    p.add_argument("--weights", default="models/yolo11n.pt", help="YOLO 權重")
    p.add_argument("--conf", type=float, default=0.25, help="YOLO 信心度門檻")
    p.add_argument("--out", default="runs", help="輸出根目錄")
    p.add_argument("--every", type=int, default=5,
                   help="每 N 幀推論一次；其餘幀只顯示，讓畫面保持流暢")
    p.add_argument("--threshold", type=float, default=0.8,
                   help="自動存檔的信心度門檻（比 --conf 嚴，只存有把握的）")
    p.add_argument("--cooldown", type=float, default=3.0,
                   help="兩次自動存檔之間至少間隔幾秒")
    p.add_argument("--loop", action="store_true", help="影片播完從頭再來")
    p.add_argument("--headless", action="store_true", help="不開視窗，只存檔")
    p.add_argument("--dry-run", action="store_true", help="不載模型、不寫檔")
    p.add_argument("--max-frames", type=int, default=0,
                   help="讀滿幾幀就停（0 = 不限制）。無人值守測試用")
    return p.parse_args(argv)


# --- 2. 載入模型 -----------------------------------------------------------

def load_model(weights):
    """載入 YOLO 並跑一張假圖 warmup，回傳 (model, 耗時 ms)。"""
    from ultralytics import YOLO

    t0 = time.perf_counter()
    model = YOLO(weights)
    model.predict(np.zeros((640, 640, 3), np.uint8), verbose=False)
    return model, (time.perf_counter() - t0) * 1000.0


# --- 3. 影像來源 -----------------------------------------------------------

def open_source(source):
    """開啟影片檔或鏡頭，回傳 (cap, fps)。失敗就直接結束程式。

    純數字當鏡頭編號，其他當檔案路徑 —— 這是 OpenCV 的慣例，照著做學生比較
    容易把在網路上看到的範例接上來。
    """
    target = int(source) if str(source).isdigit() else str(source)
    if isinstance(target, str) and not Path(target).exists():
        raise SystemExit(f"找不到影片檔：{target}")

    cap = cv2.VideoCapture(target)
    if not cap.isOpened():
        raise SystemExit(f"無法開啟來源：{target}")
    # 只留最新一格。即時來源的解碼器 buffer 裡放的是舊資料，堆著只會增加延遲。
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    return cap, (fps if 1 <= fps <= 120 else 30.0)


def read_frame(cap, loop):
    """讀下一格。來源結束回 None；loop=True 的影片檔會從頭再來。"""
    ok, frame = cap.read()
    if ok:
        return frame
    if not loop:
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, frame = cap.read()
    return frame if ok else None


# --- 4. 推論 ---------------------------------------------------------------

def detect(model, frame, conf):
    """跑一幀，回傳 (detections, 延遲 ms)。"""
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
    return COLORS[sum(name.encode()) % len(COLORS)]


def draw(frame, dets):
    """回傳畫好標註的新影像；不修改輸入。"""
    out = frame.copy()
    for d in dets:
        x1, y1, x2, y2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)
        color = _color(d.name)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        text = f"{d.name} {d.score * 100:.0f}%"
        (tw, th), base = cv2.getTextSize(text, FONT, 0.5, 1)
        ty = max(y1, th + base + 2)
        cv2.rectangle(out, (x1, ty - th - base - 2), (x1 + tw + 4, ty), color, -1)
        cv2.putText(out, text, (x1 + 2, ty - base), FONT, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return out


def draw_status(frame, text):
    """把狀態列畫在畫面下緣（就地修改，這張影像只是要顯示而已）。"""
    h = frame.shape[0]
    cv2.rectangle(frame, (0, h - 24), (frame.shape[1], h), (0, 0, 0), -1)
    cv2.putText(frame, text, (6, h - 7), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


# --- 6. 存檔 ---------------------------------------------------------------

def open_run(out_root):
    run_dir = Path(out_root) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save(run_dir, image, dets, label, latency_ms):
    """存一張標註圖，並在 index.jsonl 追加一行紀錄。"""
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

    model, load_ms = None, 0.0
    if not args.dry_run:
        model, load_ms = load_model(args.weights)
        print(f"模型載入 + warmup：{load_ms:.0f} ms")

    cap, src_fps = open_source(args.source)
    run_dir = None if args.dry_run else open_run(args.out)
    print(f"來源 {args.source}（{src_fps:.0f} fps）"
          f"　每 {args.every} 幀推論一次　門檻 {args.threshold}"
          f"　冷卻 {args.cooldown} 秒")
    if not args.headless:
        print("鍵盤：q 離開　p 拍一張（不看門檻、不受冷卻限制）")

    # 不開視窗時不需要節流，盡快跑完；開視窗時照影片 fps 播，否則會像快轉。
    delay_ms = max(1, int(1000 / src_fps))

    idx = inferred = skipped = hits = snaps = 0
    last_save = -1e9
    latencies = []
    started = time.perf_counter()

    try:
        while True:
            # 上限檢查放在讀取「之前」：否則會多讀一幀卻不處理，計數也會多一。
            if args.max_frames and idx >= args.max_frames:
                print(f"已達 --max-frames {args.max_frames}")
                break
            frame = read_frame(cap, args.loop)
            if frame is None:
                print("來源結束")
                break
            idx += 1

            # 這一幀要不要跑模型。跳過的幀直接顯示，畫面才跟得上來源。
            run_now = (idx % args.every == 0) and not args.dry_run
            if run_now:
                dets, latency = detect(model, frame, args.conf)
                latencies.append(latency)
                inferred += 1
                shown = draw(frame, dets)
            else:
                dets, latency = [], 0.0
                skipped += 1
                shown = frame.copy()

            best = max((d.score for d in dets), default=0.0)
            elapsed = time.perf_counter() - started
            status = (f"#{idx}  {'辨識' if run_now else '僅顯示'}"
                      f"  {len(dets)} 處 最高 {best:.2f}"
                      f"  {latency:.0f} ms  讀取 {idx / max(elapsed, 1e-6):.1f} fps"
                      f"  命中 {hits}")

            # 自動存檔：門檻 + 冷卻。沒有冷卻的話，對著同一個目標幾秒就存上百張。
            if run_now and best > args.threshold and \
                    (time.perf_counter() - last_save) >= args.cooldown:
                save(run_dir, shown, dets, f"frame{idx:06d}", latency)
                last_save = time.perf_counter()
                hits += 1
                status += "  [已存]"

            print(status)

            if args.headless:
                continue

            cv2.imshow("lab2 - q quit / p snapshot", draw_status(shown, status))
            key = cv2.waitKey(delay_ms) & 0xFF
            if key in (ord("q"), 27):
                print("使用者離開")
                break
            if key == ord("p"):
                # 手動拍一張：明確的指令，所以不看門檻也不受冷卻限制。
                # 這一幀若還沒跑過模型，就單獨為它跑一次再存 —— 使用者要的是
                # 有標註的結果，不是一張原始畫面。
                if not run_now and not args.dry_run:
                    dets, latency = detect(model, frame, args.conf)
                    shown = draw(frame, dets)
                if run_dir is not None:
                    path = save(run_dir, shown, dets, f"snap{idx:06d}", latency)
                    snaps += 1
                    print(f"  拍一張 -> {path.name}")
    except KeyboardInterrupt:
        print("\nCtrl-C 中斷")
    finally:
        # 不管怎麼離開都要收乾淨。這個習慣到 lab3 會變成「一定要降落／斷線」。
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()

    avg = sum(latencies) / len(latencies) if latencies else 0.0
    print(f"\n讀 {idx} 幀，推論 {inferred} 幀，跳過 {skipped} 幀"
          f"（--every {args.every}），平均推論 {avg:.0f} ms")
    print(f"自動命中 {hits} 張，手動拍 {snaps} 張")
    print(f"輸出：{run_dir}" if run_dir else "--dry-run：沒有寫入任何檔案")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
