"""Lab 3：整合 —— Tello 影像 + 飛行操控 + 按 P 拍照後做 YOLO 辨識。

    python labs\\lab3_detect.py                      # 連線、開畫面、可操控、P 辨識
    python labs\\lab3_detect.py --source video --path data\\samples\\test.mp4

把 lab1 的操控和 lab2 的影像接起來，再加上辨識。

## 辨識只在按下 P 的時候做，**不做連續辨識**

這是這一支最重要的設計。畫面以來源原本的速度跑（實測 Tello 約 20 fps），
模型完全不介入；只有你按下 P，程式才對**那一張**跑一次 YOLO。

為什麼不邊飛邊連續辨識？

  * CPU 上 yolo26n 一張要 50~60 ms，連續跑會把畫面更新率壓到個位數 fps。
    那個更新率沒辦法操控無人機 —— 等你從畫面看到障礙物，早就撞上了。
  * 連續辨識還要處理「門檻多高才算命中」「多久存一次才不會塞爆硬碟」這些問題，
    對第一次寫這種程式的人是額外的負擔。

所以規則很簡單：**你決定什麼時候辨識。** 按 P = 拍照 + 辨識 + 存檔，一次做完。

## 安全規則

1. **起飛前必須先看到畫面。** 還沒有畫面就按 T 會被拒絕並印出原因。
   沒有畫面等於盲飛。
2. **降落 / 離開是兩件事。** L 是降落，Q 是離開程式（離開前會自動降落）。
3. **離開一定收乾淨。** 不管是 Q、Ctrl-C 還是程式炸掉，finally 都會：
   rc 歸零 -> 還在飛就降落 -> streamoff -> end()。

段落順序（和 lab2 對照著看，多了 2 和 4）：
    1. 參數   2. 載入模型   3. 連線與影像   4. 推論
    5. 畫框   6. 存檔      7. 主迴圈

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

#: 教材實測的 Python 是 3.12。下限來自 pillow（>=3.10），上限來自 torch 2.7.0
#: —— 它只發布到 Python 3.13 的 wheel，3.14 以上根本裝不起來。
if not ((3, 10) <= sys.version_info[:2] <= (3, 13)):
    raise SystemExit(
        f"需要 Python 3.10 ~ 3.13，目前是 {sys.version.split()[0]}。\n"
        "  建議用 3.12（教材實測的版本）。Windows 上可以用啟動器指定：\n"
        "      py -3.12 -m venv .venv")

#: 一筆偵測結果。lab2 的 index.jsonl 欄位形狀一樣，只是那時候還是空的。
Detection = namedtuple("Detection", "x1 y1 x2 y2 score name")

#: Tello SDK 建議 rc 指令不要超過 20 Hz。灌太快會塞爆指令通道，反而讓起飛／
#: 降落這類需要回應的指令被拖延。
RC_INTERVAL_S = 0.05

STATE_INTERVAL_S = 1.0
FIRST_FRAME_TIMEOUT_S = 10.0
MIN_BATTERY_PCT = 30

#: Windows 上 cv2.waitKeyEx 的方向鍵回傳值。這組值不跨平台，所以另外留了
#: 8/2/4/6 當備援 —— 「函式庫的回傳值不保證跨平台」本身就是一課。
ARROW_UP, ARROW_DOWN, ARROW_LEFT, ARROW_RIGHT = 2490368, 2621440, 2424832, 2555904

FONT = cv2.FONT_HERSHEY_SIMPLEX


# --- 1. 參數 ---------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Lab 3：Tello 影像 + 操控 + 按 P 拍照辨識")
    p.add_argument("--source", default="tello", choices=("tello", "video"),
                   help="影像來源。video 是沒有無人機時的練習用來源")
    p.add_argument("--path", help="--source video 時的影片檔或鏡頭編號")
    p.add_argument("--weights", default="models/yolo26n.pt", help="YOLO 權重")
    p.add_argument("--conf", type=float, default=0.25, help="YOLO 信心度門檻")
    p.add_argument("--out", default="runs", help="輸出根目錄")
    p.add_argument("--speed", type=int, default=40,
                   help="rc 速度 0~100。第一次實飛請用 20")
    p.add_argument("--no-model", action="store_true",
                   help="不載模型，P 只拍照不辨識（等同 lab2 的行為）")
    p.add_argument("--no-land-on-exit", action="store_true",
                   help="離開時不自動降落，保持懸停（你要自己接手）")
    p.add_argument("--allow-emergency", action="store_true",
                   help="開啟 X 鍵＝切斷馬達。機體會直接墜落，只在必要時用")
    p.add_argument("--headless", action="store_true",
                   help="不開視窗（自動測試用）")
    p.add_argument("--max-frames", type=int, default=0,
                   help="讀滿幾幀就停（0 = 不限）。無人值守測試用")
    p.add_argument("--snap-every", type=int, default=0,
                   help="每 N 幀自動觸發一次拍照辨識（0 = 關）。"
                        "給 --headless 測試用，正常操作請按 P")
    args = p.parse_args(argv)
    if args.source == "video" and not args.path:
        p.error("--source video 需要 --path")
    return args


# --- 2. 載入模型 -----------------------------------------------------------

def load_model(weights):
    """載入 YOLO 並跑一張假圖 warmup，回傳 (model, 耗時 ms)。

    **在主迴圈開始前就載好**，不要等第一次按 P 才載 —— 不然你按下去要等三秒
    才有反應，而那三秒飛機還在空中。warmup 同理：把 lazy 初始化的成本付在
    啟動階段。

    import 放在函式裡，--no-model 就完全不會碰到 ultralytics / torch。
    """
    from ultralytics import YOLO

    t0 = time.perf_counter()
    model = YOLO(weights)
    model.predict(np.zeros((640, 640, 3), np.uint8), verbose=False)
    return model, (time.perf_counter() - t0) * 1000.0


# --- 3. 連線與影像 ---------------------------------------------------------

def open_source(args):
    """開啟影像來源，回傳 (kind, handle, frame_read)。"""
    if args.source == "video":
        target = int(args.path) if str(args.path).isdigit() else str(args.path)
        if isinstance(target, str) and not Path(target).exists():
            raise SystemExit(f"找不到影片檔：{target}")
        cap = cv2.VideoCapture(target)
        if not cap.isOpened():
            raise SystemExit(f"無法開啟來源：{target}")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return "video", cap, None

    from djitellopy import Tello

    tello = Tello()
    try:
        tello.connect()
    except Exception as exc:
        raise SystemExit(
            f"連不上 Tello：{exc}\n"
            "  1. 電腦的 Wi-Fi 要連到 TELLO-XXXXXX 這個熱點\n"
            "  2. UDP 8889 沒被占用（另一個程式還開著就會占著）") from exc

    print(f"已連線　電量 {tello.get_battery()}%　溫度 {tello.get_temperature():.0f}C")
    tello.streamon()
    tello.set_speed(10)
    return "tello", tello, tello.get_frame_read()


def read_frame(kind, handle, frame_read):
    """讀最新一格 BGR。來源結束回 None，還沒準備好回 "wait"。

    djitellopy 給的是 RGB，cv2 全家吃 BGR —— 不轉的話顏色會反，而且送進 YOLO
    的張量跟訓練時不一致，分數會怪。
    """
    if kind == "video":
        ok, frame = handle.read()
        return frame if ok else None

    frame = frame_read.frame
    if frame is None:
        return "wait"
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def close_source(kind, handle, land_first):
    """收尾：rc 歸零 -> 需要就降落 -> streamoff -> end()。"""
    if kind == "video":
        handle.release()
        return
    try:
        handle.send_rc_control(0, 0, 0, 0)
    except Exception:
        pass
    if land_first:
        print("離開前自動降落…")
        try:
            handle.land()
        except Exception as exc:
            print(f"  降落指令被拒：{exc}（機體可能已經在地上）")
    for step in ("streamoff", "end"):
        try:
            getattr(handle, step)()
        except Exception:
            pass


# --- 4. 推論 ---------------------------------------------------------------

def detect(model, frame, conf):
    """對一張影像跑 YOLO，回傳 (detections, 延遲 ms)。

    只有按下 P 的時候才會呼叫這個函式 —— 主迴圈不碰它。
    """
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


def _color(name):
    # 用字串的位元組和挑顏色，不用 hash()：hash 每次啟動都不同，顏色會跳。
    return COLORS[sum(name.encode()) % len(COLORS)]


def draw(frame, dets):
    """回傳畫好標註的新影像；不修改輸入。

    不修改輸入很重要：主迴圈要繼續用原始那一張去顯示，標註版只是拿去存檔。
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


def draw_status(frame, text):
    """把狀態列畫在畫面下緣。飛行時這行字比偵測框重要。"""
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
    """存一張標註圖，並在 index.jsonl 追加一行紀錄。回傳檔案路徑。

    欄位和 lab2 一模一樣，只是 count / classes / latency_ms 現在真的有值了。
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

def snap(model, frame, run_dir, conf, idx):
    """按 P 做的事：拍照 -> 辨識 -> 存檔。回傳 (檔名, 偵測數, 最高分)。

    這是整支程式唯一會跑模型的地方。
    """
    if model is None:
        path = save(run_dir, frame, [], f"snap{idx:06d}", 0.0)
        print(f"拍一張 -> {path.name}　（--no-model，沒有做辨識）")
        return path.name, 0, 0.0
    dets, latency = detect(model, frame, conf)
    overlay = draw(frame, dets)
    path = save(run_dir, overlay, dets, f"snap{idx:06d}", latency)
    best = max((d.score for d in dets), default=0.0)
    names = "、".join(sorted({d.name for d in dets})) or "（沒偵測到東西）"
    print(f"拍一張 -> {path.name}　{len(dets)} 處　最高 {best:.2f}"
          f"　{latency:.0f} ms　{names}")
    return path.name, len(dets), best


def main(argv=None):
    args = parse_args(argv)

    # 模型先載好，不要等按下 P 才載 —— 那時候飛機還在空中。
    model = None
    if not args.no_model:
        model, load_ms = load_model(args.weights)
        print(f"模型載入 + warmup：{load_ms:.0f} ms（之後按 P 不用再等）")

    kind, handle, frame_read = open_source(args)
    run_dir = open_run(args.out)
    print(f"來源 {kind}　輸出 {run_dir}")
    if not args.headless:
        print("鍵盤：P 拍照並辨識　T 起飛　L 降落　空白鍵懸停　Q 離開")
        if kind == "tello":
            print("　　　方向鍵或 8/2/4/6 移動　W/S 升降　A/D 轉向"
                  "（按一下持續移動，按空白鍵才停）")

    flying = False
    lr = fb = ud = yaw = 0
    idx = snaps = 0
    hits = 0
    state_text = ""
    last_state = last_rc = -1e9
    seen_frame = False
    deadline = time.perf_counter() + FIRST_FRAME_TIMEOUT_S

    try:
        while True:
            if args.max_frames and idx >= args.max_frames:
                print(f"已達 --max-frames {args.max_frames}")
                break

            frame = read_frame(kind, handle, frame_read)
            if isinstance(frame, str):       # "wait"
                if time.perf_counter() > deadline:
                    print(f"{FIRST_FRAME_TIMEOUT_S:.0f} 秒內沒有畫面，請重跑")
                    break
                time.sleep(0.05)
                continue
            if frame is None:
                print("來源結束")
                break
            seen_frame = True
            idx += 1

            if kind == "tello" and \
                    time.perf_counter() - last_state >= STATE_INTERVAL_S:
                last_state = time.perf_counter()
                try:
                    state_text = (f"　電量 {handle.get_battery()}%"
                                  f" 高度 {handle.get_height()} cm")
                except Exception:
                    state_text = "　狀態讀取失敗"

            status = (f"#{idx}{state_text}　{'飛行中' if flying else '待命'}"
                      f"　已拍 {snaps} 張")

            # --snap-every 是給 --headless 自動測試用的，正常操作請按 P
            if args.snap_every and idx % args.snap_every == 0:
                _, n, _ = snap(model, frame, run_dir, args.conf, idx)
                snaps += 1
                hits += 1 if n else 0

            if args.headless:
                continue

            cv2.imshow(f"lab3 - {kind}  P=拍照辨識  Q=離開",
                       draw_status(frame.copy(), status))
            key = cv2.waitKeyEx(1)

            if key in (ord("q"), ord("Q"), 27):
                print("使用者離開")
                break

            if key in (ord("p"), ord("P")):
                _, n, _ = snap(model, frame, run_dir, args.conf, idx)
                snaps += 1
                hits += 1 if n else 0

            if kind != "tello":
                continue

            # -- 以下只有 Tello 來源才有意義 --------------------------------
            if key in (ord("t"), ord("T")):
                if not seen_frame:
                    # 安全規則 1。不自動代開串流，也不硬起飛。
                    print("尚未看到畫面，取消起飛。沒有畫面等於盲飛。")
                elif flying:
                    print("已在飛行中，忽略起飛")
                else:
                    try:
                        pct = handle.get_battery()
                    except Exception:
                        pct = 0
                    if pct < MIN_BATTERY_PCT:
                        print(f"電量只有 {pct}%，低於 {MIN_BATTERY_PCT}% 不起飛")
                    else:
                        try:
                            handle.takeoff()
                            flying = True
                            lr = fb = ud = yaw = 0
                            print("起飛")
                        except Exception as exc:
                            print(f"起飛失敗：{exc}{state_text}")
            elif key in (ord("l"), ord("L")):
                lr = fb = ud = yaw = 0
                try:
                    handle.send_rc_control(0, 0, 0, 0)
                    handle.land()
                    flying = False
                    print("降落")
                except Exception as exc:
                    print(f"降落指令被拒：{exc}（再按一次 L）")
            elif key == ord(" "):
                lr = fb = ud = yaw = 0
                print("懸停")
            elif key in (ord("x"), ord("X")) and args.allow_emergency:
                print("緊急停止：切斷馬達")
                try:
                    handle.emergency()
                except Exception as exc:
                    print(f"緊急停止失敗：{exc}")
                flying = False
            elif key in (ARROW_UP, ord("8")):
                fb = args.speed
            elif key in (ARROW_DOWN, ord("2")):
                fb = -args.speed
            elif key in (ARROW_LEFT, ord("4")):
                lr = -args.speed
            elif key in (ARROW_RIGHT, ord("6")):
                lr = args.speed
            elif key in (ord("w"), ord("W")):
                ud = args.speed
            elif key in (ord("s"), ord("S")):
                ud = -args.speed
            elif key in (ord("a"), ord("A")):
                yaw = -args.speed
            elif key in (ord("d"), ord("D")):
                yaw = args.speed

            # rc 節流：每 50 ms 一次。主迴圈本身跑得比 20 Hz 快得多。
            if flying and (time.perf_counter() - last_rc) >= RC_INTERVAL_S:
                last_rc = time.perf_counter()
                try:
                    handle.send_rc_control(lr, fb, ud, yaw)
                except Exception:
                    pass

    except KeyboardInterrupt:
        print("\nCtrl-C 中斷")
    finally:
        # 安全規則 3：不管怎麼離開都收乾淨。
        if not args.headless:
            cv2.destroyAllWindows()
        close_source(kind, handle, land_first=flying and not args.no_land_on_exit)
        if flying and args.no_land_on_exit:
            print("！機體仍在空中（--no-land-on-exit），請自行接手降落")

    print(f"\n讀 {idx} 幀，拍照辨識 {snaps} 次，其中 {hits} 次有偵測到東西")
    print(f"輸出：{run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
