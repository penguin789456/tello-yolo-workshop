"""Lab 4：整合 —— 一支工具，三種來源，兩種語意。

    python labs\\lab4_patrol.py --source image --path data\\samples --headless
    python labs\\lab4_patrol.py --source video --path data\\samples\\test.mp4 --max-frames 50
    python labs\\lab4_patrol.py --source tello --preview-only

前三章各自能跑，但使用者不會想記三支程式。這一章把它們收斂成單一入口，
並且讓**兩種相反的語意共存於同一支程式**：

  | 來源            | is_live | 行為                                      |
  |-----------------|---------|-------------------------------------------|
  | image           | False   | 一張不漏：每張都推論、每張都存，沒有冷卻  |
  | video / tello   | True    | 丟得起就丟：每 N 幀推論，門檻＋冷卻才存檔 |

「丟不丟畫面」這兩種行為剛好相反，做成一個容易誤按的開關太危險。我們讓它綁在
--source 上 —— 來源的性質本來就決定了正確的語意 —— 並且在啟動時把生效的語意
印出來：使用者要能看到程式打算怎麼做，而不是猜。

段落編號與前三章相同：
    1. 參數   2. 載入模型   3. 影像來源   4. 推論
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

Detection = namedtuple("Detection", "x1 y1 x2 y2 score name")

#: 一個影像來源。kind 決定 read/close 怎麼做，is_live 決定要不要丟幀。
#: 用 namedtuple + 分支而不是一整套 class 繼承：這支程式要能從上讀到下。
Source = namedtuple("Source", "kind handle fps is_live extra")

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")

RC_INTERVAL_S = 0.05
STATE_INTERVAL_S = 1.0
FIRST_FRAME_TIMEOUT_S = 10.0
ARROW_UP, ARROW_DOWN, ARROW_LEFT, ARROW_RIGHT = 2490368, 2621440, 2424832, 2555904


# --- 1. 參數 ---------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Lab 4：影像巡檢 CLI")
    p.add_argument("--source", required=True, choices=("image", "video", "tello"),
                   help="來源種類。image 一張不漏；video/tello 會丟幀")
    p.add_argument("--path", help="image：圖片檔或資料夾；video：影片檔或鏡頭編號")
    p.add_argument("--weights", default="models/yolo11n.pt", help="YOLO 權重")
    p.add_argument("--conf", type=float, default=0.25, help="YOLO 信心度門檻")
    p.add_argument("--out", default="runs", help="輸出根目錄")
    p.add_argument("--every", type=int, default=5,
                   help="即時來源每 N 幀推論一次（image 一律為 1）")
    p.add_argument("--threshold", type=float, default=0.8, help="自動存檔門檻")
    p.add_argument("--cooldown", type=float, default=3.0, help="自動存檔冷卻秒數")
    p.add_argument("--save-all", action="store_true",
                   help="每張推論過的畫面都存，不看門檻（會吃硬碟）")
    p.add_argument("--loop", action="store_true", help="影片播完從頭再來")
    p.add_argument("--headless", action="store_true", help="不開視窗，只存檔")
    p.add_argument("--dry-run", action="store_true",
                   help="不載模型、不寫任何檔案，只驗證流程跑得完")
    p.add_argument("--max-frames", type=int, default=0, help="讀滿幾幀就停（0=不限）")
    p.add_argument("--speed", type=int, default=40, help="Tello rc 速度 0~100")
    p.add_argument("--preview-only", action="store_true",
                   help="完全不載模型、不推論，只看畫面。飛行時用這個")
    p.add_argument("--no-land-on-exit", action="store_true",
                   help="Tello：離開時不自動降落（你要自己接手）")
    p.add_argument("--allow-emergency", action="store_true",
                   help="Tello：開啟 X 鍵＝切斷馬達")
    args = p.parse_args(argv)
    if args.source in ("image", "video") and not args.path:
        p.error(f"--source {args.source} 需要 --path")
    return args


# --- 2. 載入模型 -----------------------------------------------------------

def load_model(weights):
    """載入 YOLO 並跑一張假圖 warmup，回傳 (model, 耗時 ms)。"""
    from ultralytics import YOLO

    t0 = time.perf_counter()
    model = YOLO(weights)
    model.predict(np.zeros((640, 640, 3), np.uint8), verbose=False)
    return model, (time.perf_counter() - t0) * 1000.0


# --- 3. 影像來源 -----------------------------------------------------------

def open_source(args):
    """依 --source 開啟來源。三種來源在這裡收斂成同一個 Source 形狀。"""
    if args.source == "image":
        path = Path(args.path)
        if path.is_file():
            paths = [path]
        elif path.is_dir():
            paths = sorted(p for p in path.iterdir()
                           if p.suffix.lower() in IMAGE_SUFFIXES)
        else:
            raise SystemExit(f"找不到來源：{args.path}")
        if not paths:
            raise SystemExit(f"{args.path} 裡沒有圖片")
        # handle 用 dict 是因為要記住讀到第幾張；is_live=False 代表「會結束」
        return Source("image", {"paths": paths, "i": 0}, 0.0, False, None)

    if args.source == "video":
        target = int(args.path) if str(args.path).isdigit() else str(args.path)
        if isinstance(target, str) and not Path(target).exists():
            raise SystemExit(f"找不到影片檔：{target}")
        cap = cv2.VideoCapture(target)
        if not cap.isOpened():
            raise SystemExit(f"無法開啟來源：{target}")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # 只留最新一格，不讓 buffer 堆舊資料
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        return Source("video", cap, fps if 1 <= fps <= 120 else 30.0, True, None)

    from djitellopy import Tello

    tello = Tello()
    try:
        tello.connect()
    except Exception as exc:
        raise SystemExit(
            f"連不上 Tello：{exc}\n"
            "  1. 電腦的 Wi-Fi 要連到 TELLO-XXXXXX 熱點\n"
            "  2. UDP 8889 沒被其他程式占用") from exc
    print(f"已連線　電量 {tello.get_battery()}%")
    tello.streamon()
    tello.set_speed(10)
    return Source("tello", tello, 30.0, True, tello.get_frame_read())


def read_frame(source, loop):
    """讀下一格。來源結束回 None；還沒準備好回 "wait"（只有 tello 會）。"""
    if source.kind == "image":
        h = source.handle
        while h["i"] < len(h["paths"]):
            path = h["paths"][h["i"]]
            h["i"] += 1
            frame = cv2.imread(str(path))
            if frame is not None:
                return frame, path.stem
            print(f"  略過讀不進來的檔案：{path.name}")   # 壞圖不中斷整批
        return None, ""

    if source.kind == "video":
        ok, frame = source.handle.read()
        if not ok and loop:
            source.handle.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = source.handle.read()
        return (frame, "") if ok else (None, "")

    # tello：frame_read.frame 是最新一格的快照，而且是 RGB，要轉 BGR。
    frame = source.extra.frame
    if frame is None:
        return "wait", ""
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), ""


def close_source(source, land_first):
    """收尾。不管程式怎麼結束都要走到這裡。"""
    if source.kind == "video":
        source.handle.release()
    elif source.kind == "tello":
        tello = source.handle
        try:
            tello.send_rc_control(0, 0, 0, 0)
        except Exception:
            pass
        if land_first:
            print("離開前自動降落…")
            try:
                tello.land()
            except Exception as exc:
                print(f"  降落指令被拒：{exc}（機體可能已在地上）")
        for step in ("streamoff", "end"):
            try:
                getattr(tello, step)()
            except Exception:
                pass


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
    """把狀態列畫在畫面下緣。"""
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
    """存一張標註圖，並在 index.jsonl 追加一行紀錄。回傳檔案路徑。"""
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

    model = None
    if not args.dry_run and not args.preview_only:
        model, load_ms = load_model(args.weights)
        print(f"模型載入 + warmup：{load_ms:.0f} ms")

    source = open_source(args)

    # 兩種語意在這裡分岔，而且印出來給使用者看，不讓他猜。
    if source.is_live:
        every, cooldown = max(1, args.every), args.cooldown
        print(f"即時來源（{source.kind}）：每 {every} 幀推論一次，"
              f"門檻 {args.threshold}，冷卻 {cooldown} 秒，來不及的幀會丟掉")
    else:
        every, cooldown = 1, 0.0
        print(f"有限來源（{source.kind}）：每一張都推論、每一張都存，一張不漏")
    if args.source == "tello":
        print("鍵盤：T 起飛　L 降落　空白鍵懸停　P 拍一張　Q 離開")
        if not args.preview_only:
            print("提醒：辨識模式畫面會變慢，飛行操控請加 --preview-only")

    run_dir = None if args.dry_run else open_run(args.out)
    delay_ms = max(1, int(1000 / source.fps)) if source.fps else 1

    flying = False
    lr = fb = ud = yaw = 0
    idx = inferred = skipped = hits = snaps = 0
    last_save = last_rc = last_state = -1e9
    state_text = ""
    latencies = []
    deadline = time.perf_counter() + FIRST_FRAME_TIMEOUT_S
    seen_frame = False
    started = time.perf_counter()

    try:
        while True:
            # 上限檢查放在讀取「之前」：否則會多讀一幀卻不處理，計數也會多一。
            if args.max_frames and idx >= args.max_frames:
                print(f"已達 --max-frames {args.max_frames}")
                break
            frame, label = read_frame(source, args.loop)
            if isinstance(frame, str):        # "wait"：Tello 還沒吐出第一格
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

            run_now = (idx % every == 0) and not args.dry_run and not args.preview_only
            if run_now:
                dets, latency = detect(model, frame, args.conf)
                latencies.append(latency)
                inferred += 1
                shown = draw(frame, dets)
            else:
                dets, latency = [], 0.0
                skipped += 1
                shown = frame.copy()

            if source.kind == "tello" and \
                    time.perf_counter() - last_state >= STATE_INTERVAL_S:
                last_state = time.perf_counter()
                try:
                    state_text = (f"　電量 {source.handle.get_battery()}%"
                                  f" 高度 {source.handle.get_height()} cm")
                except Exception:
                    state_text = "　狀態讀取失敗"

            best = max((d.score for d in dets), default=0.0)
            elapsed = time.perf_counter() - started
            status = (f"#{idx}{state_text}  {'辨識' if run_now else '僅顯示'}"
                      f" {len(dets)} 處 最高 {best:.2f}  {latency:.0f} ms"
                      f"  {idx / max(elapsed, 1e-6):.1f} fps  命中 {hits}")

            # 存檔條件：有限來源每張都存；即時來源要過門檻且過了冷卻。
            should_save = run_now and (
                not source.is_live or args.save_all
                or (best > args.threshold
                    and (time.perf_counter() - last_save) >= cooldown))
            if should_save and run_dir is not None:
                save(run_dir, shown, dets, label or f"frame{idx:06d}", latency)
                last_save = time.perf_counter()
                hits += 1
                status += "  [已存]"

            print(status)

            if args.headless:
                continue

            cv2.imshow(f"lab4 - {source.kind}", draw_status(shown, status))
            key = cv2.waitKeyEx(delay_ms if source.kind == "video" else 1)

            if key in (ord("q"), ord("Q"), 27):
                print("使用者離開")
                break

            if key in (ord("p"), ord("P")):
                # 手動拍一張：不看門檻、不受冷卻限制；沒推論過就單獨跑一次。
                if not run_now and not args.dry_run and not args.preview_only:
                    dets, latency = detect(model, frame, args.conf)
                    shown = draw(frame, dets)
                if run_dir is not None:
                    path = save(run_dir, shown, dets, f"snap{idx:06d}", latency)
                    snaps += 1
                    print(f"  拍一張 -> {path.name}")

            if source.kind != "tello":
                continue

            # -- 以下只有 Tello 來源才有意義 --------------------------------
            tello = source.handle
            if key in (ord("t"), ord("T")):
                if not seen_frame:
                    print("尚未看到畫面，取消起飛。沒有畫面等於盲飛。")
                elif flying:
                    print("已在飛行中，忽略起飛")
                else:
                    try:
                        tello.takeoff()
                        flying = True
                        lr = fb = ud = yaw = 0
                        print("起飛")
                    except Exception as exc:
                        print(f"起飛失敗：{exc}{state_text}")
            elif key in (ord("l"), ord("L")):
                lr = fb = ud = yaw = 0
                try:
                    tello.send_rc_control(0, 0, 0, 0)
                    tello.land()
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
                    tello.emergency()
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

            if flying and (time.perf_counter() - last_rc) >= RC_INTERVAL_S:
                last_rc = time.perf_counter()
                try:
                    tello.send_rc_control(lr, fb, ud, yaw)
                except Exception:
                    pass

    except KeyboardInterrupt:
        print("\nCtrl-C 中斷")
    finally:
        if not args.headless:
            cv2.destroyAllWindows()
        close_source(source, land_first=flying and not args.no_land_on_exit)
        if flying and args.no_land_on_exit:
            print("！機體仍在空中（--no-land-on-exit），請自行接手降落")

    # 結束摘要：交付物的一部分。跑完要能回答「處理了多少、命中多少、多快」。
    avg = sum(latencies) / len(latencies) if latencies else 0.0
    total = time.perf_counter() - started
    print(f"\n=== 摘要 ===")
    print(f"來源 {source.kind}　讀 {idx} 幀　推論 {inferred} 幀　跳過 {skipped} 幀")
    print(f"平均推論 {avg:.0f} ms　整體 {idx / max(total, 1e-6):.1f} fps"
          f"　耗時 {total:.1f} s")
    print(f"存檔 {hits} 張　手動拍 {snaps} 張")
    print(f"輸出：{run_dir}" if run_dir else "--dry-run：沒有寫入任何檔案")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
