"""Lab 3：接上 Tello EDU —— 連線、看畫面、用鍵盤飛、按 P 拍一張。

    python labs\\lab3_tello.py --preview-only        # 先這樣，確認看得到畫面
    python labs\\lab3_tello.py --every 10            # 邊飛邊辨識（畫面會變慢）

和 lab2 的差別（段落編號一樣，可以直接 diff）：

  3. 影像來源   改成 djitellopy：connect() -> streamon() -> frame_read.frame
  7. 主迴圈     多了狀態列（電量／高度）、飛行鍵、rc 節流、finally 一定收乾淨

## 三條安全規則（不是寫在文件裡好看的，程式真的這樣做）

1. **起飛前必須先看到畫面。** 還沒有畫面就按 T 會被拒絕並印出原因，程式不會
   替你把串流打開 —— 起飛是不可逆的實體動作，不該是別的操作的副作用。沒有
   畫面等於盲飛。
2. **停止影像 / 中斷連線 / 降落是三件不同的事。** Q 只是離開程式（離開前會
   自動降落），L 是降落，斷線是程式結束才做。
3. **離開一定收乾淨。** 不管是 Q、Ctrl-C 還是程式炸掉，finally 都會：rc 歸零
   -> 還在飛就降落 -> streamoff -> end()。

    有 GUI 的程式關窗時可以「問」使用者要不要降落，因為有人看著螢幕。CLI 沒有
    人看著，程式一結束就沒人控制它了，所以預設自動降落。要保留懸停就加
    --no-land-on-exit，並且你要自己接手。

## 鍵盤（按一下就會持續移動，放開不會停）

    ↑ ↓ ← →  前進 / 後退 / 左移 / 右移      8 2 4 6  同上（跨平台備援）
    W / S     上升 / 下降                    A / D    左轉 / 右轉
    空白鍵    全部歸零（懸停）                T / L    起飛 / 降落
    P         拍一張                          Q        離開（會先降落）

cv2.waitKey 只有「按下」沒有「放開」事件，所以做不到「按住才動、放開就停」——
那需要有 key-up 事件的 GUI 框架。這裡改成速度會 latch 住：要停下來請按空白鍵或
反向鍵。**飛之前先把這件事記牢。**

沒有 import threading。整支程式只有一條 while。
（djitellopy 內部會自己開一條解碼執行緒，那是它的事，我們不寫也不管它。）
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

#: Tello SDK 建議 rc 指令不要超過 20 Hz。灌太快會塞爆指令通道，反而讓起飛／
#: 降落這類需要回應的指令被拖延。實測灌到 125 Hz 就會出現這個現象。
RC_INTERVAL_S = 0.05

#: 狀態列更新頻率。get_battery() 讀的是狀態封包快取，不送指令、不會阻塞。
STATE_INTERVAL_S = 1.0

#: streamon 之後等第一格畫面的上限。正常 2~3 秒就會有。
FIRST_FRAME_TIMEOUT_S = 10.0

#: Windows 上 cv2.waitKeyEx 的方向鍵回傳值。這組值不跨平台，所以下面另外留了
#: 8/2/4/6 當備援 —— 「函式庫的回傳值不保證跨平台」本身就是一課。
ARROW_UP, ARROW_DOWN, ARROW_LEFT, ARROW_RIGHT = 2490368, 2621440, 2424832, 2555904


# --- 1. 參數 ---------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Lab 3：Tello EDU 巡檢")
    p.add_argument("--weights", default="models/yolo26n.pt", help="YOLO 權重")
    p.add_argument("--conf", type=float, default=0.25, help="YOLO 信心度門檻")
    p.add_argument("--out", default="runs", help="輸出根目錄")
    p.add_argument("--every", type=int, default=10,
                   help="每 N 幀推論一次。飛行時請調大或用 --preview-only")
    p.add_argument("--threshold", type=float, default=0.8, help="自動存檔門檻")
    p.add_argument("--cooldown", type=float, default=3.0, help="自動存檔冷卻秒數")
    p.add_argument("--speed", type=int, default=40,
                   help="rc 速度 0~100。第一次實飛請從 20~40 開始")
    p.add_argument("--preview-only", action="store_true",
                   help="完全不載模型、不推論，只看畫面。飛行時用這個")
    p.add_argument("--no-land-on-exit", action="store_true",
                   help="離開時不自動降落，保持懸停（你要自己接手）")
    p.add_argument("--allow-emergency", action="store_true",
                   help="開啟 X 鍵＝切斷馬達。機體會直接墜落，只在必要時用")
    return p.parse_args(argv)


# --- 2. 載入模型 -----------------------------------------------------------

def load_model(weights):
    """載入 YOLO 並跑一張假圖 warmup，回傳 (model, 耗時 ms)。

    --preview-only 不會呼叫這個函式。理由很實際：只是要看畫面的時候，不該先
    等模型載入好幾秒；而且 CPU 上推論約 0.3 fps，那個更新率沒辦法操控無人機，
    等你看到障礙物時早就撞上了。
    """
    from ultralytics import YOLO

    t0 = time.perf_counter()
    model = YOLO(weights)
    model.predict(np.zeros((640, 640, 3), np.uint8), verbose=False)
    return model, (time.perf_counter() - t0) * 1000.0


# --- 3. 影像來源 -----------------------------------------------------------

def open_tello():
    """連上 Tello 並開啟影像串流，回傳 (tello, frame_read)。"""
    from djitellopy import Tello

    tello = Tello()
    try:
        tello.connect()                      # 沒回應會丟例外
    except Exception as exc:
        raise SystemExit(
            f"連不上 Tello：{exc}\n"
            "  1. 電腦的 Wi-Fi 要連到 TELLO-XXXXXX 這個熱點\n"
            "  2. UDP 8889 沒被占用（另一個程式還開著就會占著）") from exc

    print(f"已連線　電量 {tello.get_battery()}%　溫度 {tello.get_temperature():.0f}C")
    tello.streamon()
    return tello, tello.get_frame_read()


def read_frame(frame_read):
    """讀最新一格並轉成 BGR。還沒有畫面回 None。

    兩個重點：

    * djitellopy 的 frame 是 **RGB**（內部用 frame.to_image()），而 cv2 全家
      都吃 BGR。不轉的話畫面顏色會反過來（天空變橘色），而且送進 YOLO 的張量
      也跟訓練時不一致，分數會怪。
    * frame_read.frame 永遠是「最新一格的快照」，不是佇列。所以我們不用自己
      丟舊幀，延遲也不會累積。cv2.VideoCapture 沒有這個性質，所以接 RTSP 之類
      的來源時得自己想辦法只留最新幀。
    """
    frame = frame_read.frame
    if frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def close_tello(tello, land_first):
    """收尾：rc 歸零 -> 需要就降落 -> streamoff -> end()。可以重複呼叫。"""
    try:
        tello.send_rc_control(0, 0, 0, 0)
    except Exception:
        pass
    if land_first:
        print("離開前自動降落…")
        try:
            tello.land()
        except Exception as exc:
            print(f"  降落指令被拒：{exc}（機體可能已經在地上）")
    for step in ("streamoff", "end"):
        try:
            getattr(tello, step)()
        except Exception:
            pass                             # 收尾失敗不值得再丟例外出去


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

    model = None
    if not args.preview_only:
        model, load_ms = load_model(args.weights)
        print(f"模型載入 + warmup：{load_ms:.0f} ms")
        print("提醒：辨識模式的畫面會變慢，飛行操控請用 --preview-only")

    tello, frame_read = open_tello()
    run_dir = open_run(args.out)
    tello.set_speed(10)                      # 這是 SDK 指令的預設速度，與 rc 無關

    flying = False
    lr = fb = ud = yaw = 0                   # rc 四軸速度，會一直維持到你改它
    idx = inferred = hits = snaps = 0
    last_save = last_rc = last_state = -1e9
    state_text = "狀態讀取中"
    latencies = []
    first_frame_deadline = time.perf_counter() + FIRST_FRAME_TIMEOUT_S
    seen_frame = False

    print("鍵盤：T 起飛　L 降落　空白鍵懸停　P 拍一張　Q 離開")
    try:
        while True:
            frame = read_frame(frame_read)
            if frame is None:
                # 還沒有畫面。等，但不要無限等下去。
                if time.perf_counter() > first_frame_deadline:
                    print(f"{FIRST_FRAME_TIMEOUT_S:.0f} 秒內沒有畫面，"
                          "請確認 Wi-Fi 沒斷、UDP 11111 沒被占用，再跑一次")
                    break
                time.sleep(0.05)
                continue
            seen_frame = True
            idx += 1

            run_now = (not args.preview_only) and (idx % args.every == 0)
            if run_now:
                dets, latency = detect(model, frame, args.conf)
                latencies.append(latency)
                inferred += 1
                shown = draw(frame, dets)
            else:
                dets, latency = [], 0.0
                shown = frame.copy()

            # 狀態封包每秒讀一次就夠，而且讀的是快取，不會拖慢迴圈。
            if time.perf_counter() - last_state >= STATE_INTERVAL_S:
                last_state = time.perf_counter()
                try:
                    state_text = (f"電量 {tello.get_battery()}%"
                                  f"　高度 {tello.get_height()} cm")
                except Exception:
                    state_text = "狀態讀取失敗"

            best = max((d.score for d in dets), default=0.0)
            status = (f"#{idx}  {'飛行中' if flying else '待命'}  {state_text}"
                      f"  {'辨識' if run_now else '僅顯示'} {len(dets)} 處"
                      f" 最高 {best:.2f}  命中 {hits}")

            if run_now and best > args.threshold and \
                    (time.perf_counter() - last_save) >= args.cooldown:
                save(run_dir, shown, dets, f"frame{idx:06d}", latency)
                last_save = time.perf_counter()
                hits += 1
                status += "  [已存]"

            cv2.imshow("lab3 - Tello (T/L/SPACE/P/Q)", draw_status(shown, status))
            key = cv2.waitKeyEx(1)

            # -- 飛行鍵 ------------------------------------------------------
            if key in (ord("q"), ord("Q"), 27):
                print("使用者離開")
                break

            elif key in (ord("t"), ord("T")):
                if not seen_frame:
                    # 這是規則 1。不自動代開串流，也不硬起飛。
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
                        # 被拒多半是電量或溫度，把數字一起印出來
                        print(f"起飛失敗：{exc}　{state_text}")

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
                lr = fb = ud = yaw = 0       # 懸停：四軸歸零
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

            elif key in (ord("p"), ord("P")):
                # 手動拍一張：不看門檻、不受冷卻限制。沒跑過模型的幀就單獨跑
                # 一次再存 —— 使用者要的是有標註的結果。
                if not run_now and not args.preview_only:
                    dets, latency = detect(model, frame, args.conf)
                    shown = draw(frame, dets)
                path = save(run_dir, shown, dets, f"snap{idx:06d}", latency)
                snaps += 1
                print(f"拍一張 -> {path.name}")

            # -- rc 節流 -----------------------------------------------------
            # 每 50 ms 送一次就好。迴圈本身跑得比這快，不節流會變成幾百 Hz。
            if flying and (time.perf_counter() - last_rc) >= RC_INTERVAL_S:
                last_rc = time.perf_counter()
                try:
                    tello.send_rc_control(lr, fb, ud, yaw)
                except Exception:
                    pass                     # 高頻指令偶爾掉一次不值得洗版

    except KeyboardInterrupt:
        print("\nCtrl-C 中斷")
    finally:
        # 規則 3：不管怎麼離開都收乾淨。
        cv2.destroyAllWindows()
        close_tello(tello, land_first=flying and not args.no_land_on_exit)
        if flying and args.no_land_on_exit:
            print("！機體仍在空中（--no-land-on-exit），請自行接手降落")

    avg = sum(latencies) / len(latencies) if latencies else 0.0
    print(f"\n讀 {idx} 幀，推論 {inferred} 幀，平均推論 {avg:.0f} ms")
    print(f"自動命中 {hits} 張，手動拍 {snaps} 張")
    print(f"輸出：{run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
