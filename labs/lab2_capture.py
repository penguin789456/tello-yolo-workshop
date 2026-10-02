"""Lab 2：Tello 影像與拍照 —— 看得到畫面、按 P 拍照，但**不能操控飛行**。

    python labs\\lab2_capture.py                     # 連線、開畫面、按 P 拍照
    python labs\\lab2_capture.py --source video --path data\\samples\\test.mp4

這一支刻意**沒有任何飛行指令**。按 T 不會起飛、方向鍵不會動 —— 程式會印一行
提示叫你去用 lab1。理由很簡單：這一節要練的是「影像進得來、照片存得下來」，
把飛行混進來只會讓出問題時分不清是哪一邊的錯。

無人機放在桌上開機就好，不需要飛。

和 lab1 的差別（段落編號對照著看）：

    2. 連線與收尾   多了 streamon() 與 get_frame_read()
    3. 影像來源     新增：讀最新一格、RGB 轉 BGR
    4. 畫面標註     新增：狀態列
    5. 存檔         新增：jpg + index.jsonl
    6. 主迴圈       cv2.imshow + waitKey，不送任何飛行指令

沒有 import threading。整支程式只有一條 while。
（djitellopy 內部會自己開一條解碼執行緒，那是它的事，我們不寫也不管它。）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

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

#: 狀態列更新頻率。get_battery() 讀的是狀態封包快取，不送指令、不會阻塞。
STATE_INTERVAL_S = 1.0

#: streamon 之後等第一格畫面的上限。正常 2~3 秒就會有。
FIRST_FRAME_TIMEOUT_S = 10.0

FONT = cv2.FONT_HERSHEY_SIMPLEX


# --- 1. 參數 ---------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Lab 2：Tello 影像與拍照")
    p.add_argument("--source", default="tello", choices=("tello", "video"),
                   help="影像來源。video 是沒有無人機時的練習用來源")
    p.add_argument("--path", help="--source video 時的影片檔或鏡頭編號")
    p.add_argument("--out", default="runs", help="輸出根目錄")
    p.add_argument("--headless", action="store_true",
                   help="不開視窗（自動測試用；這樣就按不到 P 了）")
    p.add_argument("--max-frames", type=int, default=0,
                   help="讀滿幾幀就停（0 = 不限）。無人值守測試用")
    args = p.parse_args(argv)
    if args.source == "video" and not args.path:
        p.error("--source video 需要 --path")
    return args


# --- 2. 連線與收尾 ---------------------------------------------------------

def open_source(args):
    """開啟影像來源，回傳 (kind, handle, frame_read)。

    tello：connect() -> streamon() -> get_frame_read()
    video：cv2.VideoCapture，給沒有無人機的人練習用
    """
    if args.source == "video":
        target = int(args.path) if str(args.path).isdigit() else str(args.path)
        if isinstance(target, str) and not Path(target).exists():
            raise SystemExit(f"找不到影片檔：{target}")
        cap = cv2.VideoCapture(target)
        if not cap.isOpened():
            raise SystemExit(f"無法開啟來源：{target}")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 只留最新一格，不讓 buffer 堆舊資料
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
    return "tello", tello, tello.get_frame_read()


def close_source(kind, handle):
    """收尾。這一支不會飛，所以不需要降落 —— 但串流一定要關。

    串流不關的話飛機會一直往 UDP 11111 推封包，下次連線時解碼器接在一段殘缺
    的 H.264 中間，畫面容易卡住不動。
    """
    if kind == "video":
        handle.release()
        return
    for step in ("streamoff", "end"):
        try:
            getattr(handle, step)()
        except Exception:
            pass


# --- 3. 影像來源 -----------------------------------------------------------

def read_frame(kind, handle, frame_read):
    """讀最新一格 BGR 影像。來源結束回 None，還沒準備好回字串 "wait"。

    Tello 的兩個重點：

    * djitellopy 的 frame 是 **RGB**（內部用 frame.to_image()），而 cv2 全家
      都吃 BGR。不轉的話畫面顏色會反過來（天空變橘色）。
    * frame_read.frame 永遠是「最新一格的快照」，不是佇列。讀它永遠拿到最新
      的，所以延遲不會累積。
    """
    if kind == "video":
        ok, frame = handle.read()
        return frame if ok else None

    frame = frame_read.frame
    if frame is None:
        return "wait"
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


# --- 4. 畫面標註 -----------------------------------------------------------

def draw_status(frame, text):
    """把狀態列畫在畫面下緣（就地修改，這張影像只是要顯示而已）。"""
    h = frame.shape[0]
    cv2.rectangle(frame, (0, h - 24), (frame.shape[1], h), (0, 0, 0), -1)
    cv2.putText(frame, text, (6, h - 7), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


# --- 5. 存檔 ---------------------------------------------------------------

def open_run(out_root):
    """每次執行開一個 runs 底下的時間戳資料夾。"""
    run_dir = Path(out_root) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save(run_dir, image, label):
    """存一張照片，並在 index.jsonl 追加一行紀錄。回傳檔案路徑。

    這一支還沒有辨識結果，所以 count / classes 先留空 —— 但欄位的形狀和 lab3
    一樣，這樣 lab3 加上辨識之後，你回頭看 index.jsonl 就認得出差在哪。

    索引用 jsonl（一行一筆）而不是一整包 json：程式中途被 Ctrl-C，已經寫進去
    的行仍然是合法內容，不會整個檔案壞掉。
    """
    safe = "".join(c for c in label if c.isalnum() or c in "-_")[:40]
    name = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + safe + ".jpg"
    cv2.imwrite(str(run_dir / name), image)

    record = {
        "file": name,
        "label": label,
        "time": datetime.now().isoformat(timespec="seconds"),
        "count": 0,
        "best_score": 0.0,
        "classes": [],
        "latency_ms": 0.0,
    }
    with (run_dir / "index.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return run_dir / name


# --- 6. 主迴圈 -------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    kind, handle, frame_read = open_source(args)
    run_dir = open_run(args.out)

    print(f"來源 {kind}　輸出 {run_dir}")
    if not args.headless:
        print("鍵盤：P 拍一張　Q 離開　（這一節不做飛行操控）")

    idx = snaps = 0
    state_text = ""
    last_state = -1e9
    deadline = time.perf_counter() + FIRST_FRAME_TIMEOUT_S

    try:
        while True:
            if args.max_frames and idx >= args.max_frames:
                print(f"已達 --max-frames {args.max_frames}")
                break

            frame = read_frame(kind, handle, frame_read)
            if isinstance(frame, str):       # "wait"：Tello 還沒吐出第一格
                if time.perf_counter() > deadline:
                    print(f"{FIRST_FRAME_TIMEOUT_S:.0f} 秒內沒有畫面，"
                          "請確認 Wi-Fi 沒斷、UDP 11111 沒被占用，再跑一次")
                    break
                time.sleep(0.05)
                continue
            if frame is None:
                print("來源結束")
                break
            idx += 1

            if kind == "tello" and \
                    time.perf_counter() - last_state >= STATE_INTERVAL_S:
                last_state = time.perf_counter()
                try:
                    state_text = (f"　電量 {handle.get_battery()}%"
                                  f" 高度 {handle.get_height()} cm")
                except Exception:
                    state_text = "　狀態讀取失敗"

            status = f"#{idx}{state_text}　已拍 {snaps} 張"

            if args.headless:
                continue

            cv2.imshow("lab2 - P=拍照  Q=離開", draw_status(frame.copy(), status))
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                print("使用者離開")
                break
            if key == ord("p"):
                path = save(run_dir, frame, f"snap{idx:06d}")
                snaps += 1
                print(f"拍一張 -> {path.name}")
            elif key in (ord("t"), ord("l")):
                # 這一支不做飛行。明確告訴使用者去哪裡做，而不是靜靜地忽略。
                print("這一節不做飛行操控。要起飛請用："
                      "python labs\\lab1_flight.py --path")

    except KeyboardInterrupt:
        print("\nCtrl-C 中斷")
    finally:
        if not args.headless:
            cv2.destroyAllWindows()
        close_source(kind, handle)

    print(f"\n讀 {idx} 幀，拍了 {snaps} 張")
    print(f"輸出：{run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
