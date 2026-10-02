"""Lab 1：Tello 操控與路徑控制 —— 起飛、前進 100 公分、降落。

    python labs\\lab1_flight.py                      # 只連線看狀態，不起飛
    python labs\\lab1_flight.py --path               # 起飛 -> 前進 100 cm -> 降落
    python labs\\lab1_flight.py --path --distance 150
    python labs\\lab1_flight.py --manual             # 用鍵盤即時操控

這一支完全不碰影像，所以**只需要 djitellopy**，不需要 opencv、不需要 YOLO、
不需要 PyTorch。第一節課先把「電腦指揮得動無人機」這件事做通就好。

兩種控制方式，差別很重要：

    路徑控制 (--path)    送一次 move_forward(100)，飛機自己飛完 1 公尺才回應。
                         指令是「阻塞」的：程式會等它飛完。適合可重複的動作。

    手動操控 (--manual)  每 50 ms 送一次 send_rc_control(四軸速度)。
                         指令是「連續」的：你不送，它就照最後一筆速度繼續飄。

**安全規則：預設不起飛。** 不加 --path 或 --manual 的話，程式只連線、印電量與
高度然後離開。起飛是不可逆的實體動作，不該是「跑一下看看」的副作用。

段落順序：
    1. 參數   2. 連線與收尾   3. 路徑控制   4. 手動操控   5. 主迴圈

沒有 import threading。
"""
from __future__ import annotations

import argparse
import sys
import time

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
#: 這一支其實不需要 torch，但三支範例共用同一個環境，門檻就統一。
if not ((3, 10) <= sys.version_info[:2] <= (3, 13)):
    raise SystemExit(
        f"需要 Python 3.10 ~ 3.13，目前是 {sys.version.split()[0]}。\n"
        "  建議用 3.12（教材實測的版本）。Windows 上可以用啟動器指定：\n"
        "      py -3.12 -m venv .venv")

#: Tello SDK 建議 rc 指令不要超過 20 Hz。灌太快會塞爆指令通道，反而讓起飛／
#: 降落這類需要回應的指令被拖延。實測灌到 125 Hz 就會出現這個現象。
RC_INTERVAL_S = 0.05

#: 狀態列更新頻率。get_battery() 讀的是狀態封包快取，不送指令、不會阻塞。
STATE_INTERVAL_S = 1.0

#: 低於這個電量就不讓它起飛。Tello 自己大約在 10~20% 會拒絕指令，但電量低時
#: 飛行也不穩，不如早點擋下來並說清楚原因。
MIN_BATTERY_PCT = 30

#: move_forward() 的合法範圍是 20~500 cm（SDK 限制），超出會被飛機拒絕。
MIN_DISTANCE_CM, MAX_DISTANCE_CM = 20, 500


# --- 1. 參數 ---------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Lab 1：Tello 操控與路徑控制")
    p.add_argument("--path", action="store_true",
                   help="跑預設路徑：起飛 -> 前進 --distance 公分 -> 降落")
    p.add_argument("--distance", type=int, default=100,
                   help=f"--path 前進幾公分（{MIN_DISTANCE_CM}~{MAX_DISTANCE_CM}）")
    p.add_argument("--manual", action="store_true",
                   help="用鍵盤即時操控（Windows 專用，見檔頭說明）")
    p.add_argument("--speed", type=int, default=40,
                   help="手動操控的 rc 速度 0~100。第一次實飛請用 20")
    p.add_argument("--allow-emergency", action="store_true",
                   help="手動模式下開啟 X 鍵＝切斷馬達。機體會直接墜落")
    args = p.parse_args(argv)
    if args.path and args.manual:
        p.error("--path 和 --manual 只能選一個")
    if not (MIN_DISTANCE_CM <= args.distance <= MAX_DISTANCE_CM):
        p.error(f"--distance 必須在 {MIN_DISTANCE_CM}~{MAX_DISTANCE_CM} 之間"
                f"（Tello SDK 的限制），你給的是 {args.distance}")
    return args


# --- 2. 連線與收尾 ---------------------------------------------------------

def open_tello():
    """連上 Tello，回傳 tello 物件。失敗時印出可行動的檢查清單。"""
    from djitellopy import Tello

    tello = Tello()
    try:
        tello.connect()                      # 沒回應會丟例外
    except Exception as exc:
        raise SystemExit(
            f"連不上 Tello：{exc}\n"
            "  1. 電腦的 Wi-Fi 要連到 TELLO-XXXXXX 這個熱點\n"
            "  2. UDP 8889 沒被占用（另一個程式還開著就會占著）") from exc
    return tello


def show_state(tello):
    """印一行狀態。讀的是狀態封包快取，不送指令、不會阻塞。"""
    try:
        print(f"電量 {tello.get_battery()}%　高度 {tello.get_height()} cm"
              f"　溫度 {tello.get_temperature():.0f}C")
    except Exception:
        print("狀態讀取失敗")


def check_battery(tello):
    """電量不足就不要飛。回傳 True 表示可以飛。"""
    try:
        pct = tello.get_battery()
    except Exception:
        print("讀不到電量，為安全起見不起飛")
        return False
    if pct < MIN_BATTERY_PCT:
        print(f"電量只有 {pct}%，低於 {MIN_BATTERY_PCT}% 不起飛。請換電池。")
        return False
    return True


def close_tello(tello, land_first):
    """收尾：rc 歸零 -> 需要就降落 -> end()。可以重複呼叫。

    不管程式怎麼結束（正常跑完、Ctrl-C、例外），都要走到這裡。CLI 程式一結束
    就沒人控制飛機了，所以預設會降落。
    """
    try:
        tello.send_rc_control(0, 0, 0, 0)
    except Exception:
        pass
    if land_first:
        print("收尾：降落…")
        try:
            tello.land()
        except Exception as exc:
            print(f"  降落指令被拒：{exc}（機體可能已經在地上）")
    try:
        tello.end()
    except Exception:
        pass                                 # 收尾失敗不值得再丟例外出去


# --- 3. 路徑控制 -----------------------------------------------------------

def fly_path(tello, distance_cm):
    """起飛 -> 前進 distance_cm -> 降落。回傳是否全部完成。

    這三個指令都是**阻塞**的：`move_forward(100)` 送出去之後，djitellopy 會等
    飛機回 "ok" 才返回，也就是等它真的飛完 1 公尺。所以這個函式跑完就是動作
    做完，不需要自己算時間、也不需要 sleep。

    反過來說，飛行途中程式什麼都不能做 —— 要邊飛邊看畫面就得用手動模式
    （送 rc 速度，不阻塞），那是 lab3 的做法。
    """
    print(f"路徑：起飛 -> 前進 {distance_cm} cm -> 降落")
    try:
        print("  起飛…")
        tello.takeoff()
    except Exception as exc:
        print(f"  起飛失敗：{exc}")
        show_state(tello)                    # 多半是電量或溫度，把數字印出來
        return False

    try:
        print(f"  前進 {distance_cm} cm…（指令會等它飛完才返回）")
        tello.move_forward(distance_cm)
        print("  前進完成")
    except Exception as exc:
        # 動作失敗不代表可以不管它 —— 還在空中，一定要降落
        print(f"  前進失敗：{exc}，直接降落")
        return False
    return True


# --- 4. 手動操控 -----------------------------------------------------------

#: 鍵盤 -> (四軸欄位, 倍率)。四軸 = (左右, 前後, 上下, 旋轉)
KEY_AXIS = {
    b"8": ("for_back", 1), b"2": ("for_back", -1),
    b"4": ("left_right", -1), b"6": ("left_right", 1),
    b"w": ("up_down", 1), b"s": ("up_down", -1),
    b"a": ("yaw", -1), b"d": ("yaw", 1),
}

#: msvcrt 的方向鍵是兩個位元組：先 \xe0，再跟著這些。
ARROW_AXIS = {
    b"H": ("for_back", 1), b"P": ("for_back", -1),
    b"K": ("left_right", -1), b"M": ("left_right", 1),
}

MANUAL_HELP = """鍵盤（按一下會持續移動，放開不會停）：
    ↑ ↓ ← →  前進 / 後退 / 左移 / 右移      8 2 4 6  同上
    W / S     上升 / 下降                    A / D    左轉 / 右轉
    空白鍵    四軸歸零（懸停）                T / L    起飛 / 降落
    Q         離開（會先降落）"""


def read_key():
    """非阻塞讀一個按鍵。沒人按就回 None。

    這一支沒有影像視窗，所以不能用 cv2.waitKey 收鍵盤，改用 msvcrt：
    kbhit() 先問「有沒有人按」，有才 getch() 把它讀走 —— 這樣主迴圈不會被
    卡住，才能繼續每 50 ms 送一次 rc 指令。

    **msvcrt 是 Windows 專用。** 非 Windows 請改用 lab2 / lab3（它們有影像
    視窗，用 cv2.waitKeyEx 收鍵盤，跨平台）。
    """
    import msvcrt

    if not msvcrt.kbhit():
        return None
    ch = msvcrt.getch()
    if ch in (b"\x00", b"\xe0"):             # 方向鍵的前導位元組
        return ("arrow", msvcrt.getch())
    return ("key", ch.lower())


def fly_manual(tello, speed, allow_emergency):
    """鍵盤即時操控。回傳離開時是否還在飛。

    和路徑控制最大的差別：這裡的指令**不阻塞**。`send_rc_control` 送出去就
    返回，飛機照著那組速度一直飛，直到你送新的一組。所以主迴圈必須持續送，
    而且要節流到 20 Hz 以內。
    """
    print(MANUAL_HELP)
    flying = False
    lr = fb = ud = yaw = 0                   # 四軸速度，會一直維持到你改它
    last_rc = last_state = -1e9

    while True:
        got = read_key()
        if got is not None:
            kind, ch = got
            if kind == "arrow" and ch in ARROW_AXIS:
                field, sign = ARROW_AXIS[ch]
                lr, fb, ud, yaw = _set_axis(lr, fb, ud, yaw, field, sign * speed)
            elif kind == "key":
                if ch in (b"q", b"\x1b"):
                    print("使用者離開")
                    break
                elif ch == b"t":
                    if flying:
                        print("已在飛行中，忽略起飛")
                    elif check_battery(tello):
                        try:
                            tello.takeoff()
                            flying = True
                            lr = fb = ud = yaw = 0
                            print("起飛")
                        except Exception as exc:
                            print(f"起飛失敗：{exc}")
                            show_state(tello)
                elif ch == b"l":
                    lr = fb = ud = yaw = 0
                    try:
                        tello.send_rc_control(0, 0, 0, 0)
                        tello.land()
                        flying = False
                        print("降落")
                    except Exception as exc:
                        print(f"降落指令被拒：{exc}（再按一次 L）")
                elif ch == b" ":
                    lr = fb = ud = yaw = 0
                    print("懸停")
                elif ch == b"x" and allow_emergency:
                    print("緊急停止：切斷馬達")
                    try:
                        tello.emergency()
                    except Exception as exc:
                        print(f"緊急停止失敗：{exc}")
                    flying = False
                elif ch in KEY_AXIS:
                    field, sign = KEY_AXIS[ch]
                    lr, fb, ud, yaw = _set_axis(lr, fb, ud, yaw, field,
                                                sign * speed)

        # rc 節流：每 50 ms 一次就好。這個迴圈本身跑得比 20 Hz 快得多。
        now = time.perf_counter()
        if flying and (now - last_rc) >= RC_INTERVAL_S:
            last_rc = now
            try:
                tello.send_rc_control(lr, fb, ud, yaw)
            except Exception:
                pass                         # 高頻指令偶爾掉一次不值得洗版

        if (now - last_state) >= STATE_INTERVAL_S:
            last_state = now
            print(f"  {'飛行中' if flying else '待命'}"
                  f"  速度 lr={lr} fb={fb} ud={ud} yaw={yaw}", end="\r")

        time.sleep(0.005)                    # 不睡會把一顆核心燒滿
    return flying


def _set_axis(lr, fb, ud, yaw, field, value):
    """把指定的那一軸換成 value，其餘不動。回傳新的四軸。"""
    axes = {"left_right": lr, "for_back": fb, "up_down": ud, "yaw": yaw}
    axes[field] = value
    return axes["left_right"], axes["for_back"], axes["up_down"], axes["yaw"]


# --- 5. 主迴圈 -------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    tello = open_tello()
    print("已連線")
    show_state(tello)

    flying = False
    try:
        if args.path:
            if not check_battery(tello):
                return 1
            flying = fly_path(tello, args.distance)
        elif args.manual:
            tello.set_speed(10)              # SDK 指令的預設速度，與 rc 無關
            flying = fly_manual(tello, args.speed, args.allow_emergency)
        else:
            # 預設什麼都不做。起飛是不可逆的實體動作，不該是跑程式的副作用。
            print("\n沒有加 --path 或 --manual，所以不起飛。")
            print("  --path    起飛 -> 前進 100 cm -> 降落")
            print("  --manual  用鍵盤操控")
    except KeyboardInterrupt:
        print("\nCtrl-C 中斷")
    finally:
        # 不管怎麼離開都收乾淨：還在飛就降落，然後斷線。
        close_tello(tello, land_first=flying)

    show_state(tello)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
