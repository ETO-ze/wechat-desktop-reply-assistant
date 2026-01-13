\
# wechat_auto_bot_single_v6_4.py
# 阶段三-3（下一步）：更可靠的新消息触发 + 自动按“当前聊天标题”隔离会话
#
# 你当前已完成：
# - Weixin.exe 主窗口选择稳定
# - 发送链路稳定（点击聚焦输入框→粘贴→校验→回车）
# - Tesseract OCR 可用，消息区域校准 OK
#
# v6_4 新增能力（重点）
# 1) 自动会话路由：
#    - 通过 OCR 识别“当前聊天标题”（顶部聊天对象名称），自动将 person_key 切换为该标题
#    - 不再强依赖 Ctrl+Alt+1..0 手动切槽（仍保留作备用）
# 2) 只对“对方消息”触发：
#    - 将消息区域拆分成“左侧（对方气泡）/右侧（自己气泡）”两块
#    - 默认只用左侧 OCR 的尾部变化作为“新消息”触发源，降低把自己消息当新消息的概率
# 3) 更强防自触发：
#    - 发送后短时间抑制 OCR 触发
#    - 与最近一次发送内容高度相似则忽略
#
# 默认安全策略保持不变：
# - AutoReply 默认 OFF
# - 必须 Arm
# - 默认 confirm_before_send=true：生成回复后需 Ctrl+Alt+Y 确认发送
#
# 依赖安装：
#   python -m pip install psutil keyboard pyperclip pywinauto mss pillow pytesseract openai rapidfuzz
#
# 配置文件：wechat_auto_bot_config.json（首次运行自动生成）
# 你需要新增/校准两个区域：
# - regions.title：当前聊天标题区域（截图后 OCR 得到聊天对象名称）
# - regions.message：消息区域（你已校准）
#
# 快捷键：
#   Ctrl+Alt+S : Arm 开关（默认 10 分钟）
#   Ctrl+Alt+A : AutoReply 开关（必须已 Arm）
#   Ctrl+Alt+G : 手动发送剪贴板（必须已 Arm）
#   Ctrl+Alt+Y : 确认发送待发送回复（confirm_before_send=true 时）
#   Ctrl+Alt+C : 保存校准截图（整窗 + 标题区 + 消息区 + 左右拆分区）到 ./logs
#   Ctrl+Alt+R : 重置当前会话（清空历史 + 去重状态）
#   Ctrl+Alt+Q : 紧急退出
#   Ctrl+Alt+1..0 : 切换备用会话槽（当 title OCR 不稳定时可用）
#
# 输出日志说明：
# - [CHAT] 当前会话 key（来自 title OCR 或 slot）
# - [IN]   识别到“左侧新消息”
# - [LLM]  生成回复
# - [AUTO] 是否需要确认发送
# - [SEND] 实际发送结果

from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
import re
import time
import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import shutil


def load_local_dotenv(dotenv_path: Path):
    """
    读取同目录 .env 文件（可选），用于在不改系统环境变量的情况下提供 API KEY。
    文件格式示例（任意一种即可）：
        GEMINI_API_KEY=AIzaSy...
        OPENAI_API_KEY=sk-...
    说明：
      - 若系统环境变量已存在对应 KEY，则不会覆盖。
      - .env 文件仅建议存放在本机，不要提交到 GitHub。
    """
    try:
        if not dotenv_path.exists():
            return
        for raw in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if not k or not v:
                continue
            if k in ("OPENAI_API_KEY", "GEMINI_API_KEY") and not os.environ.get(k):
                os.environ[k] = v
    except Exception:
        return


import psutil
import keyboard
import pyperclip
import mss
from PIL import Image, ImageOps, ImageEnhance

from pywinauto import Application
from rapidfuzz import fuzz

# ---------- 路径 ----------

APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "wechat_auto_bot_config.json"
SESSIONS_PATH = APP_DIR / "sessions.json"
STATE_PATH = APP_DIR / "state.json"
LOG_DIR = APP_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 默认配置 ----------
# 重点：regions.title 需要你按 Ctrl+Alt+C 截图后校准一次。

DEFAULT_CONFIG = {
    "preferred_proc_names": ["weixin.exe"],
    "soft_proc_hints": ["wechat", "weixin"],
    "title_hints": ["微信", "WeChat"],
    "class_hints": ["Qt", "WeChatMainWndForPC", "WX"],

    "tesseract_cmd": r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    "tesseract_lang": "chi_sim+eng",

    "poll_interval_sec": 1.0,

    "regions": {
        # 聊天标题区域（用于会话路由）：需要你校准
        "title":   {"x1": 0.28, "y1": 0.02, "x2": 0.78, "y2": 0.11},
        # 消息区域：你已校准过，可保持/微调
        "message": {"x1": 0.28, "y1": 0.10, "x2": 0.98, "y2": 0.74},
    },

    # 消息区域左右拆分：用于“只触发左侧（对方消息）”
    # split_left_ratio：左侧区域宽度占消息区域宽度的比例（建议 0.55~0.70）
    # overlap_ratio：左右区域中间重叠比例（防止气泡靠中被切断，建议 0.10~0.20）
    "incoming_detection": {
        "mode": "left_only",          # left_only / full
        "split_left_ratio": 0.62,
        "overlap_ratio": 0.15
    },

    # 视觉差分门控：先判断画面是否变化，再决定是否 OCR（更省 CPU、更少误触发）
    "diff_gate": {
        "enabled": True,
        "hash_size": 16,             # 感知哈希尺寸（16 => 256 bit）
        "hamming_thresh": 10,        # 哈希海明距离阈值：越小越“严格认为没变化”
        "min_interval_sec": 0.4      # 两次差分判断最小间隔（避免过度频繁）
    },

    "click_points": [
        [0.50, 0.88],
        [0.50, 0.82],
        [0.35, 0.88],
        [0.65, 0.88],
    ],

    "arm_minutes": 10,
    "require_foreground_wechat": True,
    "confirm_before_send": True,
    "confirm_timeout_sec": 30,

    "auto_reply": {
        "enabled_default": False,
        "cooldown_sec": 6,
        "max_replies_per_hour": 20,
        "max_session_messages": 30
    },

    # 会话路由模式：
    # - title_ocr：默认，自动识别聊天标题作为 person_key（推荐）
    # - slot：只使用 slot1..slot10 手动切换
    "session_routing": {
        "mode": "title_ocr",
        "min_title_len": 1,                 # 标题长度过短则认为无效
        "title_update_cooldown": 0.8,       # 识别节流：避免频繁 OCR 标题
        "stability_required": 2,            # 需要连续 N 次识别到同一标题才切换（抗抖动）
        "merge_similar_threshold": 92       # 标题相似度>=该阈值则视为同一联系人（合并 key）
    },

    "openai": {
        "provider": "gemini",
        "api_key": "",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-2.5-flash-lite",
        "system_prompt": "你是一个桌面微信聊天回复助手。请用简洁、自然、符合上下文的中文回复对方。避免长篇大论；必要时先澄清再回答。",
        "temperature": 0.4
    }
}

# ---------- Win32 API ----------

user32 = ctypes.windll.user32
EnumWindows = user32.EnumWindows
EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

GetWindowTextW = user32.GetWindowTextW
GetWindowTextLengthW = user32.GetWindowTextLengthW
GetClassNameW = user32.GetClassNameW
IsWindowVisible = user32.IsWindowVisible
GetWindowRect = user32.GetWindowRect
GetWindowThreadProcessId = user32.GetWindowThreadProcessId
SetForegroundWindow = user32.SetForegroundWindow
ShowWindow = user32.ShowWindow
GetForegroundWindow = user32.GetForegroundWindow

SW_RESTORE = 9

class RECT(ctypes.Structure):
    _fields_ = [
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    ]

def _get_text(hwnd: int) -> str:
    length = GetWindowTextLengthW(hwnd)
    if length == 0:
        return ""
    buf = ctypes.create_unicode_buffer(length + 1)
    GetWindowTextW(hwnd, buf, length + 1)
    return buf.value

def _get_class(hwnd: int) -> str:
    buf = ctypes.create_unicode_buffer(256)
    GetClassNameW(hwnd, buf, 256)
    return buf.value

def _get_rect(hwnd: int) -> Optional[Tuple[int, int, int, int]]:
    r = RECT()
    if not GetWindowRect(hwnd, ctypes.byref(r)):
        return None
    return (r.left, r.top, r.right, r.bottom)

def _rect_area(rect) -> int:
    if not rect:
        return 0
    l, t, r, b = rect
    return max(0, r - l) * max(0, b - t)

def _pid_of(hwnd: int) -> int:
    pid = wintypes.DWORD()
    GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return int(pid.value)

def _proc_name(pid: int) -> str:
    try:
        return (psutil.Process(pid).name() or "")
    except Exception:
        return ""

def _any_hint_in(s: str, hints: List[str]) -> bool:
    s = (s or "").lower()
    return any(h.lower() in s for h in hints)

def _focus_hwnd(hwnd: int):
    try:
        ShowWindow(hwnd, SW_RESTORE)
    except Exception:
        pass
    try:
        SetForegroundWindow(hwnd)
    except Exception:
        pass

def _is_foreground(hwnd: int) -> bool:
    try:
        fg = int(GetForegroundWindow())
        return fg == int(hwnd)
    except Exception:
        return False

# ---------- OCR：Tesseract ----------

def resolve_tesseract_cmd(cfg_cmd: str) -> str:
    """解析 tesseract 可执行文件路径（支持 PATH / 常见安装路径 / 绝对路径）"""
    if cfg_cmd:
        p = Path(cfg_cmd)
        if p.exists():
            return str(p)
        found = shutil.which(cfg_cmd)
        if found:
            return found

    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        str(Path.home() / r"AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return cfg_cmd or "tesseract"

def preprocess_for_ocr(img: Image.Image) -> Image.Image:
    """OCR 前处理：灰度 + 对比度 + 锐化"""
    g = ImageOps.grayscale(img)
    g = ImageEnhance.Contrast(g).enhance(2.2)
    g = ImageEnhance.Sharpness(g).enhance(1.6)
    return g


def phash_ahash(img: Image.Image, hash_size: int = 16) -> int:
    """
    计算简易感知哈希（aHash）：
    - 转灰度
    - 缩放到 hash_size x hash_size
    - 与平均灰度比较得到 bit 矩阵
    返回：一个整数，包含 hash_size^2 个 bit。
    """
    g = ImageOps.grayscale(img).resize((hash_size, hash_size))
    pixels = list(g.getdata())
    avg = sum(pixels) / max(1, len(pixels))
    bits = 0
    for p in pixels:
        bits = (bits << 1) | (1 if p > avg else 0)
    return bits

def hamming_distance(a: int, b: int) -> int:
    """计算两个整数 bit 串的海明距离"""
    return (a ^ b).bit_count()

def ocr_lines_tesseract(img: Image.Image, tesseract_cmd: str, lang: str) -> List[str]:
    """对图像做 OCR，返回非空行列表"""
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = resolve_tesseract_cmd(tesseract_cmd)
    img = preprocess_for_ocr(img)
    text = pytesseract.image_to_string(img, lang=lang)
    return [l.strip() for l in text.splitlines() if l.strip()]

def normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def signature(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8", errors="ignore")).hexdigest()

def filter_noise_lines(lines: List[str]) -> List[str]:
    """过滤常见噪声行"""
    out = []
    for l in lines:
        l = l.strip()
        if not l:
            continue
        # 常见时间戳/日期
        if re.fullmatch(r"\d{1,2}:\d{2}", l):
            continue
        if re.fullmatch(r"\d{4}[/-]\d{1,2}[/-]\d{1,2}.*", l):
            continue
        # 常见 UI 词
        if l in ("微信", "发送", "Send", "搜索", "Search"):
            continue
        out.append(l)
    return out

def tail_delta(prev_lines: List[str], curr_lines: List[str]) -> str:
    """提取尾部变化片段作为新消息候选"""
    prev = prev_lines[-8:]
    curr = curr_lines[-8:]
    if not curr:
        return ""
    if not prev:
        return "\n".join(curr[-2:])[:400]

    i = 1
    while i <= min(len(prev), len(curr)):
        if normalize_text(prev[-i]) != normalize_text(curr[-i]):
            break
        i += 1
    start = max(0, len(curr) - i + 1)
    tail = curr[start:] or [curr[-1]]
    return "\n".join(tail)[:400]

def sanitize_key(name: str) -> str:
    """把聊天标题转为可用的会话 key（去除奇怪字符/收敛空白）"""
    name = normalize_text(name)
    # 去掉一些 OCR 常见杂质符号
    name = re.sub(r"[^\w\u4e00-\u9fff·\-\s]", "", name).strip()
    name = re.sub(r"\s+", " ", name)
    # 控制长度，防止异常长 key
    if len(name) > 40:
        name = name[:40]
    return name or ""

# ---------- JSON 持久化 ----------

def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default

def save_json(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

# ---------- OpenAI 调用 ----------

def openai_generate_reply(model: str,
                          system_prompt: str,
                          temperature: float,
                          history: List[Dict],
                          user_text: str,
                          api_key: str = "",
                          base_url: str = "") -> str:
    """
    统一用 Chat Completions 调用（兼容 Gemini OpenAI-compatible API）。

    Key 来源优先级：
      1) 参数 api_key（来自 config）
      2) 环境变量 OPENAI_API_KEY
      3) 环境变量 GEMINI_API_KEY
      4) （可选）同目录 .env 中的 OPENAI_API_KEY / GEMINI_API_KEY（由 load_local_dotenv 注入）
    """
    from openai import OpenAI

    k = (api_key or "").strip()         or (os.environ.get("OPENAI_API_KEY") or "").strip()         or (os.environ.get("GEMINI_API_KEY") or "").strip()

    if not k:
        raise RuntimeError("未检测到 API KEY：请设置 OPENAI_API_KEY 或 GEMINI_API_KEY，或在 config 的 openai.api_key 中填写。")

    bu = (base_url or "").strip()
    client = OpenAI(api_key=k, base_url=bu) if bu else OpenAI(api_key=k)

    messages: List[Dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )

    content = ""
    try:
        content = resp.choices[0].message.content
    except Exception:
        content = ""
    return (content or "").strip()

# ---------- 核心自动化 ----------

@dataclass
class SelectedWindow:
    hwnd: int
    pid: int
    proc: str
    title: str
    cls: str
    rect: Tuple[int, int, int, int]

class WeChatAutoBot:
    def __init__(self, cfg: Dict):
        self.cfg = cfg

        self.sessions: Dict[str, List[Dict]] = load_json(SESSIONS_PATH, {})
        self.state: Dict = load_json(
            STATE_PATH,
            {"last_sig": {}, "rate": {}, "last_text": {}, "last_send_ts": {}, "last_sent_text": {}}
        )

        # 备用会话槽（当 title OCR 不稳定时用）
        self.slot_keys = [f"slot{i}" for i in range(1, 11)]
        self.person_key = self.slot_keys[0]

        self._last_title_key = ""
        self._last_title_update_ts = 0.0

        self.armed = False
        self.arm_until = 0.0
        self.auto_reply_enabled = bool(cfg["auto_reply"]["enabled_default"])

        self.pending_reply: Optional[str] = None
        self.pending_deadline: float = 0.0

        self.win: Optional[SelectedWindow] = None
        self.app = None
        self.uia_win = None

        # OCR 记忆（用于尾部变化判断）
        self.prev_in_lines: List[str] = []   # 左侧（对方）
        self.prev_full_lines: List[str] = [] # 全量（备用）

        # tesseract 是否可用（避免刷屏）
        self._tess_ok: Optional[bool] = None

        # 发送后短时间抑制 OCR 触发（避免自回复死循环）
        self._suppress_until: float = 0.0

        # 视觉差分门控状态：上一帧哈希与时间
        self._last_diff_ts: float = 0.0
        self._prev_left_hash: Optional[int] = None
        self._prev_full_hash: Optional[int] = None

        # 标题稳定化：连续识别计数
        self._title_candidate: str = ""
        self._title_candidate_count: int = 0

    # ---- 窗口选择 ----

    def enumerate_windows(self) -> List[Dict]:
        results = []
        @EnumWindowsProc
        def callback(hwnd, lparam):
            try:
                hwnd_i = int(hwnd)
                rect = _get_rect(hwnd_i)
                area = _rect_area(rect)
                if area <= 0:
                    return True

                l, t, r, b = rect
                if (r - l) < 300 or (b - t) < 300:
                    return True

                title = _get_text(hwnd_i).strip()
                cls = _get_class(hwnd_i).strip()
                visible = bool(IsWindowVisible(hwnd_i))
                pid = _pid_of(hwnd_i)
                pname = _proc_name(pid).strip()

                results.append({
                    "hwnd": hwnd_i, "rect": rect, "area": area,
                    "title": title, "class": cls, "visible": visible,
                    "pid": pid, "proc": pname,
                })
            except Exception:
                pass
            return True
        EnumWindows(callback, 0)
        return results

    def score(self, w: Dict) -> int:
        if not w.get("visible", False):
            return -1

        score = int(w["area"]) + 5_000_000
        pname = (w.get("proc") or "").lower()
        title = w.get("title") or ""
        cls = w.get("class") or ""

        preferred = [x.lower() for x in self.cfg["preferred_proc_names"]]
        if pname in preferred:
            score += 20_000_000
        if _any_hint_in(pname, self.cfg["soft_proc_hints"]):
            score += 6_000_000
        if _any_hint_in(title, self.cfg["title_hints"]):
            score += 10_000_000
        if _any_hint_in(cls, self.cfg["class_hints"]):
            score += 8_000_000
        if not title:
            score -= 200_000
        return score

    def pick_window(self) -> SelectedWindow:
        windows = self.enumerate_windows()
        if not windows:
            raise RuntimeError("未找到任何顶层窗口。")
        windows.sort(key=self.score, reverse=True)
        best = windows[0]
        if self.score(best) < 0:
            raise RuntimeError("未找到可见微信候选窗口：请保持微信主窗口可见后重试。")
        return SelectedWindow(
            hwnd=best["hwnd"], pid=best["pid"], proc=best["proc"],
            title=best["title"], cls=best["class"], rect=best["rect"]
        )

    def connect(self):
        self.win = self.pick_window()
        try:
            self.app = Application(backend="uia").connect(handle=self.win.hwnd)
        except Exception:
            self.app = Application(backend="uia").connect(process=self.win.pid)
        self.uia_win = self.app.window(handle=self.win.hwnd)

        try:
            self.uia_win.set_focus()
        except Exception:
            _focus_hwnd(self.win.hwnd)

        print(f"[SELECTED] pid={self.win.pid} proc={self.win.proc} hwnd=0x{self.win.hwnd:X} rect={self.win.rect} class={self.win.cls!r} title={self.win.title!r}")
        print(f"[CHAT] 初始会话 key = {self.person_key}")

    # ---- 区域计算 / 截图 ----

    def _region_from_ratio(self, name: str) -> Tuple[int, int, int, int]:
        assert self.win is not None
        l, t, r, b = self.win.rect
        W = r - l
        H = b - t
        rr = self.cfg["regions"][name]
        x1 = int(l + W * rr["x1"]); y1 = int(t + H * rr["y1"])
        x2 = int(l + W * rr["x2"]); y2 = int(t + H * rr["y2"])
        x1 = max(l, min(x1, r - 1))
        x2 = max(l + 1, min(x2, r))
        y1 = max(t, min(y1, b - 1))
        y2 = max(t + 1, min(y2, b))
        return (x1, y1, x2, y2)

    def _split_message_rects(self, msg_rect: Tuple[int, int, int, int]) -> Tuple[Tuple[int,int,int,int], Tuple[int,int,int,int]]:
        """把 message 区域拆成左/右（带重叠），用于区分对方/自己消息"""
        x1, y1, x2, y2 = msg_rect
        w = x2 - x1
        split_left = float(self.cfg["incoming_detection"]["split_left_ratio"])
        overlap = float(self.cfg["incoming_detection"]["overlap_ratio"])

        left_end = x1 + int(w * split_left)
        overlap_px = int(w * overlap)

        left_rect = (x1, y1, min(x2, left_end + overlap_px), y2)
        right_rect = (max(x1, left_end - overlap_px), y1, x2, y2)
        return left_rect, right_rect

    def screenshot_region(self, rect: Tuple[int, int, int, int]) -> Image.Image:
        left, top, right, bottom = rect
        with mss.mss() as sct:
            img = sct.grab({"left": left, "top": top, "width": right - left, "height": bottom - top})
            return Image.frombytes("RGB", img.size, img.rgb)

    def save_calibration_screens(self):
        """保存校准截图：整窗 + 标题区 + 消息区 + 左/右拆分区"""
        assert self.win is not None
        ts = time.strftime("%Y%m%d_%H%M%S")

        full_rect = self.win.rect
        title_rect = self._region_from_ratio("title")
        msg_rect = self._region_from_ratio("message")
        left_rect, right_rect = self._split_message_rects(msg_rect)

        self.screenshot_region(full_rect).save(LOG_DIR / f"calib_full_{ts}.png")
        self.screenshot_region(title_rect).save(LOG_DIR / f"calib_title_{ts}.png")
        self.screenshot_region(msg_rect).save(LOG_DIR / f"calib_message_{ts}.png")
        self.screenshot_region(left_rect).save(LOG_DIR / f"calib_message_left_{ts}.png")
        self.screenshot_region(right_rect).save(LOG_DIR / f"calib_message_right_{ts}.png")

        print(f"[CALIB] 已保存校准截图（ts={ts}）")
        print(f"[CALIB] title={title_rect}  message={msg_rect}  left={left_rect}  right={right_rect}")
        print("[CALIB] 如标题识别不稳定：优先调整 regions.title；如新消息触发不准：调整 split_left_ratio/overlap_ratio。")

    # ---- 会话路由（根据聊天标题 OCR 自动切换 person_key） ----

    def maybe_update_chat_key_by_title(self):
        """根据 title OCR 更新当前会话 key（防抖：cooldown）"""
        if self.cfg["session_routing"]["mode"] != "title_ocr":
            return

        now = time.time()
        cooldown = float(self.cfg["session_routing"]["title_update_cooldown"])
        if now - self._last_title_update_ts < cooldown:
            return

        try:
            title_rect = self._region_from_ratio("title")
            img = self.screenshot_region(title_rect)
            lines = ocr_lines_tesseract(img, self.cfg["tesseract_cmd"], self.cfg["tesseract_lang"])
            lines = filter_noise_lines(lines)
            text = sanitize_key(" ".join(lines[:2]))  # 通常标题一行就够，取前两行容错
        except Exception:
            return

        # 标题稳定化：需要连续 N 次识别到同一个标题才切换
        required = int(self.cfg["session_routing"].get("stability_required", 2))
        if text != self._title_candidate:
            self._title_candidate = text
            self._title_candidate_count = 1
            return
        else:
            self._title_candidate_count += 1
            if self._title_candidate_count < max(1, required):
                return

        # 合并相似标题：避免 OCR 抖动把同一个人/群识别成多个 key
        merge_th = int(self.cfg["session_routing"].get("merge_similar_threshold", 92))
        known_keys = list(self.sessions.keys())
        best_key = ""
        best_score = -1
        for k in known_keys:
            s = fuzz.ratio(text, k)
            if s > best_score:
                best_score = s
                best_key = k
        if best_key and best_score >= merge_th:
            text = best_key

        if len(text) < int(self.cfg["session_routing"]["min_title_len"]):
            return

        self._last_title_update_ts = now

        # 标题变化才更新
        if text and text != self._last_title_key:
            self._last_title_key = text
            # 切换会话 key
            self.person_key = text
            # 切换后清空 OCR 记忆，避免把上一聊天尾部当“新消息”
            self.prev_in_lines = []
            self.prev_full_lines = []
            print(f"[CHAT] 切换会话 key -> {self.person_key}")

    # ---- 聚焦输入框与发送（安全发送链路） ----

    def click_relative(self, x_ratio: float, y_ratio: float) -> bool:
        assert self.uia_win is not None
        r = self.uia_win.rectangle()
        x = int(r.left + r.width() * x_ratio)
        y = int(r.top + r.height() * y_ratio)
        try:
            self.uia_win.click_input(coords=(x - r.left, y - r.top))
            return True
        except Exception:
            return False

    def verify_paste_in_input(self, expected: str, delay=0.08) -> bool:
        """Ctrl+A/Ctrl+C 验证粘贴是否落在输入框"""
        clip_before = pyperclip.paste()
        try:
            keyboard.send("ctrl+a"); time.sleep(delay)
            keyboard.send("ctrl+c"); time.sleep(delay)
            got = (pyperclip.paste() or "").strip()
            exp = (expected or "").strip()
            pyperclip.copy(clip_before)

            if not exp:
                return False
            if got == exp:
                return True
            if got.replace(" ", "") == exp.replace(" ", ""):
                return True
            if exp in got:
                return True
            return False
        except Exception:
            try:
                pyperclip.copy(clip_before)
            except Exception:
                pass
            return False

    def send_text(self, text: str) -> bool:
        """发送文本到当前聊天输入框（要求微信前台）"""
        assert self.win is not None and self.uia_win is not None

        if self.cfg["require_foreground_wechat"] and not _is_foreground(self.win.hwnd):
            print("[SEND] 已阻止：微信不在前台。")
            return False

        try:
            self.uia_win.set_focus()
        except Exception:
            _focus_hwnd(self.win.hwnd)

        focused = False
        for xr, yr in self.cfg["click_points"]:
            if self.click_relative(float(xr), float(yr)):
                time.sleep(0.06)
                focused = True
                break
        if not focused:
            print("[SEND] 无法聚焦输入框区域。")
            return False

        clip_before = pyperclip.paste()
        pyperclip.copy(text)
        keyboard.send("ctrl+v"); time.sleep(0.10)
        pyperclip.copy(clip_before)

        if not self.verify_paste_in_input(text):
            print("[SEND] 校验失败：粘贴可能未落在输入框，已中止发送。")
            return False

        keyboard.send("enter")

        # 记录本次发送文本，用于防自触发
        self.state.setdefault("last_sent_text", {})[self.person_key] = text
        save_json(STATE_PATH, self.state)

        # 发送后 2 秒抑制 OCR 触发
        self._suppress_until = time.time() + 2.0

        print("[SEND] 已发送。")
        return True

    # ---- 会话历史 ----

    def get_history(self) -> List[Dict]:
        return self.sessions.get(self.person_key, [])

    def set_history(self, hist: List[Dict]):
        maxn = int(self.cfg["auto_reply"]["max_session_messages"])
        if len(hist) > maxn:
            hist = hist[-maxn:]
        self.sessions[self.person_key] = hist
        save_json(SESSIONS_PATH, self.sessions)

    def reset_current_session(self):
        """重置当前会话：清空历史 + 清空去重状态"""
        self.sessions[self.person_key] = []
        self.state.setdefault("last_sig", {})[self.person_key] = ""
        self.state.setdefault("last_text", {})[self.person_key] = ""
        self.state.setdefault("last_sent_text", {})[self.person_key] = ""
        save_json(SESSIONS_PATH, self.sessions)
        save_json(STATE_PATH, self.state)
        self.prev_in_lines = []
        self.prev_full_lines = []
        print(f"[SESSION] 已重置：{self.person_key}")

    # ---- 限频 ----

    def _rate_bucket(self) -> Dict:
        self.state.setdefault("rate", {})
        self.state["rate"].setdefault(self.person_key, {"t0": time.time(), "count": 0})
        return self.state["rate"][self.person_key]

    def can_auto_reply(self) -> bool:
        bucket = self._rate_bucket()
        now = time.time()
        if now - bucket["t0"] > 3600:
            bucket["t0"] = now
            bucket["count"] = 0
        if bucket["count"] >= int(self.cfg["auto_reply"]["max_replies_per_hour"]):
            return False
        last_send = self.state.setdefault("last_send_ts", {}).get(self.person_key, 0.0)
        if now - float(last_send) < float(self.cfg["auto_reply"]["cooldown_sec"]):
            return False
        return True

    def mark_sent(self):
        bucket = self._rate_bucket()
        bucket["count"] += 1
        self.state.setdefault("last_send_ts", {})[self.person_key] = time.time()
        save_json(STATE_PATH, self.state)

    # ---- OCR 轮询：只以“左侧新消息”为触发 ----

    def poll_once(self):
        assert self.win is not None

        # 会话路由：先更新当前聊天 key（可选）
        self.maybe_update_chat_key_by_title()

        # 发送后短时间抑制 OCR
        if time.time() < self._suppress_until:
            return

        msg_rect = self._region_from_ratio("message")
        left_rect, _right_rect = self._split_message_rects(msg_rect)

        # 视觉差分门控：画面几乎没变化则跳过 OCR（降低 CPU 与误触发）
        dg = self.cfg.get("diff_gate", {})
        if bool(dg.get("enabled", True)):
            now = time.time()
            if now - self._last_diff_ts >= float(dg.get("min_interval_sec", 0.4)):
                self._last_diff_ts = now
                try:
                    hash_size = int(dg.get("hash_size", 16))
                    thresh = int(dg.get("hamming_thresh", 10))
                    if self.cfg["incoming_detection"]["mode"] == "left_only":
                        img_gate = self.screenshot_region(left_rect)
                        h = phash_ahash(img_gate, hash_size=hash_size)
                        if self._prev_left_hash is not None and hamming_distance(h, self._prev_left_hash) <= thresh:
                            return
                        self._prev_left_hash = h
                    else:
                        img_gate = self.screenshot_region(msg_rect)
                        h = phash_ahash(img_gate, hash_size=hash_size)
                        if self._prev_full_hash is not None and hamming_distance(h, self._prev_full_hash) <= thresh:
                            return
                        self._prev_full_hash = h
                except Exception:
                    pass

        # OCR
        try:
            if self.cfg["incoming_detection"]["mode"] == "left_only":
                img = self.screenshot_region(left_rect)
                lines = ocr_lines_tesseract(img, self.cfg["tesseract_cmd"], self.cfg["tesseract_lang"])
                self._tess_ok = True
                lines = filter_noise_lines(lines)
                if not lines:
                    return
                delta = tail_delta(self.prev_in_lines, lines)
                self.prev_in_lines = lines
            else:
                img = self.screenshot_region(msg_rect)
                lines = ocr_lines_tesseract(img, self.cfg["tesseract_cmd"], self.cfg["tesseract_lang"])
                self._tess_ok = True
                lines = filter_noise_lines(lines)
                if not lines:
                    return
                delta = tail_delta(self.prev_full_lines, lines)
                self.prev_full_lines = lines
        except Exception as e:
            if self._tess_ok is not False:
                self._tess_ok = False
                print(f"[OCR] tesseract 未就绪：{e}")
                print("[OCR] 解决：安装 Tesseract 或在 config 中设置 tesseract_cmd 为 tesseract.exe 的绝对路径。")
            return

        if not delta:
            return

        delta_norm = normalize_text(delta)

        # 防自触发：与最近一次发送内容相似则忽略
        last_sent = (self.state.setdefault("last_sent_text", {}).get(self.person_key) or "").strip()
        if last_sent and fuzz.ratio(delta_norm, normalize_text(last_sent)) >= 90:
            sig_ = signature(delta_norm)
            self.state.setdefault("last_sig", {})[self.person_key] = sig_
            self.state.setdefault("last_text", {})[self.person_key] = delta_norm
            save_json(STATE_PATH, self.state)
            return

        sig = signature(delta_norm)
        last_sig = (self.state.setdefault("last_sig", {}).get(self.person_key) or "")
        if sig == last_sig:
            return

        # OCR 抖动去重
        last_text = (self.state.setdefault("last_text", {}).get(self.person_key) or "")
        if last_text and fuzz.ratio(delta_norm, last_text) >= 92:
            self.state["last_sig"][self.person_key] = sig
            self.state["last_text"][self.person_key] = delta_norm
            save_json(STATE_PATH, self.state)
            return

        self.state["last_sig"][self.person_key] = sig
        self.state["last_text"][self.person_key] = delta_norm
        save_json(STATE_PATH, self.state)

        print(f"\n[IN] ({self.person_key}) {delta_norm}")

        # 自动回复条件
        if not (self.armed and self.auto_reply_enabled):
            return
        if not self.can_auto_reply():
            print("[AUTO] 已限频，跳过本次。")
            return

        self.generate_and_maybe_send(delta_norm)

    def generate_and_maybe_send(self, incoming_text: str):
        """生成回复：写会话→待发送→（可选确认）→发送"""
        hist = self.get_history() + [{"role": "user", "content": incoming_text}]

        oa = self.cfg["openai"]
        try:
            reply = openai_generate_reply(
                model=oa["model"],
                system_prompt=oa["system_prompt"],
                temperature=float(oa.get("temperature", 0.4)),
                history=hist[-int(self.cfg["auto_reply"]["max_session_messages"]):],
                user_text="请给出回复：",
                api_key=str(oa.get("api_key", "") or "").strip(),
                base_url=str(oa.get("base_url", "") or "").strip()
            )
        except Exception as e:
            print(f"[LLM] 调用失败：{e}")
            return

        reply = (reply or "").strip()
        if not reply:
            print("[LLM] 回复为空，跳过。")
            return

        # 保存会话
        hist.append({"role": "assistant", "content": reply})
        self.set_history(hist)

        self.pending_reply = reply
        self.pending_deadline = time.time() + float(self.cfg["confirm_timeout_sec"])

        print(f"[LLM] 已生成回复（{len(reply)} 字）")
        print(f"[LLM] {reply[:200]}{'...' if len(reply) > 200 else ''}")

        # 复制到剪贴板，方便你手动编辑/确认
        pyperclip.copy(reply)

        if self.cfg["confirm_before_send"]:
            print("[AUTO] 需要确认：按 Ctrl+Alt+Y 发送（超时作废）。")
            return

        ok = self.send_text(reply)
        if ok:
            self.mark_sent()

    # ---- 快捷键动作 ----

    def toggle_arm(self):
        if self.armed:
            self.armed = False
            self.arm_until = 0.0
            print("[ARM] 已关闭。")
        else:
            self.armed = True
            self.arm_until = time.time() + float(self.cfg["arm_minutes"]) * 60.0
            print(f"[ARM] 已开启，截止 {time.ctime(self.arm_until)}")

    def toggle_auto(self):
        if not self.armed:
            print("[AUTO] 已阻止：未 Arm。")
            return
        self.auto_reply_enabled = not self.auto_reply_enabled
        print(f"[AUTO] {'已开启' if self.auto_reply_enabled else '已关闭'}")

    def confirm_send_pending(self):
        if not self.pending_reply:
            print("[CONFIRM] 当前没有待发送回复。")
            return
        if time.time() > self.pending_deadline:
            print("[CONFIRM] 待发送回复已超时作废。")
            self.pending_reply = None
            return

        ok = self.send_text(self.pending_reply)
        if ok:
            self.mark_sent()
        self.pending_reply = None

    def manual_send_clipboard(self):
        if (not self.armed) or (time.time() > self.arm_until):
            print("[SEND] 已阻止：未 Arm 或已超时。")
            return
        text = (pyperclip.paste() or "").strip()
        if not text:
            print("[SEND] 剪贴板为空。")
            return
        self.send_text(text)

    def set_person_slot(self, idx: int):
        """备用：手动切换 slot，会关闭 title_ocr 的自动切换效果（除非你把 mode 改回 slot）"""
        idx = max(1, min(idx, 10))
        self.person_key = self.slot_keys[idx - 1]
        self.prev_in_lines = []
        self.prev_full_lines = []
        print(f"[CHAT] 手动切换会话 key -> {self.person_key}（如需禁用 title OCR，请在 config 中设置 session_routing.mode=slot）")

    def tick(self):
        if self.armed and time.time() > self.arm_until:
            self.armed = False
            self.arm_until = 0.0
            print("[ARM] 已超时关闭。")

        if self.pending_reply and time.time() > self.pending_deadline:
            print("[CONFIRM] 待发送回复已超时作废。")
            self.pending_reply = None

# ---------- 配置合并 ----------

def ensure_config():
    """确保 config 存在，并补齐缺失字段（不会覆盖你已有的自定义项）"""
    if not CONFIG_PATH.exists():
        save_json(CONFIG_PATH, DEFAULT_CONFIG)
        print(f"[INIT] 已创建配置：{CONFIG_PATH.name}")

    cfg = load_json(CONFIG_PATH, DEFAULT_CONFIG)

    def deep_merge(dst, src):
        for k, v in src.items():
            if k not in dst:
                dst[k] = v
            else:
                if isinstance(v, dict) and isinstance(dst.get(k), dict):
                    deep_merge(dst[k], v)
        return dst

    cfg = deep_merge(cfg, DEFAULT_CONFIG)
    save_json(CONFIG_PATH, cfg)
    return cfg

def main():
    # 可选：从同目录 .env 读取 KEY（推荐，本机使用，不要提交到仓库）
    load_local_dotenv(APP_DIR / ".env")

    cfg = ensure_config()
    bot = WeChatAutoBot(cfg)
    bot.connect()

    # 提示：如果没有设置 API Key，会导致 [LLM] 调用失败
    oa = cfg.get("openai", {})
    if not (str(oa.get("api_key", "")).strip() or os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        print("[KEY] 未检测到 KEY。请：1) 设置环境变量 OPENAI_API_KEY 或 GEMINI_API_KEY；或 2) 在 config 的 openai.api_key 中填写；或 3) 在同目录创建 .env 写入 GEMINI_API_KEY=...")

    # 注册快捷键
    keyboard.add_hotkey("ctrl+alt+s", bot.toggle_arm)
    keyboard.add_hotkey("ctrl+alt+a", bot.toggle_auto)
    keyboard.add_hotkey("ctrl+alt+g", bot.manual_send_clipboard)
    keyboard.add_hotkey("ctrl+alt+y", bot.confirm_send_pending)
    keyboard.add_hotkey("ctrl+alt+c", bot.save_calibration_screens)
    keyboard.add_hotkey("ctrl+alt+r", bot.reset_current_session)
    keyboard.add_hotkey("ctrl+alt+q", lambda: os._exit(0))

    for i in range(1, 10):
        keyboard.add_hotkey(f"ctrl+alt+{i}", lambda i=i: bot.set_person_slot(i))
    keyboard.add_hotkey("ctrl+alt+0", lambda: bot.set_person_slot(10))

    print("\n快捷键：")
    print("  Ctrl+Alt+S Arm武装 | Ctrl+Alt+A 自动回复开关 | Ctrl+Alt+G 手动发送剪贴板")
    print("  Ctrl+Alt+Y 确认发送待回复 | Ctrl+Alt+C 保存校准截图")
    print("  Ctrl+Alt+R 重置当前会话 | Ctrl+Alt+Q 紧急退出")
    print("  Ctrl+Alt+1..0 备用会话槽切换（必要时用）")
    print("\n下一步：先按 Ctrl+Alt+C 保存一次校准截图，重点检查 logs/calib_title_*.png 是否截到聊天标题。")

    while True:
        bot.tick()
        try:
            bot.poll_once()
        except Exception as e:
            print(f"[POLL] 轮询异常：{e}")
        time.sleep(float(cfg["poll_interval_sec"]))

if __name__ == "__main__":
    main()
