# 安装指南（Windows）

## 1. 环境与依赖

- Windows 10/11，微信桌面端已安装。运行程序依赖 Win32 与 UI Automation，不能在 Linux/macOS 上直接操作微信。
- Python 3.10+（代码使用 `int.bit_count()`）；建议先在 Python 3.11/3.12 独立虚拟环境中验证。
- Tesseract OCR 可执行程序，以及 `chi_sim`、`eng` 语言数据。
- 自动生成草稿需要可用的兼容 OpenAI Chat Completions 的 API 服务、模型及密钥。手动剪贴板发送不需要模型服务。

获取本仓库后，在目录内运行以下 PowerShell 命令，无需改变系统执行策略或以管理员运行：

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

`requirements.txt` 列出直接运行依赖，尚未提供经过 Windows 验证的版本锁定文件。安装成功并不代表 UI 自动化已兼容你的微信版本。依赖用途：

- `psutil`：进程检查
- `keyboard`：全局快捷键与模拟按键
- `pyperclip`：系统剪贴板
- `pywinauto`：Windows 窗口操作
- `mss`、`Pillow`：截图与图像处理
- `pytesseract`：调用外部 Tesseract（不会替你安装 OCR 引擎）
- `openai`：兼容接口客户端，仅生成草稿时调用
- `rapidfuzz`：消息相似度去重，发送收件人检查不使用模糊匹配

## 2. 安装 Tesseract 与语言数据

参照 [Tesseract 官方安装说明的 Windows 部分](https://tesseract-ocr.github.io/tessdoc/Installation.html#windows) 获取其链接的 UB Mannheim Windows 构建。不要把上游源码发布包当成 Windows 安装程序。

安装所需语言数据后，在 PowerShell 验证（按实际安装位置调整路径）：

```powershell
& 'C:\Program Files\Tesseract-OCR\tesseract.exe' --version
& 'C:\Program Files\Tesseract-OCR\tesseract.exe' --list-langs
```

语言列表应包含 `chi_sim` 和 `eng`。若已加入 PATH，可用 `where.exe tesseract` 查看位置。

## 3. 首次运行与校准

```powershell
.\.venv\Scripts\python.exe .\wechat_auto_bot_single_v6_stable_resend.py
```

脚本创建 `wechat_auto_bot_config.json` 和 `logs/`。`sessions.json`、`state.json` 会在相关会话或状态首次写入时出现。终端消息默认只输出到控制台；`logs/` 用于热键保存的校准截图。

先保持 Arm 和 AutoReply 关闭。程序启动仍会进行本地 OCR 与去重状态记录。用无敏感内容的测试聊天校准：

1. 确认终端 `[SELECTED]` 是正确的微信主窗口；不要继续操作其他窗口。
2. 微信保持前台，按 `Ctrl+Alt+C` 保存截图。
3. 检查 `logs/calib_title_*.png` 是否仅覆盖聊天标题，消息区是否正确。
4. 退出脚本，编辑配置，重新启动；配置不支持热重载。

相关配置片段（不是可直接覆盖整个配置文件的完整 JSON）：

```json
{
  "tesseract_cmd": "C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
  "tesseract_lang": "chi_sim+eng",
  "regions": {
    "title": {"x1": 0.28, "y1": 0.02, "x2": 0.78, "y2": 0.11},
    "message": {"x1": 0.28, "y1": 0.10, "x2": 0.98, "y2": 0.74}
  }
}
```

坐标是窗口宽高的相对比例，需要针对窗口布局、缩放和微信版本校准。

## 4. API 密钥与服务

生成草稿会将 OCR 文字、会话历史和系统提示发送到 `openai.base_url` 指向的服务。先读 [PRIVACY.md](PRIVACY.md)。

可复制 `.env.example` 为 `.env`，只填写目标服务的密钥。不要提交密钥，也不要把真实密钥放进 Issue。脚本自己的简易 `.env` 读取器只支持 `KEY=value`，不支持 shell 展开或完整 dotenv 语法；已有环境变量不会被覆盖。

密钥优先级：配置 `openai.api_key` → `OPENAI_API_KEY` → `GEMINI_API_KEY`。不要同时放置不同服务的有效密钥，否则可能把错误密钥发往配置的 endpoint。推荐配置中的 `api_key` 保持空字符串，只设置需要的环境变量。

默认代码配置为 Gemini 兼容 endpoint：`https://generativelanguage.googleapis.com/v1beta/openai/`，默认模型名是 `gemini-2.5-flash-lite`。这只是仓库默认值，不保证该模型在你的账户中仍可用；请使用服务商当前可用且支持 Chat Completions 的模型。

切换服务时必须同时检查 `base_url`、`model` 和密钥。`provider` 字段只是配置标签，不会自动切换 endpoint 或密钥选择。不要只改 `provider` 或环境变量就开始调用。

## 5. 首次验证

先运行 [离线测试](TESTING.md)，再按照手动检查清单使用无敏感内容的测试环境。当前改动未做 Windows / 微信真实发送验证；请保留 `confirm_before_send=true`，不要开启无人值守发送。
