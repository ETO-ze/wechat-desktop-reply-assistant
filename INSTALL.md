# 安装指南（Windows）

本指南详细说明如何配置和运行 **WeChat 桌面回复助手**，包括安装依赖、配置 OCR 以及 API 密钥的设置。此指南假设你使用的是 Windows 10 或 Windows 11，并且已具备基本的 Python 使用经验。

## 1. 获取代码

将仓库中的 Python 脚本复制到你选择的目录。例如：

```
D:\wechat-desktop-reply-assistant\
  wechat_auto_bot_single_v6_stable_resend.py
```

在正式运行前，请确保此目录对当前用户具有读写权限，避免放在系统盘根目录下。

## 2. 创建虚拟环境并安装依赖

打开 PowerShell 或终端，切换到项目目录，然后运行以下命令以创建虚拟环境并安装所需依赖：

```powershell
# 创建虚拟环境
py -3 -m venv .venv

# 激活虚拟环境（PowerShell）
\.venv\Scripts\Activate.ps1

# 升级 pip
python -m pip install --upgrade pip

# 安装项目依赖
python -m pip install psutil keyboard pyperclip pywinauto mss pillow pytesseract openai rapidfuzz
```

依赖包说明：

| 包名       | 用途说明                               |
|------------|----------------------------------------|
| psutil     | 获取进程信息，识别 WeChat 主窗口         |
| keyboard   | 注册全局快捷键                         |
| pyperclip  | 读写系统剪贴板                         |
| pywinauto  | 控制窗口、获取控件                     |
| mss        | 高效多屏截图                          |
| pillow     | 图像处理                               |
| pytesseract| 使用 Tesseract OCR 识别文本           |
| openai     | 调用 Gemini 或 OpenAI 兼容接口（可选） |
| rapidfuzz  | 文本相似度计算                         |

## 3. 安装并配置 Tesseract OCR

本工具依赖 [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) 来识别微信界面中的文字。请按以下步骤安装：

1. 从 [tesseract-ocr/tesseract 的发布页面](https://github.com/tesseract-ocr/tesseract/releases) 下载 Windows 安装包。
2. 安装完成后，默认路径通常为：
   - `C:\\Program Files\\Tesseract-OCR\\tesseract.exe`
3. 验证安装是否成功：
   ```powershell
   where tesseract
   tesseract --version
   ```
4. 在生成的 `wechat_auto_bot_config.json` 中设置 Tesseract 路径，例如：
   ```json
   "tesseract_cmd": "C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
   "tesseract_lang": "chi_sim+eng"
   ```

## 4. 配置 API 密钥（可选）

如果你希望启用自动生成回复功能，需要提供一个可兼容 OpenAI 接口的 API 密钥。当前默认配置使用 Google Gemini 的 OpenAI 兼容接口（你也可以切换回 OpenAI）。推荐将密钥写入项目目录的 `.env` 文件（不要提交到 GitHub）：

```
GEMINI_API_KEY=你的密钥
# 或
OPENAI_API_KEY=你的密钥
```

脚本会自动读取同目录 `.env` 文件，将密钥写入环境变量（仅在系统环境变量未设置时生效）。

## 5. 首次运行与配置生成

激活虚拟环境后，首次运行脚本：

```powershell
python .\wechat_auto_bot_single_v6_stable_resend.py
```

程序将生成以下文件（若不存在）：

- `wechat_auto_bot_config.json`：主配置文件（可根据需要调整区域坐标、差分阈值等）。
- `sessions.json`：保存每个会话的历史对话上下文。
- `state.json`：保存发送去重、节流状态。
- `./logs/`：保存校准截图及日志。

## 6. 校准 OCR 区域

为确保 OCR 精准识别，需要手动校准截图：

1. 保持微信 PC 客户端处于主窗口且尺寸稳定。
2. 进入某个对话窗口。
3. 按 `Ctrl+Alt+C`，脚本会将当前窗口、聊天标题区域和消息区域的截图保存到 `./logs` 目录。
4. 根据截图调整 `wechat_auto_bot_config.json` 中 `regions.title` 和 `regions.message` 的值（它们以相对坐标 `[0,1]` 表示）。

示例：

```json
"regions": {
  "title":   {"x1": 0.28, "y1": 0.02, "x2": 0.78, "y2": 0.11},
  "message": {"x1": 0.28, "y1": 0.10, "x2": 0.98, "y2": 0.74}
}
```

通过日志中的截图对照调整这些值，确保标题区域仅覆盖对话标题，消息区域覆盖聊天内容。

## 7. 后续步骤

校准完成后，你可以参考《使用指南》了解如何启动、Arm 武装、自动回复、确认发送等操作。本安装文件仅聚焦环境配置和依赖安装。如有问题，请先阅读《常见问题与故障排除》文档。
