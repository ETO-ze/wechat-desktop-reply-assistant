# 常见问题与故障排除

本指南列出了使用 **WeChat 桌面回复助手** 时最常见的问题及其解决方法。如果你在使用过程中遇到问题，可以先从这里查找答案。

## Q1：脚本检测不到微信窗口

- 确认微信 PC 版本已运行，并且主窗口未最小化。
- 脚本默认搜索进程名称 `weixin.exe`；如果你的 WeChat 安装名称不同，请在 `wechat_auto_bot_config.json` 的 `preferred_proc_names` 中加入相应名称。
- 某些系统下可能需要以管理员权限运行脚本，以便访问其他窗口句柄。尝试“以管理员身份运行” PowerShell/命令行。

## Q2：提示 "tesseract is not installed or it's not in your PATH"

- 请按《安装指南》中的说明安装 Tesseract，并确认其路径正确。
- 在配置文件中设置正确的 `tesseract_cmd` 值。例如：
  ```json
  "tesseract_cmd": "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
  ```
- 执行 `tesseract --version` 检查是否能正常调用。

## Q3：脚本误把自己发送的消息当作新消息

- 检查配置 `incoming_detection.mode` 是否为 `left_only`（默认）。
- 校准 `regions.message`，确保左侧区域仅覆盖对方气泡，不覆盖自己发出的气泡。
- 可适当增大 `incoming_detection.split_left_ratio` 或 `overlap_ratio`，让左侧检测区域更宽，减少靠中间的自己消息。

## Q4：OCR 识别结果不稳定

- OCR 稳定性与截图区域和环境光照有关。可尝试提升 `ocr_stability.required_hits`（在多次连续识别一致后才判定为新消息）。
- 调整 `diff_gate.hamming_thresh` 或二级差分阈值，避免轻微界面变化导致 OCR 过度触发。
- 保持微信窗口尺寸一致，并避免在 Windows 桌面上快速切换缩放比例。

## Q5：热键无响应或冲突

- 确保脚本在 `Arm` 状态（已按 `Ctrl+Alt+S`）。未武装时部分功能不会生效。
- 部分系统或安全软件会拦截全局热键。尝试以管理员身份运行脚本，或修改热键组合避免与系统快捷键冲突。
- 如果热键仍然无响应，请检查是否存在其他程序注册了相同的热键。

## Q6：调用 LLM 接口报错或超时

- 确认 `.env` 文件中的 `GEMINI_API_KEY` 或 `OPENAI_API_KEY` 是否正确，且网络环境可以访问对应的 API。
- 如果使用 Gemini 兼容接口，请确保 `base_url` 配置指向 `https://generativelanguage.googleapis.com/v1beta/openai/`。
- API 请求可能受到速率限制。检查 `auto_reply.cooldown_sec` 和 `max_replies_per_hour` 的设置，避免频繁触发。

## Q7：CPU 占用较高

- 调整 `poll_interval_sec` 以降低轮询频率，例如从 1.0 增加到 2.0 秒。
- 启用 `diff_gate.enabled` 并调高阈值，可在画面不变化时跳过 OCR，节省资源。
- 关闭不必要的应用，确保电脑资源充足。

## Q8：如何完全关闭自动发送

- 保持配置 `confirm_before_send=true`（默认），脚本在生成草稿后不自动发送，需要按 `Ctrl+Alt+Y` 确认。
- 如果你不想生成自动草稿，可以保持 `auto_reply.enabled_default=false`，仅在需要时按 `Ctrl+Alt+A` 手动开启。

---

如以上方法仍无法解决问题，建议对照脚本中的打印日志（位于 `./logs/` 中）进一步定位问题，并根据日志调整配置参数。
