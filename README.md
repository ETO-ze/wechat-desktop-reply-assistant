# WeChat Desktop Reply Assistant

Windows 微信桌面端的实验性半自动回复工具：截图 → 本地 Tesseract OCR → 调用兼容 OpenAI 的模型接口 → 生成草稿 → 热键确认发送。

默认 AutoReply 关闭，默认需要确认发送。**目前仅验证了离线模拟测试，未完成 Windows / 微信真实界面验证，不能把它当作可靠的无人值守机器人。**

## 功能与边界

- 使用聊天标题路由会话，左侧消息区域 OCR 与视觉差分减少重复触发；这些都是启发式判断，不能保证消息或联系人识别正确。
- 草稿绑定生成时的会话、完整 OCR 标题、窗口和有效期；切换会话、关闭 Arm / AutoReply、重置会话、手动剪贴板发送或 Arm 到期会使旧草稿失效。
- 发送前重新识别标题；粘贴和回车前再次检查 Arm、前台窗口及收件人。标题为空、变化或无法读取时中止。
- 相似标题不再模糊合并。手动 slot 模式仍需要可读标题，不能绕过发送检查。
- 草稿生成后显示在终端并复制到剪贴板，**不会立即粘贴到微信输入框**。确认发送的是当时保存的草稿，修改剪贴板不会修改该草稿。
- 相同显示名称、OCR 一致误识别、校准错误，以及最后一次检查与键盘事件之间的桌面变化仍有误发风险。发送过程中不要切换聊天、移动窗口或操作键鼠。

## 快速开始

需要 Windows 10/11、Python 3.10+、微信桌面端、Tesseract 与 `chi_sim` / `eng` 语言数据。建议使用独立虚拟环境；详细步骤见 [安装指南](INSTALL.md)。

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\wechat_auto_bot_single_v6_stable_resend.py
```

1. 先阅读 [隐私与安全说明](PRIVACY.md)，确认愿意将相关聊天文字发送给配置的模型服务。
2. 保持微信主窗口前台，使用无敏感内容的测试对话；按 `Ctrl+Alt+C` 保存校准截图并调整配置，修改后重启脚本。
3. 按 `Ctrl+Alt+S` 开启 Arm，再按 `Ctrl+Alt+A` 开启草稿生成。
4. 检查终端显示的收件人和草稿，保持该聊天在前台，按 `Ctrl+Alt+Y` 确认。默认草稿 30 秒过期。
5. 停止使用时按 `Ctrl+Alt+Q` 退出。Arm 关闭不会停止 OCR 轮询，只有退出程序才会停止。

## 文档

- [INSTALL.md](INSTALL.md)：安装、依赖、密钥与模型配置
- [USER_GUIDE.md](USER_GUIDE.md)：快捷键、草稿生命周期与限制
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md)：校准与常见错误
- [PRIVACY.md](PRIVACY.md)：本地文件、剪贴板与第三方 API 数据流
- [TESTING.md](TESTING.md)：离线测试及尚未完成的 Windows 检查清单

## 离线测试

测试只使用 Python 标准库，所有 Windows、键盘、剪贴板、截图、OCR 与模型调用均被模拟，不需要安装运行依赖，也不会读取真实聊天或发送消息。

```sh
python -m unittest discover -s tests -v
python -m py_compile wechat_auto_bot_single_v6_stable_resend.py
```

## 安全、合规与贡献

请遵守微信平台条款，自动化可能触发账号限制。Arm、确认发送与标题检查只能降低风险，不能保证安全或免于平台处罚。不要用于敏感聊天或无人值守发送。

提交 Issue / PR 时仅使用虚构数据。不要上传聊天记录、截图、配置或密钥。`.gitignore` 不会清除已经提交过的文件，也不能代替人工检查。

原 README 声明 MIT，但仓库当前没有独立 LICENSE 文件；许可证文本需要维护者确认。本次修改不新增或更改许可授权。
