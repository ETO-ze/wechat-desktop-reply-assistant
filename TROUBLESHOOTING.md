# 常见问题与故障排除

## 检测不到或选错微信窗口

确认微信主窗口可见且未最小化，检查终端 `[SELECTED]` 的进程和标题。默认优先查找 `weixin.exe`，其他版本可按实际进程名调整 `preferred_proc_names`。如果窗口不正确，先退出，不要尝试发送。通常不应先提升管理员权限；优先保持微信和脚本以同一普通用户运行。

## Tesseract 不可用或中文识别为空

按 [安装指南](INSTALL.md) 设置真实 `tesseract_cmd` 路径，运行 `--version` 和 `--list-langs`。安装 `pytesseract` 不等于安装 OCR 引擎；使用 `chi_sim+eng` 时需同时安装两种语言数据。

## 标题变化或无法验证，草稿被作废

这是发送保护。检查 `calib_title_*.png`，避免标题区包含成员状态、搜索框或其他文字；保持微信前台，缩放与布局稳定。不要通过关闭前台检查、调低相似度或复用 slot 绕过保护。标题身份使用精确匹配，旧配置 `merge_similar_threshold` 已不生效。

## 自己的消息触发了回复

保持 `incoming_detection.mode=left_only`，重新校准消息区域。较大的 `split_left_ratio` / `overlap_ratio` 会扩大左侧检测范围，也可能纳入更多右侧气泡；请对照截图调整，不能盲目增大。左右分割无法保证完全区分消息发送者。

## OCR 抖动、重复或漏消息

标题可通过 `session_routing.stability_required` 控制连续命中次数。程序没有 `ocr_stability.required_hits` 配置项，也没有通用的消息 OCR 多帧稳定性保证。调整 `diff_gate.hamming_thresh` 时注意：更大的值会跳过更多变化，也更容易漏掉细微的新消息。修改配置后重启。

## 草稿已显示，但没有发送

先检查 Arm 是否仍有效、草稿是否已超过默认 30 秒、前台聊天是否仍为原收件人。切换会话、关闭 AutoReply、重置会话或重新 Arm 都会作废旧草稿。草稿生成只复制到剪贴板，不会立即粘贴到微信。

发送失败后草稿已被消费，避免重复热键误发。人工检查输入框残留文字；不要直接按回车。`Ctrl+Alt+G` 会发送当前剪贴板内容，与草稿确认是不同操作。

## LLM 接口报错

检查 `base_url`、当前账户支持的 `model`、对应密钥与网络。`provider` 不会自动选择 endpoint。密钥优先级为配置中的值、`OPENAI_API_KEY`、`GEMINI_API_KEY`；不要同时配置不同服务密钥后依赖程序猜测。默认模型名可能随服务变化而失效。错误报告须先删去密钥及聊天内容。

## 热键冲突或 CPU 占用较高

检查其他软件是否占用相同热键。部分操作需要 Arm，但校准和退出不需要。可增大 `poll_interval_sec`，保留差分门控；不要通过放松发送检查解决性能问题。发送链路的重复标题 OCR 会增加延迟。

## 如何停止

保留 `confirm_before_send=true` 可关闭无确认发送。关闭 AutoReply 可停止自动生成，关闭 Arm 可撤销发送许可；只有 `Ctrl+Alt+Q` / 退出进程才停止全部 OCR。默认日志在控制台，`logs/` 是按热键生成的校准截图目录。
