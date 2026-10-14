"""Offline regression tests. No real Windows hooks, screenshots, clipboard or API."""
import copy
import ctypes
from difflib import SequenceMatcher
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import MagicMock, patch


SOURCE = Path(__file__).resolve().parents[1] / "wechat_auto_bot_single_v6_stable_resend.py"


def load_bot_module(directory):
    modules = {name: types.ModuleType(name) for name in (
        "psutil", "keyboard", "pyperclip", "mss", "PIL", "pywinauto", "rapidfuzz")}
    modules["keyboard"].send = MagicMock()
    modules["pyperclip"].paste = MagicMock(return_value="original clipboard")
    modules["pyperclip"].copy = MagicMock()
    modules["PIL"].Image = MagicMock()
    modules["PIL"].ImageOps = MagicMock()
    modules["PIL"].ImageEnhance = MagicMock()
    modules["pywinauto"].Application = MagicMock()
    modules["rapidfuzz"].fuzz = types.SimpleNamespace(
        ratio=lambda a, b: SequenceMatcher(None, a, b).ratio() * 100)
    module = types.ModuleType("wechat_bot_under_test")
    # All startup filesystem writes land in a temporary test directory.
    module.__file__ = str(Path(directory) / SOURCE.name)
    modules[module.__name__] = module
    with patch.dict(sys.modules, modules), \
            patch.object(ctypes, "windll", MagicMock(), create=True), \
            patch.object(ctypes, "WINFUNCTYPE", lambda *args: lambda fn: fn, create=True):
        exec(compile(SOURCE.read_text(encoding="utf-8"), str(SOURCE), "exec"), module.__dict__)
    module.openai_generate_reply = MagicMock(return_value="test reply")
    module.ocr_lines_tesseract = MagicMock(side_effect=AssertionError("Real OCR forbidden"))
    module._is_foreground = MagicMock(return_value=True)
    module._get_rect = MagicMock(return_value=(0, 0, 800, 600))
    module._pid_of = MagicMock(return_value=123)
    module.save_json = MagicMock()
    return module


class DraftSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.mod = load_bot_module(self.temp.name)
        self.bot = self.mod.WeChatAutoBot(copy.deepcopy(self.mod.DEFAULT_CONFIG))
        self.bot.person_key = "Alice"
        self.bot._last_title_key = "Alice"
        self.bot.win = self.mod.SelectedWindow(1, 123, "Weixin.exe", "微信", "Qt", (0, 0, 800, 600))
        self.bot.uia_win = MagicMock()
        self.bot.armed = True
        self.bot.arm_until = self.mod.time.time() + 600
        self.bot.auto_reply_enabled = True
        self.bot._read_current_title = MagicMock(return_value="Alice")
        self.bot.screenshot_region = MagicMock(side_effect=AssertionError("Screenshots forbidden"))
        self.bot.click_relative = MagicMock(return_value=True)
        self.bot.verify_paste_in_input = MagicMock(return_value=True)
        self.sleep = patch.object(self.mod.time, "sleep", return_value=None)
        self.sleep.start()
        self.addCleanup(self.sleep.stop)

    def draft(self):
        self.bot.generate_and_maybe_send("hello")
        self.assertIsNotNone(self.bot.pending_reply)
        return self.bot.pending_reply

    def assert_no_enter(self):
        self.assertNotIn(unittest.mock.call("enter"), self.mod.keyboard.send.call_args_list)

    def test_unrelated_window_cannot_be_selected_as_wechat(self):
        window = {"visible": True, "area": 99999999, "proc": "notepad.exe",
                  "title": "微信 notes", "class": "Qt"}
        self.assertEqual(self.bot.score(window), -1)
        self.bot.enumerate_windows = MagicMock(return_value=[window])
        with self.assertRaises(RuntimeError):
            self.bot.pick_window()
        window["proc"] = "Weixin.exe"
        self.assertGreater(self.bot.score(window), 0)

    def test_matching_draft_sends_once_and_records_origin(self):
        self.draft()
        self.bot.confirm_send_pending()
        self.bot.confirm_send_pending()
        self.assertEqual(self.mod.keyboard.send.call_args_list.count(unittest.mock.call("enter")), 1)
        self.assertEqual(self.bot.state["last_sent_text"], {"Alice": "test reply"})
        self.assertEqual(self.bot.sessions["Alice"][-1], {"role": "assistant", "content": "test reply"})
        self.assertIsNone(self.bot.pending_reply)

    def test_unconfirmed_reply_is_not_stored_as_sent_history(self):
        self.draft()
        self.assertEqual(self.bot.sessions["Alice"], [{"role": "user", "content": "hello"}])
        self.assert_no_enter()

    def test_recipient_change_without_poll_blocks_confirmation(self):
        self.draft()
        self.bot._read_current_title.return_value = "Bob"
        self.bot.confirm_send_pending()
        self.assert_no_enter()
        self.bot.click_relative.assert_not_called()
        self.assertIsNone(self.bot.pending_reply)

    def test_similar_title_and_unknown_title_do_not_pass(self):
        for title in ("Alicé", "", "Alice!"):
            with self.subTest(title=title):
                self.bot._read_current_title.return_value = "Alice"
                self.draft()
                self.bot._read_current_title.return_value = title
                self.bot.confirm_send_pending()
                self.assert_no_enter()

    def test_switch_invalidation_happens_before_title_stabilizes(self):
        self.draft()
        self.bot._read_current_title.return_value = "Bob"
        self.assertFalse(self.bot.maybe_update_chat_key_by_title())
        self.assertEqual(self.bot.person_key, "Alice")
        self.assertIsNone(self.bot.pending_reply)

    def test_similar_titles_are_not_merged_into_another_history(self):
        self.bot.sessions["Alice!"] = [{"role": "user", "content": "private"}]
        self.bot._read_current_title.return_value = "Alice?"
        self.bot.cfg["session_routing"]["stability_required"] = 1
        self.assertTrue(self.bot.maybe_update_chat_key_by_title())
        self.assertEqual(self.bot.person_key, "Alice?")
        self.assertEqual(self.bot.get_history(), [])

    def test_slot_change_clears_draft(self):
        self.draft()
        self.bot.set_person_slot(2)
        self.bot.confirm_send_pending()
        self.assertIsNone(self.bot.pending_reply)
        self.assert_no_enter()

    def test_slot_mode_still_checks_actual_recipient(self):
        self.bot.cfg["session_routing"]["mode"] = "slot"
        self.bot.person_key = "slot1"
        self.draft()
        self.bot._read_current_title.return_value = "Bob"
        self.bot.confirm_send_pending()
        self.assert_no_enter()

    def test_disarm_then_rearm_does_not_revive_draft(self):
        self.draft()
        self.bot.toggle_arm()
        self.bot.toggle_arm()
        self.bot.confirm_send_pending()
        self.assert_no_enter()

    def test_arm_expiry_blocks_confirmation_without_tick(self):
        self.draft()
        self.bot.arm_until = self.mod.time.time() - 1
        self.bot.confirm_send_pending()
        self.assert_no_enter()

    def test_tick_disarm_and_draft_expiry_clear_pending(self):
        self.draft()
        self.bot.arm_until = self.mod.time.time() - 1
        self.bot.tick()
        self.assertFalse(self.bot.armed)
        self.assertIsNone(self.bot.pending_reply)
        self.bot.toggle_arm()
        self.draft()
        d = self.bot.pending_reply
        self.bot.pending_reply = self.mod.PendingReply(d.text, d.context, self.mod.time.time() - 1)
        self.bot.tick()
        self.assertIsNone(self.bot.pending_reply)

    def test_expired_draft_is_consumed_without_send(self):
        d = self.draft()
        self.bot.pending_reply = self.mod.PendingReply(d.text, d.context, self.mod.time.time() - 1)
        self.bot.confirm_send_pending()
        self.assert_no_enter()
        self.assertIsNone(self.bot.pending_reply)

    def test_background_window_blocks_even_with_legacy_flag_false(self):
        self.draft()
        self.bot.cfg["require_foreground_wechat"] = False
        self.mod._is_foreground.return_value = False
        self.bot.confirm_send_pending()
        self.assert_no_enter()
        self.bot.uia_win.set_focus.assert_not_called()

    def test_window_replacement_blocks_draft(self):
        self.draft()
        self.bot.win.hwnd = 2
        self.bot.confirm_send_pending()
        self.assert_no_enter()

    def test_delayed_reply_discarded_after_switch_disarm_or_auto_off(self):
        for action in (lambda: self.bot.set_person_slot(2), self.bot.toggle_arm, self.bot.toggle_auto):
            with self.subTest(action=action):
                self.bot.person_key = "Alice"
                self.bot.armed = self.bot.auto_reply_enabled = True
                self.bot.arm_until = self.mod.time.time() + 600
                self.mod.openai_generate_reply.side_effect = lambda **kwargs: (action(), "late reply")[1]
                self.bot.generate_and_maybe_send("hello")
                self.assertIsNone(self.bot.pending_reply)
                self.assert_no_enter()
                self.assertNotIn("slot2", self.bot.sessions)

    def test_delayed_reply_discarded_after_switch_away_and_back(self):
        def response(**kwargs):
            self.bot.set_person_slot(2)
            self.bot.person_key = "Alice"
            return "late reply"
        self.mod.openai_generate_reply.side_effect = response
        self.bot.generate_and_maybe_send("hello")
        self.assertIsNone(self.bot.pending_reply)
        self.assertEqual(self.bot.sessions, {})

    def test_recipient_change_during_api_call_discards_reply(self):
        def response(**kwargs):
            self.bot._read_current_title.return_value = "Bob"
            return "late reply"
        self.mod.openai_generate_reply.side_effect = response
        self.bot.generate_and_maybe_send("hello")
        self.assertIsNone(self.bot.pending_reply)
        self.assert_no_enter()

    def test_disarm_during_final_validation_prevents_enter(self):
        self.draft()
        def verified(*args):
            self.bot.toggle_arm()
            return True
        self.bot.verify_paste_in_input.side_effect = verified
        self.bot.confirm_send_pending()
        self.assert_no_enter()

    def test_title_change_after_paste_prevents_enter(self):
        self.draft()
        def verified(*args):
            self.bot._read_current_title.return_value = "Bob"
            return True
        self.bot.verify_paste_in_input.side_effect = verified
        self.bot.confirm_send_pending()
        self.assertIn(unittest.mock.call("ctrl+v"), self.mod.keyboard.send.call_args_list)
        self.assert_no_enter()

    def test_paste_failure_restores_clipboard_and_does_not_enter(self):
        self.draft()
        self.mod.keyboard.send.side_effect = RuntimeError("mock paste failure")
        with self.assertRaises(RuntimeError):
            self.bot.confirm_send_pending()
        self.assertEqual(self.mod.pyperclip.copy.call_args, unittest.mock.call("original clipboard"))
        self.assert_no_enter()
        self.assertFalse(self.bot._send_lock.locked())

    def test_failed_validation_does_not_send_or_keep_draft(self):
        self.draft()
        self.bot.verify_paste_in_input.return_value = False
        self.bot.confirm_send_pending()
        self.assert_no_enter()
        self.assertIsNone(self.bot.pending_reply)

    def test_auto_send_consumes_draft_so_confirm_cannot_resend(self):
        self.bot.cfg["confirm_before_send"] = False
        self.bot.generate_and_maybe_send("hello")
        self.bot.confirm_send_pending()
        self.assertEqual(self.mod.keyboard.send.call_args_list.count(unittest.mock.call("enter")), 1)
        self.assertIsNone(self.bot.pending_reply)

    def test_manual_send_invalidates_existing_draft(self):
        self.draft()
        self.mod.pyperclip.paste.return_value = "test reply"
        self.bot.manual_send_clipboard()
        self.bot.confirm_send_pending()
        self.assertIsNone(self.bot.pending_reply)
        self.assertEqual(self.mod.keyboard.send.call_args_list.count(unittest.mock.call("enter")), 1)

    def test_reset_after_send_does_not_resurrect_history(self):
        self.draft()
        def send(*args):
            self.bot.reset_current_session()
            return True
        self.bot.send_text = MagicMock(side_effect=send)
        self.bot.confirm_send_pending()
        self.assertEqual(self.bot.sessions["Alice"], [])

    def test_reset_invalidates_draft(self):
        self.draft()
        self.bot.reset_current_session()
        self.bot.confirm_send_pending()
        self.assert_no_enter()

    def test_simultaneous_confirmations_do_not_duplicate_send(self):
        self.draft()
        entered, release = threading.Event(), threading.Event()
        def verify(*args):
            entered.set()
            if not release.wait(2):
                raise AssertionError("test synchronization timed out")
            return True
        self.bot.verify_paste_in_input.side_effect = verify
        first = threading.Thread(target=self.bot.confirm_send_pending)
        first.start()
        self.assertTrue(entered.wait(2))
        self.bot.confirm_send_pending()
        release.set()
        first.join(2)
        self.assertFalse(first.is_alive())
        self.assertEqual(self.mod.keyboard.send.call_args_list.count(unittest.mock.call("enter")), 1)

    def test_old_message_context_is_not_rebound_to_new_chat(self):
        context = self.bot._capture_context()
        self.bot.set_person_slot(2)
        self.bot.generate_and_maybe_send("Alice's message", context)
        self.mod.openai_generate_reply.assert_not_called()
        self.assert_no_enter()

    def test_slot_change_during_message_ocr_does_not_write_new_slot_state(self):
        self.bot.cfg["session_routing"]["mode"] = "slot"
        self.bot.cfg["diff_gate"]["enabled"] = False
        self.bot.person_key = "slot1"
        self.bot.screenshot_region.side_effect = None
        def ocr(*args):
            self.bot.set_person_slot(2)
            return ["old chat message"]
        self.mod.ocr_lines_tesseract.side_effect = ocr
        self.bot.poll_once()
        self.assertNotIn("slot2", self.bot.state["last_text"])
        self.assertEqual(self.bot.prev_in_lines, [])
        self.mod.openai_generate_reply.assert_not_called()

    def test_title_change_during_message_ocr_does_not_record_or_generate(self):
        self.bot.cfg["diff_gate"]["enabled"] = False
        self.bot._title_candidate = "Alice"
        self.bot._title_candidate_count = 2
        self.bot.screenshot_region.side_effect = None
        def ocr(*args):
            self.bot._read_current_title.return_value = "Bob"
            return ["ambiguous chat message"]
        self.mod.ocr_lines_tesseract.side_effect = ocr
        self.bot.poll_once()
        self.assertNotIn("Alice", self.bot.state["last_text"])
        self.assertEqual(self.bot.prev_in_lines, [])
        self.mod.openai_generate_reply.assert_not_called()

    def test_manual_send_requires_arm_and_title(self):
        self.bot.armed = False
        self.bot.manual_send_clipboard()
        self.assert_no_enter()
        self.bot.armed = True
        self.bot._read_current_title.return_value = ""
        self.bot.manual_send_clipboard()
        self.assert_no_enter()

    def test_title_reader_preserves_punctuation_and_rejects_errors(self):
        self.bot.screenshot_region.side_effect = None
        self.mod.ocr_lines_tesseract.side_effect = None
        self.mod.ocr_lines_tesseract.return_value = ["Alice!"]
        read_title = self.mod.WeChatAutoBot._read_current_title
        self.assertEqual(read_title(self.bot), "Alice!")
        self.mod.ocr_lines_tesseract.side_effect = RuntimeError("mock OCR failure")
        self.assertEqual(read_title(self.bot), "")
        self.mod._pid_of.return_value = 456
        self.assertEqual(read_title(self.bot), "")

    def test_real_paste_validation_requires_exact_copy_not_containment(self):
        verify = self.mod.WeChatAutoBot.verify_paste_in_input
        for copied, expected in (("prefix test reply suffix", False), ("test reply", True), ("", False)):
            with self.subTest(copied=copied):
                self.mod.pyperclip.paste.side_effect = ["original clipboard", copied]
                self.assertEqual(verify(self.bot, "test reply", self.bot._capture_context()), expected)
                self.assertEqual(self.mod.pyperclip.copy.call_args, unittest.mock.call("original clipboard"))


if __name__ == "__main__":
    unittest.main()
