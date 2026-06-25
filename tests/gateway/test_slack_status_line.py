"""Tests for Slack status-line progress wiring primitives."""

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


def _ensure_slack_mock():
    """Install mock slack modules so SlackAdapter can be imported."""
    if "slack_bolt" in sys.modules and hasattr(sys.modules["slack_bolt"], "__file__"):
        return

    slack_bolt = MagicMock()
    slack_bolt.async_app.AsyncApp = MagicMock
    slack_bolt.adapter.socket_mode.async_handler.AsyncSocketModeHandler = MagicMock

    slack_sdk = MagicMock()
    slack_sdk.web.async_client.AsyncWebClient = MagicMock

    for name, mod in [
        ("slack_bolt", slack_bolt),
        ("slack_bolt.async_app", slack_bolt.async_app),
        ("slack_bolt.adapter", slack_bolt.adapter),
        ("slack_bolt.adapter.socket_mode", slack_bolt.adapter.socket_mode),
        (
            "slack_bolt.adapter.socket_mode.async_handler",
            slack_bolt.adapter.socket_mode.async_handler,
        ),
        ("slack_sdk", slack_sdk),
        ("slack_sdk.web", slack_sdk.web),
        ("slack_sdk.web.async_client", slack_sdk.web.async_client),
    ]:
        sys.modules.setdefault(name, mod)

    sys.modules.setdefault("aiohttp", MagicMock())


_ensure_slack_mock()

import plugins.platforms.slack.adapter as _slack_mod

_slack_mod.SLACK_AVAILABLE = True

from gateway.display_config import resolve_display_setting  # noqa: E402
from plugins.platforms.slack.adapter import SlackAdapter  # noqa: E402


@pytest.fixture()
def adapter():
    config = PlatformConfig(enabled=True, token="xoxb-fake-token")
    instance = SlackAdapter(config)
    instance._app = MagicMock()
    instance._app.client = AsyncMock()
    return instance


class TestSlackStatusLineAdapter:
    @pytest.mark.asyncio
    async def test_send_typing_uses_custom_status(self, adapter):
        adapter._app.client.assistant_threads_setStatus = AsyncMock()

        await adapter.send_typing(
            "C123",
            metadata={"thread_id": "ts1"},
            status="looking that up…",
        )

        adapter._app.client.assistant_threads_setStatus.assert_called_once_with(
            channel_id="C123",
            thread_ts="ts1",
            status="looking that up…",
        )

    @pytest.mark.asyncio
    async def test_send_typing_defaults_to_legacy_status(self, adapter):
        adapter._app.client.assistant_threads_setStatus = AsyncMock()

        await adapter.send_typing("C123", metadata={"thread_id": "ts1"})

        adapter._app.client.assistant_threads_setStatus.assert_called_once_with(
            channel_id="C123",
            thread_ts="ts1",
            status="is thinking...",
        )

    @pytest.mark.asyncio
    async def test_update_thread_status_alias_uses_safe_label(self, adapter):
        adapter._app.client.assistant_threads_setStatus = AsyncMock()

        await adapter.update_thread_status(
            "C123",
            "drafting a reply…",
            metadata={"thread_id": "ts1"},
        )

        adapter._app.client.assistant_threads_setStatus.assert_called_once_with(
            channel_id="C123",
            thread_ts="ts1",
            status="drafting a reply…",
        )

    def test_supports_status_line_capability_is_enabled(self):
        assert SlackAdapter.supports_status_line is True

    @pytest.mark.asyncio
    async def test_update_thread_status_falls_back_to_active_thread(self, adapter):
        # Top-level @mention case: no thread reference in metadata, but the
        # turn-start "is thinking..." already established the thread, so the
        # update must still target it instead of silently no-opping.
        adapter._app.client.assistant_threads_setStatus = AsyncMock()
        adapter._active_status_threads["C123"] = "ts-established"

        await adapter.update_thread_status("C123", "posting…", metadata=None)

        adapter._app.client.assistant_threads_setStatus.assert_called_once_with(
            channel_id="C123",
            thread_ts="ts-established",
            status="posting…",
        )

    @pytest.mark.asyncio
    async def test_update_thread_status_noops_without_any_thread(self, adapter):
        # No metadata and no established thread → nothing to target; must not
        # call setStatus (and must not raise).
        adapter._app.client.assistant_threads_setStatus = AsyncMock()

        await adapter.update_thread_status("C999", "posting…", metadata=None)

        adapter._app.client.assistant_threads_setStatus.assert_not_called()


class TestSlackStatusLineSelection:
    def test_status_label_mapping_prefers_safe_configured_phrase(self):
        status_labels = {"search_board_memory": "looking that up…"}
        status_line_default = "working on it…"

        selected_label = status_labels.get("search_board_memory") or status_line_default

        assert selected_label == "looking that up…"

    def test_status_label_mapping_falls_back_to_generic_default(self):
        status_labels = {"search_board_memory": "looking that up…"}
        status_line_default = "working on it…"

        selected_label = status_labels.get("send_message") or status_line_default

        assert selected_label == "working on it…"
        assert "send_message" not in selected_label

    def test_status_line_progress_false_is_normalised_to_off(self):
        config = {"display": {"platforms": {"slack": {"status_line_progress": False}}}}

        assert resolve_display_setting(config, "slack", "status_line_progress") is False
