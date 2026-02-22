"""OpenClawChannelBridge — WebSocket/HTTP bridge to the OpenClaw gateway."""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Dict, List, Optional

from openjarvis.channels._stubs import (
    BaseChannel,
    ChannelHandler,
    ChannelMessage,
    ChannelStatus,
)
from openjarvis.core.events import EventBus, EventType
from openjarvis.core.registry import ChannelRegistry

logger = logging.getLogger(__name__)


@ChannelRegistry.register("openclaw")
class OpenClawChannelBridge(BaseChannel):
    """Bridge to the OpenClaw gateway via WebSocket with HTTP fallback.

    Parameters
    ----------
    gateway_url:
        WebSocket URL of the OpenClaw gateway (e.g. ``ws://127.0.0.1:18789/ws``).
    reconnect_interval:
        Seconds to wait before reconnecting after a disconnect.
    bus:
        Optional event bus for publishing channel events.
    """

    channel_id = "openclaw"

    def __init__(
        self,
        gateway_url: str = "ws://127.0.0.1:18789/ws",
        *,
        reconnect_interval: float = 5.0,
        bus: Optional[EventBus] = None,
    ) -> None:
        self._gateway_url = gateway_url
        self._reconnect_interval = reconnect_interval
        self._bus = bus
        self._handlers: List[ChannelHandler] = []
        self._status = ChannelStatus.DISCONNECTED
        self._ws: Any = None
        self._listener_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # -- connection lifecycle ---------------------------------------------------

    def connect(self) -> None:
        """Establish connection to the OpenClaw gateway."""
        self._status = ChannelStatus.CONNECTING
        self._stop_event.clear()

        try:
            import websockets.sync.client  # type: ignore[import-untyped]

            self._ws = websockets.sync.client.connect(self._gateway_url)
            self._status = ChannelStatus.CONNECTED
            self._listener_thread = threading.Thread(
                target=self._listener_loop, daemon=True,
            )
            self._listener_thread.start()
            logger.info(
                "Connected to OpenClaw gateway via WebSocket: %s",
                self._gateway_url,
            )
        except ImportError:
            # websockets not installed — use HTTP fallback mode
            logger.info(
                "websockets not installed; HTTP fallback: %s",
                self._gateway_url,
            )
            self._ws = None
            self._status = ChannelStatus.CONNECTED
        except Exception:
            logger.exception(
                "Failed to connect to OpenClaw gateway: %s",
                self._gateway_url,
            )
            self._status = ChannelStatus.ERROR

    def disconnect(self) -> None:
        """Close connection to the OpenClaw gateway."""
        self._stop_event.set()
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                logger.debug("Error closing WebSocket", exc_info=True)
            self._ws = None
        if self._listener_thread is not None:
            self._listener_thread.join(timeout=self._reconnect_interval + 1)
            self._listener_thread = None
        self._status = ChannelStatus.DISCONNECTED

    # -- send / receive --------------------------------------------------------

    def send(
        self,
        channel: str,
        content: str,
        *,
        conversation_id: str = "",
        metadata: Dict[str, Any] | None = None,
    ) -> bool:
        """Send a message to a channel. Returns True on success."""
        payload = {
            "channel": channel,
            "content": content,
            "conversation_id": conversation_id,
            "metadata": metadata or {},
        }

        # Try WebSocket first
        if self._ws is not None:
            try:
                self._ws.send(json.dumps(payload))
                self._publish_sent(channel, content, conversation_id)
                return True
            except Exception:
                logger.debug(
                    "WebSocket send failed, falling back to HTTP",
                    exc_info=True,
                )

        # HTTP fallback
        return self._send_http(payload, channel, content, conversation_id)

    def _send_http(
        self,
        payload: Dict[str, Any],
        channel: str,
        content: str,
        conversation_id: str,
    ) -> bool:
        """Send message via HTTP POST fallback."""
        http_url = self._http_url("/send")
        try:
            import httpx

            resp = httpx.post(http_url, json=payload, timeout=10.0)
            if resp.status_code < 300:
                self._publish_sent(channel, content, conversation_id)
                return True
            logger.warning("HTTP send returned status %d", resp.status_code)
            return False
        except Exception:
            logger.debug("HTTP send failed", exc_info=True)
            return False

    def status(self) -> ChannelStatus:
        """Return the current connection status."""
        return self._status

    def list_channels(self) -> List[str]:
        """Query the gateway for available channels."""
        http_url = self._http_url("/channels")
        try:
            import httpx

            resp = httpx.get(http_url, timeout=10.0)
            data = resp.json()
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "channels" in data:
                return data["channels"]
            return []
        except Exception:
            logger.debug("Failed to list channels", exc_info=True)
            return []

    def on_message(self, handler: ChannelHandler) -> None:
        """Register a callback for incoming messages."""
        self._handlers.append(handler)

    # -- internal helpers -------------------------------------------------------

    def _listener_loop(self) -> None:
        """Background loop receiving messages from WebSocket."""
        while not self._stop_event.is_set():
            try:
                if self._ws is None:
                    break
                raw = self._ws.recv(timeout=1.0)
                data = json.loads(raw)
                msg = ChannelMessage(
                    channel=data.get("channel", ""),
                    sender=data.get("sender", ""),
                    content=data.get("content", ""),
                    message_id=data.get("message_id", ""),
                    conversation_id=data.get("conversation_id", ""),
                    metadata=data.get("metadata", {}),
                )
                for handler in self._handlers:
                    try:
                        handler(msg)
                    except Exception:
                        logger.exception("Channel handler error")
                if self._bus is not None:
                    self._bus.publish(
                        EventType.CHANNEL_MESSAGE_RECEIVED,
                        {
                            "channel": msg.channel,
                            "sender": msg.sender,
                            "content": msg.content,
                            "message_id": msg.message_id,
                        },
                    )
            except TimeoutError:
                continue
            except Exception:
                if self._stop_event.is_set():
                    break
                logger.debug("WebSocket listener error, reconnecting", exc_info=True)
                self._status = ChannelStatus.CONNECTING
                self._stop_event.wait(self._reconnect_interval)
                if self._stop_event.is_set():
                    break
                try:
                    import websockets.sync.client  # type: ignore[import-untyped]

                    self._ws = websockets.sync.client.connect(self._gateway_url)
                    self._status = ChannelStatus.CONNECTED
                except Exception:
                    logger.debug("Reconnect failed", exc_info=True)
                    self._status = ChannelStatus.ERROR

    def _http_url(self, path: str = "") -> str:
        """Convert the WebSocket URL to an HTTP URL."""
        url = self._gateway_url
        url = url.replace("ws://", "http://").replace("wss://", "https://")
        # Strip trailing /ws
        if url.endswith("/ws"):
            url = url[:-3]
        return url + path

    def _publish_sent(self, channel: str, content: str, conversation_id: str) -> None:
        """Publish a CHANNEL_MESSAGE_SENT event on the bus."""
        if self._bus is not None:
            self._bus.publish(
                EventType.CHANNEL_MESSAGE_SENT,
                {
                    "channel": channel,
                    "content": content,
                    "conversation_id": conversation_id,
                },
            )


__all__ = ["OpenClawChannelBridge"]
