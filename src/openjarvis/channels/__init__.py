"""Channel abstraction for multi-platform messaging via OpenClaw."""

from openjarvis.channels._stubs import (
    BaseChannel,
    ChannelHandler,
    ChannelMessage,
    ChannelStatus,
)

# Trigger registration of built-in channels
try:
    from openjarvis.channels.openclaw_bridge import OpenClawChannelBridge  # noqa: F401
except ImportError:
    pass

__all__ = [
    "BaseChannel",
    "ChannelHandler",
    "ChannelMessage",
    "ChannelStatus",
]
