"""Tests for channel configuration."""

from __future__ import annotations

import textwrap
from pathlib import Path

from openjarvis.core.config import ChannelConfig, JarvisConfig, load_config


class TestChannelConfigDefaults:
    def test_channel_config_defaults(self) -> None:
        cfg = ChannelConfig()
        assert cfg.enabled is False
        assert cfg.gateway_url == "ws://127.0.0.1:18789/ws"
        assert cfg.default_agent == "simple"
        assert cfg.reconnect_interval == 5.0

    def test_channel_config_custom(self) -> None:
        cfg = ChannelConfig(
            enabled=True,
            gateway_url="ws://custom:9999/ws",
            default_agent="orchestrator",
            reconnect_interval=10.0,
        )
        assert cfg.enabled is True
        assert cfg.gateway_url == "ws://custom:9999/ws"
        assert cfg.default_agent == "orchestrator"
        assert cfg.reconnect_interval == 10.0


class TestChannelConfigInJarvisConfig:
    def test_channel_config_in_jarvis_config(self) -> None:
        cfg = JarvisConfig()
        assert hasattr(cfg, "channel")
        assert isinstance(cfg.channel, ChannelConfig)

    def test_jarvis_config_channel_defaults(self) -> None:
        cfg = JarvisConfig()
        assert cfg.channel.enabled is False
        assert cfg.channel.gateway_url == "ws://127.0.0.1:18789/ws"


class TestLoadConfigWithChannel:
    def test_load_config_with_channel_section(self, tmp_path: Path) -> None:
        """Create a temp TOML with [channel] section and verify values."""
        toml_content = textwrap.dedent("""\
            [channel]
            enabled = true
            gateway_url = "ws://my-gateway:7777/ws"
            default_agent = "orchestrator"
            reconnect_interval = 15.0
        """)
        config_file = tmp_path / "config.toml"
        config_file.write_text(toml_content)

        cfg = load_config(path=config_file)
        assert cfg.channel.enabled is True
        assert cfg.channel.gateway_url == "ws://my-gateway:7777/ws"
        assert cfg.channel.default_agent == "orchestrator"
        assert cfg.channel.reconnect_interval == 15.0

    def test_load_config_without_channel_section(self, tmp_path: Path) -> None:
        """When no [channel] section, defaults should apply."""
        toml_content = textwrap.dedent("""\
            [engine]
            default = "ollama"
        """)
        config_file = tmp_path / "config.toml"
        config_file.write_text(toml_content)

        cfg = load_config(path=config_file)
        assert cfg.channel.enabled is False
        assert cfg.channel.gateway_url == "ws://127.0.0.1:18789/ws"

    def test_load_config_partial_channel_section(self, tmp_path: Path) -> None:
        """Partial [channel] section overlays only specified fields."""
        toml_content = textwrap.dedent("""\
            [channel]
            enabled = true
        """)
        config_file = tmp_path / "config.toml"
        config_file.write_text(toml_content)

        cfg = load_config(path=config_file)
        assert cfg.channel.enabled is True
        # Non-specified fields keep defaults
        assert cfg.channel.gateway_url == "ws://127.0.0.1:18789/ws"
        assert cfg.channel.default_agent == "simple"
        assert cfg.channel.reconnect_interval == 5.0
