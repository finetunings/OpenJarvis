"""Tests for Phase 3 config additions (AgentConfig expansion, ServerConfig)."""

from __future__ import annotations

from openjarvis.core.config import (
    AgentConfig,
    HardwareInfo,
    JarvisConfig,
    ServerConfig,
    generate_default_toml,
)


class TestAgentConfig:
    def test_defaults(self):
        cfg = AgentConfig()
        assert cfg.default_agent == "simple"
        assert cfg.max_turns == 3
        assert cfg.default_tools == ""
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 1024

    def test_custom_values(self):
        cfg = AgentConfig(
            default_agent="orchestrator",
            max_turns=5,
            default_tools="calculator,think",
            temperature=0.1,
            max_tokens=512,
        )
        assert cfg.default_agent == "orchestrator"
        assert cfg.default_tools == "calculator,think"


class TestServerConfig:
    def test_defaults(self):
        cfg = ServerConfig()
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 8000
        assert cfg.agent == "orchestrator"
        assert cfg.model == ""
        assert cfg.workers == 1

    def test_custom_values(self):
        cfg = ServerConfig(host="127.0.0.1", port=9000, agent="simple")
        assert cfg.host == "127.0.0.1"
        assert cfg.port == 9000


class TestJarvisConfig:
    def test_has_server(self):
        cfg = JarvisConfig()
        assert hasattr(cfg, "server")
        assert isinstance(cfg.server, ServerConfig)

    def test_agent_config_expanded(self):
        cfg = JarvisConfig()
        assert hasattr(cfg.agent, "default_tools")
        assert hasattr(cfg.agent, "temperature")
        assert hasattr(cfg.agent, "max_tokens")


class TestGenerateDefaultToml:
    def test_includes_server_section(self):
        hw = HardwareInfo(cpu_brand="Test CPU", cpu_count=4, ram_gb=16.0)
        toml_str = generate_default_toml(hw)
        assert "[server]" in toml_str
        assert "port = 8000" in toml_str

    def test_includes_agent_section(self):
        hw = HardwareInfo()
        toml_str = generate_default_toml(hw)
        assert "[agent]" in toml_str
        assert "default_agent" in toml_str
