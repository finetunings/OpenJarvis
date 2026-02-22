"""``jarvis channel`` -- channel management commands."""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table


@click.group()
def channel() -> None:
    """Manage messaging channels."""


@channel.command("list")
@click.option("--gateway", default=None, help="OpenClaw gateway URL.")
def channel_list(gateway: str | None) -> None:
    """List available channels from the gateway."""
    console = Console()
    from openjarvis.core.config import load_config

    config = load_config()
    gw_url = gateway or config.channel.gateway_url

    from openjarvis.channels.openclaw_bridge import OpenClawChannelBridge

    bridge = OpenClawChannelBridge(gateway_url=gw_url)

    try:
        channels = bridge.list_channels()
    except Exception as exc:
        console.print(f"[red]Failed to list channels: {exc}[/red]")
        return

    if not channels:
        console.print(
            "[yellow]No channels available"
            " (is OpenClaw gateway running?)[/yellow]"
        )
        return

    table = Table(title="Available Channels")
    table.add_column("Channel", style="cyan")
    for ch in channels:
        table.add_row(ch)
    console.print(table)


@channel.command("send")
@click.argument("target")
@click.argument("message")
@click.option("--gateway", default=None, help="OpenClaw gateway URL.")
def channel_send(target: str, message: str, gateway: str | None) -> None:
    """Send a message to a channel."""
    console = Console()
    from openjarvis.core.config import load_config

    config = load_config()
    gw_url = gateway or config.channel.gateway_url

    from openjarvis.channels.openclaw_bridge import OpenClawChannelBridge

    bridge = OpenClawChannelBridge(gateway_url=gw_url)

    ok = bridge.send(target, message)
    if ok:
        console.print(f"[green]Message sent to {target}[/green]")
    else:
        console.print(f"[red]Failed to send message to {target}[/red]")


@channel.command("status")
@click.option("--gateway", default=None, help="OpenClaw gateway URL.")
def channel_status(gateway: str | None) -> None:
    """Show channel bridge connection status."""
    console = Console()
    from openjarvis.core.config import load_config

    config = load_config()
    gw_url = gateway or config.channel.gateway_url

    from openjarvis.channels.openclaw_bridge import OpenClawChannelBridge

    bridge = OpenClawChannelBridge(gateway_url=gw_url)

    st = bridge.status()
    color = {
        "connected": "green",
        "disconnected": "yellow",
        "connecting": "blue",
        "error": "red",
    }.get(st.value, "white")
    console.print(f"Gateway: [cyan]{gw_url}[/cyan]")
    console.print(f"Status: [{color}]{st.value}[/{color}]")
