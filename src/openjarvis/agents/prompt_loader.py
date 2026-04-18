"""Load system prompt overrides from $OPENJARVIS_HOME/agents/{name}/system_prompt.md.

Distillation (M1) proposes system prompt edits that get written to disk by
``ReplaceSystemPromptApplier``.  This module lets agents pick those overrides
up at runtime, falling back to their hardcoded default when no file exists.

Override files are templates — they may contain ``{tool_descriptions}`` and
other format placeholders that the agent fills in via ``.format()``, exactly
like the hardcoded constants.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def load_system_prompt_override(agent_name: str) -> str | None:
    """Return the override prompt for *agent_name*, or ``None``.

    Looks for ``$OPENJARVIS_HOME/agents/<agent_name>/system_prompt.md``.
    ``OPENJARVIS_HOME`` defaults to ``~/.openjarvis`` when unset.
    """
    home = Path(os.environ.get("OPENJARVIS_HOME", "~/.openjarvis")).expanduser()
    prompt_path = home / "agents" / agent_name / "system_prompt.md"
    if not prompt_path.exists():
        return None
    try:
        content = prompt_path.read_text(encoding="utf-8")
        logger.info(
            "Loaded system prompt override for %s from %s", agent_name, prompt_path
        )
        return content
    except Exception:
        logger.warning(
            "Failed to read system prompt override at %s", prompt_path, exc_info=True
        )
        return None
