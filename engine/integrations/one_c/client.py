from __future__ import annotations

"""Legacy compatibility shim.

Use `app.integrations.one_c.client.OneCClient` in new code.
"""

from app.integrations.one_c.client import OneCClient

__all__ = ["OneCClient"]
