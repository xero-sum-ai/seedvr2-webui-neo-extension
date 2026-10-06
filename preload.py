"""Forge runs this before the WebUI UI is built.

A line here means the extension directory was found and is not disabled.
"""

import os
import sys
import traceback


def preload(parser):
    root = os.path.dirname(os.path.abspath(__file__))
    inserted = False
    if root not in sys.path:
        sys.path.insert(0, root)
        inserted = True
    try:
        import src.webui_diag as webui_diag
        webui_diag.note_preload(root)
    except Exception as exc:
        print(f"[SeedVR2] preload failed: {exc}")
        traceback.print_exc()
    finally:
        if inserted:
            try:
                sys.path.remove(root)
            except ValueError:
                pass
