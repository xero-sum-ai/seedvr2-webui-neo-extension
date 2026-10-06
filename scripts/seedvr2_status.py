"""Always-visible status panel. Does not upscale."""

import os
import sys

import gradio as gr
from modules import scripts, shared

EXTENSION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EXTENSION_ROOT not in sys.path:
    sys.path.insert(0, EXTENSION_ROOT)

import src.webui_diag as webui_diag


def _models_path():
    return getattr(shared, "models_path", None)


def _refresh_status():
    webui_diag.collect(EXTENSION_ROOT, _models_path())
    webui_diag.print_report("status refreshed from the WebUI")
    ready = webui_diag.is_ready()
    headline = "Ready to upscale." if ready else "Not ready. Read the log below."
    return f"**{headline}**\n\n{webui_diag.markdown()}"


class SeedVR2StatusScript(scripts.Script):
    sorting_priority = 20

    def title(self):
        return "SeedVR2 status"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        webui_diag.collect(EXTENSION_ROOT, _models_path())
        webui_diag.print_once("UI status")
        ready = webui_diag.is_ready()
        headline = "Ready to upscale." if ready else "Not ready. Read the log below."

        with gr.Accordion("SeedVR2 — extension status", open=True):
            status = gr.Markdown(f"**{headline}**\n\n{webui_diag.markdown()}")
            gr.Markdown(
                "This panel only reports whether the extension loaded. "
                "It does not upscale. Open the **Script** dropdown at the bottom of this page "
                "and choose **SeedVR2 Native Upscaler (24G)**. "
                "Turn on **Show Debug Logs** there for extra lines while an upscale is running."
            )
            refresh = gr.Button("Refresh SeedVR2 status")
            refresh.click(fn=_refresh_status, outputs=status)

        return []
