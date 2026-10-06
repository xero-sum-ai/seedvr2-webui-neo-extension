"""Startup diagnostics for the Forge Neo extension.

Kept free of torch and Gradio so preload.py can import it before the UI exists.
"""

import importlib.metadata
import importlib.util
import os
import sys

PREFIX = "[SeedVR2]"

_state = {
    "lines": [],
    "extension_root": None,
    "model_dir": None,
    "model_dir_exists": False,
    "dit_files": [],
    "vae_files": [],
    "script_loaded": False,
    "script_error": None,
    "rotary_ok": False,
    "ui_printed": False,
}


def resolve_model_dir(extension_root, webui_models_path=None):
    """Return the directory the upscaler will actually read.

    The extension folder is checked first. The WebUI models folder is the
    fallback. The old README path ``model/seedvr2`` is not used.
    """
    primary = os.path.join(extension_root, "models", "SeedVR2")
    if os.path.isdir(primary):
        return primary
    if webui_models_path:
        fallback = os.path.join(webui_models_path, "SeedVR2")
        if os.path.isdir(fallback):
            return fallback
    return primary


def mark_script_loaded():
    _state["script_loaded"] = True
    _state["script_error"] = None


def mark_script_error(message):
    _state["script_loaded"] = False
    _state["script_error"] = message


def _add(line):
    _state["lines"].append(line)


def _import_status(dist_name, import_name):
    version = None
    try:
        version = importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        version = None
    except Exception as exc:
        return False, f"version check failed: {exc}"

    try:
        spec = importlib.util.find_spec(import_name)
    except Exception as exc:
        return False, f"import check failed: {exc}"

    if spec is None and version is None:
        return False, "not installed"
    if spec is None:
        return False, f"package {version} is installed but cannot be imported"
    if version is None:
        return True, "importable (version unknown)"
    return True, version


def collect(extension_root, webui_models_path=None):
    """Rebuild the status lines. Safe to call more than once."""
    script_loaded = _state["script_loaded"]
    script_error = _state["script_error"]
    ui_printed = _state["ui_printed"]

    _state.clear()
    _state.update({
        "lines": [],
        "extension_root": extension_root,
        "model_dir": None,
        "model_dir_exists": False,
        "dit_files": [],
        "vae_files": [],
        "script_loaded": script_loaded,
        "script_error": script_error,
        "rotary_ok": False,
        "ui_printed": ui_printed,
    })

    primary = os.path.join(extension_root, "models", "SeedVR2")
    fallback = os.path.join(webui_models_path, "SeedVR2") if webui_models_path else None
    legacy = os.path.join(extension_root, "model", "seedvr2")
    model_dir = resolve_model_dir(extension_root, webui_models_path)

    _state["model_dir"] = model_dir
    _state["model_dir_exists"] = os.path.isdir(model_dir)

    _add(f"Extension root: {extension_root}")
    _add(f"Python: {sys.version.split()[0]}")
    _add(f"Model directory in use: {model_dir}")
    _add(f"Model directory exists: {'yes' if os.path.isdir(model_dir) else 'no'}")

    for label, path in (
        ("extension models/SeedVR2", primary),
        ("WebUI models/SeedVR2", fallback),
    ):
        if not path:
            _add(f"{label}: WebUI models path is not available yet")
            continue
        _add(f"{label}: {'found' if os.path.isdir(path) else 'missing'} — {path}")

    if os.path.isdir(legacy):
        _add(
            "Old path model/seedvr2 exists. Move those files to models/SeedVR2. "
            f"This extension does not load {legacy}."
        )

    weight_files = []
    if os.path.isdir(model_dir):
        try:
            weight_files = sorted(
                name for name in os.listdir(model_dir)
                if name.endswith((".gguf", ".safetensors", ".pt", ".pth"))
            )
        except OSError as exc:
            _add(f"Could not list the model directory: {exc}")

    dit_files = [name for name in weight_files if "vae" not in name.lower()]
    vae_files = [name for name in weight_files if "vae" in name.lower()]
    _state["dit_files"] = dit_files
    _state["vae_files"] = vae_files
    _add("DiT checkpoints: " + (", ".join(dit_files) if dit_files else "none"))
    _add("VAE checkpoints: " + (", ".join(vae_files) if vae_files else "none"))

    checks = (
        ("rotary-embedding-torch", "rotary_embedding_torch"),
        ("torch", "torch"),
        ("safetensors", "safetensors"),
        ("gguf", "gguf"),
        ("einops", "einops"),
        ("omegaconf", "omegaconf"),
    )
    for dist_name, import_name in checks:
        ok, detail = _import_status(dist_name, import_name)
        if import_name == "rotary_embedding_torch":
            _state["rotary_ok"] = ok
        _add(f"Dependency {dist_name}: {'ok' if ok else 'MISSING'} ({detail})")

    if script_loaded:
        _add("Upscaler script: imported")
    elif script_error:
        _add("Upscaler script: FAILED")
        _add(script_error)
    else:
        _add(
            "Upscaler script: not imported. If the WebUI is already up, "
            "search the console for 'Error loading script: seedvr2_script.py'."
        )

    if not _state["model_dir_exists"] or not dit_files or not vae_files:
        _add(
            "Weights are not ready. Put ema_vae_fp16.safetensors and a SeedVR2 "
            "DiT file (.gguf or .safetensors) in the model directory above."
        )

    _add(
        "To upscale, open the Script dropdown at the bottom of txt2img or img2img "
        "and choose 'SeedVR2 Native Upscaler (24G)'. This status panel does not upscale."
    )
    return _state


def is_ready():
    return bool(
        _state["script_loaded"]
        and _state["model_dir_exists"]
        and _state["dit_files"]
        and _state["vae_files"]
        and _state["rotary_ok"]
    )


def text():
    if not _state["lines"]:
        return "(no status collected yet)"
    return "\n".join(_state["lines"])


def markdown():
    return "```\n" + text() + "\n```"


def print_report(header):
    print(f"{PREFIX} {header}")
    for line in _state["lines"]:
        print(f"{PREFIX} {line}")
    if not _state["lines"]:
        print(f"{PREFIX} (no status collected yet)")


def print_once(header):
    if _state["ui_printed"]:
        return
    _state["ui_printed"] = True
    print_report(header)


def note_preload(extension_root):
    print(f"{PREFIX} preload: Forge is loading this extension from {extension_root}")
    collect(extension_root, None)
    print_report("preload status (the WebUI model path is checked again when the UI starts)")
