# SeedVR2 for Stable Diffusion WebUI Forge Neo

SeedVR2 image upscaler for [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo).

The extension does nothing until you select it in the **Script** dropdown. After you restart Forge, a **SeedVR2 — extension status** panel on txt2img and img2img shows whether the extension loaded, which model folder it is using, and which weights it found. The same report is printed in the Forge console with the prefix `[SeedVR2]`.

![SeedVR2 script controls](https://github.com/user-attachments/assets/777c34e7-aca6-4e51-9994-f02f817311ea)

## Install

1. In Forge Neo, open **Extensions → Install from URL** and paste this repository URL, or clone it into `extensions/seedvr2-webui-neo-extension`.
2. Restart Forge. On startup, `install.py` installs `rotary-embedding-torch` if it is missing.
3. If that step fails, install it yourself in the Forge Python environment:

```
pip install rotary-embedding-torch
```

Confirm the install in the console. You should see these lines:

```
[SeedVR2] install.py starting
[SeedVR2] install.py finished
[SeedVR2] preload: Forge is loading this extension from ...
[SeedVR2] UI status
```

If none of those lines appear, Forge did not load the extension. In **Extensions → Installed**, confirm `seedvr2-webui-neo-extension` is present and enabled, then restart.

## Models

Put the weights in one of these folders:

- `extensions/seedvr2-webui-neo-extension/models/SeedVR2/`
- `<Forge Neo>/models/SeedVR2/`

The extension folder is used when it exists. Otherwise the WebUI `models/SeedVR2` folder is used.

You need both of these:

- VAE: `ema_vae_fp16.safetensors` (the filename must contain `vae`)
- A SeedVR2 DiT checkpoint, for example `seedvr2_ema_7b_sharp-Q4_K_M.gguf` (`.gguf` or `.safetensors`)

The extension does not download weights. A folder named `model/seedvr2` is ignored. Move files from that old path into `models/SeedVR2`.

Click **Refresh SeedVR2 status** after adding files. You do not need to restart unless the status panel itself is missing.

## Use

1. Restart Forge after installing the extension.
2. On txt2img or img2img, open the **Script** dropdown at the bottom of the page.
3. Choose **SeedVR2 Native Upscaler (24G)**.
4. Set **Upscale Resolution (Shortest Edge)**. The other controls can stay at their defaults.

What the script does:

- **txt2img**: generates the image first, then upscales that image with SeedVR2.
- **img2img**: skips the normal img2img pass and upscales the input image with SeedVR2.

The console prints `[SeedVR2] run started` when the script actually runs, then one line per phase (encode, upscale, decode, post-process). **Show Debug Logs** adds the extra lines (model name, seed, cache, unloading the SD checkpoint).

## Status panel

**SeedVR2 — extension status** is always visible. It does not upscale. It reports:

- the extension directory Forge loaded
- which model directory is in use, and whether it exists
- DiT and VAE files found in that directory
- whether `rotary-embedding-torch` and the other runtime dependencies import
- whether `scripts/seedvr2_script.py` imported

When the upscaler script fails to import, this panel still loads. The console line to search for is `Error loading script: seedvr2_script.py`.
