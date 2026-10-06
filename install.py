"""Install hook. Forge runs this in a separate Python process at startup."""

print("[SeedVR2] install.py starting")

try:
    import launch
except Exception as exc:
    print(f"[SeedVR2] install.py could not import launch: {exc}")
    raise

PACKAGE = "rotary-embedding-torch"

try:
    installed = launch.is_installed(PACKAGE)
except Exception as exc:
    print(f"[SeedVR2] install.py could not check {PACKAGE}: {exc}")
    raise

if installed:
    print(f"[SeedVR2] install.py: {PACKAGE} is already installed")
else:
    print(f"[SeedVR2] install.py: {PACKAGE} is missing, installing it")
    launch.run_pip(f"install {PACKAGE}", PACKAGE)
    print(f"[SeedVR2] install.py: finished install step for {PACKAGE}")

print("[SeedVR2] install.py finished")
