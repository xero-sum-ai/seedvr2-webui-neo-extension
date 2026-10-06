import os
import sys
import gradio as gr
from modules import scripts, processing, images, shared
from modules.processing import Processed, create_infotext, StableDiffusionProcessingTxt2Img
from torchvision import transforms
import torch
import numpy as np
from PIL import Image
import random

# SD checkpoint unload, used to free VRAM before SeedVR2 loads.
import modules.sd_models

EXTENSION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EXTENSION_ROOT not in sys.path:
    sys.path.insert(0, EXTENSION_ROOT)

import src.webui_diag as webui_diag

# Extension models/SeedVR2, then the WebUI models/SeedVR2 folder.
MODEL_DIR = webui_diag.resolve_model_dir(EXTENSION_ROOT, getattr(shared, "models_path", None))


class SeedVR2Interrupted(Exception):
    """Raised when the user stops generation mid-upscale."""
    pass

class Script(scripts.Script):
    def title(self):
        return "SeedVR2 Native Upscaler (24G)"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        webui_diag.collect(EXTENSION_ROOT, getattr(shared, "models_path", None))
        webui_diag.print_once("UI status")
        gr.Markdown(webui_diag.markdown())

        all_files = []
        if os.path.exists(MODEL_DIR):
            all_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".gguf") or f.endswith(".safetensors")]
        
        dit_choices = [f for f in all_files if "vae" not in f.lower()] 
        vae_choices = [f for f in all_files if "vae" in f.lower()]     
        
        if not dit_choices: dit_choices = all_files
        if not vae_choices: vae_choices = all_files

        default_dit = dit_choices[0] if dit_choices else ""
        default_vae = next((f for f in vae_choices if "ema_vae_fp16" in f), vae_choices[0] if vae_choices else "")

        with gr.Row():
            dit_model = gr.Dropdown(label="DiT Model", choices=dit_choices, value=default_dit)
            vae_model = gr.Dropdown(label="VAE Model", choices=vae_choices, value=default_vae)
        
        with gr.Row():
            seed = gr.Number(label="Seed (-1 = Random/Follow)", value=-1, precision=0)
            resolution = gr.Slider(label="Upscale Resolution (Shortest Edge)", minimum=512, maximum=3840, step=64, value=1080)
        
        with gr.Row():
            unload_sd = gr.Checkbox(label="Unload SD Checkpoint", value=True, elem_classes="force-unload")
            save_original = gr.Checkbox(label="Save Extra Copy of Original", value=False)
            force_reload = gr.Checkbox(label="Force Reload", value=False)
            debug_mode = gr.Checkbox(label="Show Debug Logs", value=False, info="Extra console lines while an upscale is running. Startup status is always printed.")

        with gr.Accordion("Advanced Settings (Noise & Tiling)", open=False):
            with gr.Row():
                input_noise = gr.Slider(label="Input Noise Scale", minimum=0.0, maximum=1.0, step=0.05, value=0.0)
                latent_noise = gr.Slider(label="Latent Noise Scale", minimum=0.0, maximum=1.0, step=0.05, value=0.0)
            
            gr.Markdown("ℹ️ **Tiling reduces VRAM usage for huge resolutions.**")
            with gr.Row():
                use_tile_vae = gr.Checkbox(label="Enable VAE Tiling", value=True)
                tile_size = gr.Slider(label="Tile Size", minimum=512, maximum=2048, step=128, value=1024)
                tile_overlap = gr.Slider(label="Tile Overlap", minimum=64, maximum=512, step=32, value=128)

        return [dit_model, vae_model, seed, resolution, input_noise, latent_noise, force_reload, unload_sd, save_original, use_tile_vae, tile_size, tile_overlap, debug_mode]

    def run(self, p, dit_model_name, vae_model_name, seed, resolution, input_noise, latent_noise, force_reload, unload_sd, save_original, use_tile_vae, tile_size, tile_overlap, debug_mode):
        if EXTENSION_ROOT not in sys.path:
            sys.path.insert(0, EXTENSION_ROOT)

        tab = "txt2img" if isinstance(p, StableDiffusionProcessingTxt2Img) else "img2img"
        print(f"[SeedVR2] run started on {tab}. dir={MODEL_DIR} dit={dit_model_name!r} vae={vae_model_name!r} debug={bool(debug_mode)}")

        if not dit_model_name or not vae_model_name:
            message = f"[SeedVR2] missing weights. DiT={dit_model_name!r} VAE={vae_model_name!r} dir={MODEL_DIR}"
            print(message)
            return Processed(p, [], p.seed, message)

        def check_interruption(*args, **kwargs):
            # Only shared.state.interrupted is reliable. .stopping was removed.
            if shared.state.interrupted:
                raise SeedVR2Interrupted("User interrupted generation.")

        # txt2img: generate the base image first, then upscale it.
        if isinstance(p, StableDiffusionProcessingTxt2Img):
            print("[SeedVR2] Generating base image (txt2img)...")
            
            current_script = None
            if p.scripts is not None:
                for s in p.scripts.scripts:
                    if s.title() == self.title():
                        current_script = s
                        break
                if current_script:
                    p.scripts.scripts.remove(current_script)
            
            try:
                processed_base = processing.process_images(p)
            finally:
                if current_script and p.scripts:
                    p.scripts.scripts.append(current_script)
            
            if shared.state.interrupted:
                print("[SeedVR2] Interrupted during txt2img generation. Stopping SeedVR2.")
                return processed_base

            input_img = processed_base.images[0]
            
            if save_original:
                 images.save_image(input_img, p.outpath_samples, "", processed_base.seed, p.prompt, shared.opts.samples_format, info=processed_base.info, p=p, suffix="-original")
            
        else:
            if shared.state.interrupted:
                 print("[SeedVR2] Interrupted before img2img upscale.")
                 return Processed(p, [], p.seed, "Interrupted")
            if not getattr(p, "init_images", None):
                message = "[SeedVR2] img2img has no input image."
                print(message)
                return Processed(p, [], p.seed, message)
            input_img = p.init_images[0]

        if unload_sd:
            if debug_mode: print("[SeedVR2] Unloading SD models...")
            modules.sd_models.unload_model_weights()
            import modules.devices as devices
            devices.torch_gc()

        if seed == -1:
            if hasattr(p, 'all_seeds') and p.all_seeds is not None and len(p.all_seeds) > 0 and p.all_seeds[0] != -1:
                actual_seed = int(p.all_seeds[0])
            elif p.seed is not None and p.seed != -1:
                actual_seed = int(p.seed)
            else:
                actual_seed = random.randint(0, 2147483647)
        else:
            actual_seed = int(seed)

        if debug_mode:
            print(f"[SeedVR2] Active. DiT: {dit_model_name} | Seed: {actual_seed}")
        
        try:
            import src.utils.debug as debug_module
            from src.core.generation_utils import setup_generation_context, prepare_runner, compute_generation_info, load_text_embeddings, script_directory
            from src.core.generation_phases import encode_all_batches, upscale_all_batches, decode_all_batches, postprocess_all_batches
            from src.utils.downloads import download_weight
            
            import argparse
            args = argparse.Namespace()
            args.dit_model = dit_model_name
            args.vae_model = vae_model_name 
            args.seed = int(actual_seed)
            args.resolution = int(resolution)
            args.input_noise_scale = float(input_noise)
            args.latent_noise_scale = float(latent_noise)
            
            args.model_dir = MODEL_DIR
            args.dit_offload_device = "none"
            args.vae_offload_device = "none"
            args.tensor_offload_device = "cpu"
            args.blocks_to_swap = 0
            args.swap_io_components = False
            args.cache_dit = True
            args.cache_vae = True
            
            args.vae_encode_tiled = use_tile_vae
            args.vae_encode_tile_size = int(tile_size)
            args.vae_encode_tile_overlap = int(tile_overlap)
            args.vae_decode_tiled = use_tile_vae
            args.vae_decode_tile_size = int(tile_size)
            args.vae_decode_tile_overlap = int(tile_overlap)

            args.max_resolution = 0
            args.batch_size = 1
            args.uniform_batch_size = False
            args.prepend_frames = 0
            args.temporal_overlap = 0
            args.color_correction = "lab"
            args.tile_debug = "false"
            args.attention_mode = "sdpa"
            args.compile_dit = False
            args.compile_vae = False

        except ImportError as e:
            print(f"[SeedVR2] import failed: {e}")
            return Processed(p, [], p.seed, f"Error: {e}")

        img_tensor = transforms.ToTensor()(input_img).unsqueeze(0).permute(0, 2, 3, 1).to(dtype=torch.float16)

        if not hasattr(self, 'runner_cache'):
            self.runner_cache = {}
            
        if force_reload:
            self.runner_cache.clear()
            if debug_mode: print("[SeedVR2] Cache cleared.")
        
        ctx = None 

        try:
            debug = debug_module.Debug(enabled=debug_mode)
            device_id = "0"
            inference_device = f"cuda:{device_id}"

            check_interruption()

            if 'ctx' in self.runner_cache:
                ctx = self.runner_cache['ctx']
            else:
                ctx = setup_generation_context(
                    dit_device=inference_device,
                    vae_device=inference_device,
                    dit_offload_device=None,
                    vae_offload_device=None,
                    tensor_offload_device="cpu",
                    debug=debug
                )
                self.runner_cache['ctx'] = ctx

            runner, cache_context = prepare_runner(
                dit_model=args.dit_model,
                vae_model=args.vae_model, 
                model_dir=args.model_dir,
                debug=debug,
                ctx=ctx,
                dit_cache=True,
                vae_cache=True,
                dit_id="forge_dit",
                vae_id="forge_vae",
                block_swap_config={'blocks_to_swap': 0, 'swap_io_components': False, 'offload_device': None},
                encode_tiled=args.vae_encode_tiled, 
                encode_tile_size=(args.vae_encode_tile_size, args.vae_encode_tile_size), 
                encode_tile_overlap=(args.vae_encode_tile_overlap, args.vae_encode_tile_overlap),
                decode_tiled=args.vae_decode_tiled, 
                decode_tile_size=(args.vae_decode_tile_size, args.vae_decode_tile_size), 
                decode_tile_overlap=(args.vae_decode_tile_overlap, args.vae_decode_tile_overlap),
                tile_debug="false", attention_mode="sdpa",
                torch_compile_args_dit=None, torch_compile_args_vae=None
            )
            
            ctx['cache_context'] = cache_context 
            
            if 'text_embeds' not in ctx:
                 ctx['text_embeds'] = load_text_embeddings(script_directory, ctx['dit_device'], ctx['compute_dtype'], debug)

            frames_tensor, gen_info = compute_generation_info(
                ctx=ctx, images=img_tensor, resolution=args.resolution, max_resolution=0,
                batch_size=1, uniform_batch_size=False, seed=args.seed,
                prepend_frames=0, temporal_overlap=0, debug=debug
            )
            
            print(f"[SeedVR2] Phase 1/4: Encoding {'(Tiled)' if use_tile_vae else ''}...")
            ctx = encode_all_batches(runner, ctx=ctx, images=frames_tensor, debug=debug, batch_size=1, uniform_batch_size=False, seed=args.seed, 
                                     progress_callback=check_interruption, 
                                     temporal_overlap=0, resolution=args.resolution, max_resolution=0, input_noise_scale=args.input_noise_scale, color_correction="lab")
            
            print("[SeedVR2] Phase 2/4: Upscaling...")
            ctx = upscale_all_batches(runner, ctx=ctx, debug=debug, 
                                      progress_callback=check_interruption, 
                                      seed=args.seed, latent_noise_scale=args.latent_noise_scale, cache_model=True)
            
            print(f"[SeedVR2] Phase 3/4: Decoding {'(Tiled)' if use_tile_vae else ''}...")
            ctx = decode_all_batches(runner, ctx=ctx, debug=debug, 
                                     progress_callback=check_interruption, 
                                     cache_model=True)
            
            print("[SeedVR2] Phase 4/4: Post-processing...")
            ctx = postprocess_all_batches(ctx=ctx, debug=debug, 
                                          progress_callback=check_interruption, 
                                          color_correction="lab", prepend_frames=0, temporal_overlap=0, batch_size=1)

            output_tensor = ctx['final_video']
            tensor_f32 = output_tensor[0].cpu().to(dtype=torch.float32)
            tensor_f32 = torch.clamp(tensor_f32, 0, 1)
            
            out_np = (tensor_f32.numpy() * 255.0).astype(np.uint8)
            out_pil = Image.fromarray(out_np)

            p.extra_generation_params["SeedVR2 Model"] = dit_model_name
            p.extra_generation_params["SeedVR2 Resolution"] = resolution
            
            try:
                infotext = create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, index=0)
            except:
                infotext = ""

            images.save_image(
                out_pil, 
                p.outpath_samples, 
                "", 
                args.seed, 
                p.prompt, 
                shared.opts.samples_format, 
                info=infotext,
                p=p,
                suffix="-seedvr2"
            )

            print("[SeedVR2] Done.")
            return Processed(p, [out_pil], args.seed, f"SeedVR2 Upscaled")

        except SeedVR2Interrupted:
            print("\n⏹️ [SeedVR2] Process Interrupted by User.")
            return Processed(p, [input_img], p.seed, "Interrupted")

        except Exception as e:
            print(f"❌ [SeedVR2 Runtime Error]: {e}")
            import traceback
            traceback.print_exc()
            return Processed(p, [], p.seed, f"Runtime Error: {e}")
        
        finally:
            if ctx is not None:
                for k in ['images', 'encoded_latents', 'upscaled_latents', 'final_video']:
                    if k in ctx: del ctx[k]


webui_diag.mark_script_loaded()
print(f"[SeedVR2] upscaler script imported. model dir={MODEL_DIR}")