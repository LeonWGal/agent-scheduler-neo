import io
import os
import json
import zlib
import base64
import pickle
import inspect
import requests
import numpy as np
import torch
from typing import Union, List, Dict, Any, Optional
from enum import Enum
from PIL import Image, ImageOps, ImageChops, ImageEnhance, ImageFilter, PngImagePlugin
from numpy import ndarray
from torch import Tensor

from modules import sd_samplers, scripts, shared, sd_vae, images, txt2img, img2img
from modules.infotext_utils import create_override_settings_dict
from modules.sd_models import CheckpointInfo, get_closet_checkpoint_match
from modules.api.models import (
    StableDiffusionTxt2ImgProcessingAPI,
    StableDiffusionImg2ImgProcessingAPI,
)

from .helpers import log, get_dict_attribute

img2img_image_args_by_mode: Dict[int, List[List[str]]] = {
    0: [["init_img"]],
    1: [["sketch"], ["sketch_fg"]],
    2: [["init_img_with_mask"], ["init_img_with_mask_fg"], ["init_img_with_mask", "image"], ["init_img_with_mask", "mask"]],
    3: [["inpaint_color_sketch"], ["inpaint_color_sketch_fg"], ["inpaint_color_sketch_orig"]],
    4: [["init_img_inpaint"], ["init_mask_inpaint"]],
}


def get_script_by_name(script_name: str, is_img2img: bool = False, is_always_on: bool = False) -> scripts.Script:
    script_runner = scripts.scripts_img2img if is_img2img else scripts.scripts_txt2img
    available_scripts = script_runner.alwayson_scripts if is_always_on else script_runner.selectable_scripts

    return next(
        (s for s in available_scripts if s.title().lower() == script_name.lower()),
        None,
    )


def load_image_from_url(url: str):
    try:
        response = requests.get(url)
        buffer = io.BytesIO(response.content)
        return Image.open(buffer)
    except Exception as e:
        log.error(f"[AgentScheduler] Error downloading image from url: {e}")
        return None


def encode_image_to_base64(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype("uint8"))
    elif isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            image = load_image_from_url(image)

    if not isinstance(image, Image.Image):
        return image

    geninfo, _ = images.read_info_from_image(image)
    pnginfo = PngImagePlugin.PngInfo()
    if geninfo:
        pnginfo.add_text("parameters", geninfo)

    with io.BytesIO() as output_bytes:
        if geninfo:
            image.save(output_bytes, format="PNG", pnginfo=pnginfo)
        else:
            image.save(output_bytes, format="PNG") # remove pnginfo to save space
        bytes_data = output_bytes.getvalue()
        return "data:image/png;base64," + base64.b64encode(bytes_data).decode("utf-8")


def serialize_image(image):
    if isinstance(image, np.ndarray):
        shape = image.shape
        dtype = image.dtype
        data = base64.b64encode(zlib.compress(image.tobytes())).decode()
        return {"shape": shape, "data": data, "cls": "ndarray", "dtype": str(dtype)}
    elif isinstance(image, torch.Tensor):
        shape = image.shape
        dtype = image.dtype
        data = base64.b64encode(zlib.compress(image.detach().numpy().tobytes())).decode()
        return {
            "shape": shape,
            "data": data,
            "cls": "Tensor",
            "device": image.device.type,
            "dtype": str(dtype),
        }
    elif isinstance(image, Image.Image):
        size = image.size
        mode = image.mode
        data = base64.b64encode(zlib.compress(image.tobytes())).decode()
        return {
            "size": size,
            "mode": mode,
            "data": data,
            "cls": "Image",
        }
    else:
        return image


def deserialize_image(image_str):
    if isinstance(image_str, dict) and image_str.get("cls", None):
        cls = image_str["cls"]
        data = zlib.decompress(base64.b64decode(image_str["data"]))

        if cls == "ndarray":
            # warn if required fields are missing
            if image_str.get("dtype", None) is None:
                log.warning(f"Missing dtype for ndarray")
            shape = tuple(image_str["shape"])
            dtype = np.dtype(image_str.get("dtype", "uint8"))
            image = np.frombuffer(data, dtype=dtype)
            return image.reshape(shape)
        elif cls == "Tensor":
            if image_str.get("device", None) is None:
                log.warning(f"Missing device for Tensor")
            shape = tuple(image_str["shape"])
            dtype = np.dtype(image_str.get("dtype", "uint8"))
            image_np = np.frombuffer(data, dtype=dtype)
            return torch.from_numpy(image_np.reshape(shape)).to(device=image_str.get("device", "cpu"))
        else:
            size = tuple(image_str["size"])
            mode = image_str["mode"]
            return Image.frombytes(mode, size, data)
    else:
        return image_str


def serialize_img2img_image_args(args: Dict):
    for mode, image_args in img2img_image_args_by_mode.items():
        for keys in image_args:
            if mode != args["mode"]:
                # set None to unused image args to save space
                args[keys[0]] = None
            elif len(keys) == 1:
                image = args.get(keys[0], None)
                args[keys[0]] = serialize_image(image)
            else:
                value = args.get(keys[0], {})
                image = value.get(keys[1], None)
                value[keys[1]] = serialize_image(image)
                args[keys[0]] = value


def deserialize_img2img_image_args(args: Dict):
    for mode, image_args in img2img_image_args_by_mode.items():
        if mode != args["mode"]:
            continue

        for keys in image_args:
            if len(keys) == 1:
                image = args.get(keys[0], None)
                args[keys[0]] = deserialize_image(image)
            else:
                value = args.get(keys[0], {})
                image = value.get(keys[1], None)
                value[keys[1]] = deserialize_image(image)
                args[keys[0]] = value


CONTROLNET_TYPE_NAMES = ("UiControlNetUnit", "ControlNetUnit")
CONTROLNET_UNIT_FIELDS = (
    "use_preview_as_input",
    "generated_image",
    "mask_image",
    "mask_image_fg",
    "hr_option",
    "enabled",
    "module",
    "model",
    "weight",
    "image",
    "image_fg",
    "resize_mode",
    "processor_res",
    "threshold_a",
    "threshold_b",
    "guidance_start",
    "guidance_end",
    "pixel_perfect",
    "control_mode",
    "type_filter",
    "save_detected_map",
    "_idx",
)


def is_controlnet_unit_type(obj) -> bool:
    return type(obj).__name__ in CONTROLNET_TYPE_NAMES


def looks_like_controlnet_dict(obj) -> bool:
    if not isinstance(obj, dict):
        return False
    if obj.get("is_cnet", False):
        return True
    # Forge Neo Gradio State / JSON payload shape
    return (
        "enabled" in obj
        and "module" in obj
        and "model" in obj
        and ("control_mode" in obj or "resize_mode" in obj)
    )


def get_controlnet_unit_cls(UiControlNetUnit=None):
    if UiControlNetUnit is not None:
        return UiControlNetUnit
    try:
        from lib_controlnet.external_code import ControlNetUnit

        return ControlNetUnit
    except Exception:
        pass
    try:
        from scripts.external_code import ControlNetUnit  # type: ignore

        return ControlNetUnit
    except Exception:
        return None


def serialize_controlnet_args(cnet_unit):
    if isinstance(cnet_unit, dict):
        args = dict(cnet_unit)
    else:
        args = dict(getattr(cnet_unit, "__dict__", {}))

    serialized_args = {"is_cnet": True}
    for k, v in args.items():
        if k in ("is_cnet", "is_ui"):
            continue
        if isinstance(v, Enum):
            serialized_args[k] = v.value
        elif isinstance(v, (np.ndarray, Tensor, Image.Image)):
            serialized_args[k] = serialize_image(v)
        elif isinstance(v, dict) and any(
            isinstance(v.get(key), (np.ndarray, Tensor, Image.Image)) for key in ("image", "mask")
        ):
            nested = dict(v)
            for key in ("image", "mask"):
                if key in nested:
                    nested[key] = serialize_image(nested[key])
            serialized_args[k] = nested
        else:
            serialized_args[k] = v

    return serialized_args


def deserialize_controlnet_args(args: Dict):
    new_args = args.copy()
    new_args.pop("is_cnet", None)
    new_args.pop("is_ui", None)

    for k, v in list(new_args.items()):
        if isinstance(v, dict) and v.get("cls"):
            new_args[k] = deserialize_image(v)
        elif isinstance(v, dict) and any(isinstance(v.get(key), dict) and v.get(key, {}).get("cls") for key in ("image", "mask")):
            nested = dict(v)
            for key in ("image", "mask"):
                if isinstance(nested.get(key), dict) and nested[key].get("cls"):
                    nested[key] = deserialize_image(nested[key])
            new_args[k] = nested

    return new_args


def ensure_controlnet_unit(value, unit_cls):
    """Coerce null/dict/list/unit into a ControlNetUnit instance for Forge Neo."""
    if unit_cls is None:
        return value

    if value is None:
        return unit_cls(enabled=False, module="None", model="None")

    if isinstance(value, unit_cls) or is_controlnet_unit_type(value):
        return value

    if isinstance(value, dict) and looks_like_controlnet_dict(value):
        raw = deserialize_controlnet_args(value)
        filtered = {k: raw[k] for k in CONTROLNET_UNIT_FIELDS if k in raw}
        try:
            if hasattr(unit_cls, "from_dict"):
                return unit_cls.from_dict(filtered)
            return unit_cls(**filtered)
        except Exception as e:
            log.warning(f"[AgentScheduler] Failed to rebuild ControlNet unit from dict: {e}")
            return unit_cls(enabled=False, module="None", model="None")

    if isinstance(value, (list, tuple)):
        # Positional args matching ControlNetUnit field order (Forge CN UI builder)
        kwargs = {}
        for i, field in enumerate(CONTROLNET_UNIT_FIELDS):
            if i >= len(value):
                break
            kwargs[field] = value[i]
        try:
            return unit_cls(**{k: v for k, v in kwargs.items() if k != "_idx" or v is not None})
        except Exception as e:
            log.warning(f"[AgentScheduler] Failed to rebuild ControlNet unit from list: {e}")
            return unit_cls(enabled=False, module="None", model="None")

    log.warning(f"[AgentScheduler] Unexpected ControlNet arg type {type(value)}; using disabled unit")
    return unit_cls(enabled=False, module="None", model="None")


def normalize_controlnet_script_args(script_args: List, UiControlNetUnit=None, is_img2img: bool = False) -> List:
    """Ensure ControlNet alwayson slots are real ControlNetUnit instances."""
    if not script_args:
        return script_args

    unit_cls = get_controlnet_unit_cls(UiControlNetUnit)
    if unit_cls is None:
        return script_args

    script_args = list(script_args)
    cnet_script = get_script_by_name("controlnet", is_img2img=is_img2img, is_always_on=True)
    if cnet_script is not None:
        start = max(int(getattr(cnet_script, "args_from", 0) or 0), 0)
        end = int(getattr(cnet_script, "args_to", start) or start)
        end = min(end, len(script_args))
        for i in range(start, end):
            script_args[i] = ensure_controlnet_unit(script_args[i], unit_cls)
        return script_args

    # Fallback: coerce any CN-shaped values anywhere in script_args
    for i, a in enumerate(script_args):
        if a is None or looks_like_controlnet_dict(a) or is_controlnet_unit_type(a) or isinstance(a, (list, tuple)):
            # Only coerce clearly CN-shaped dicts / existing units; skip plain lists of unrelated args
            if a is None:
                continue
            if looks_like_controlnet_dict(a) or is_controlnet_unit_type(a):
                script_args[i] = ensure_controlnet_unit(a, unit_cls)

    return script_args


def normalize_script_runner_index(script_args: List, is_img2img: bool = False) -> List:
    """Gradio Dropdown(type='index') may arrive as label 'None' when bypassing preprocess."""
    if not script_args:
        return script_args

    script_args = list(script_args)
    idx = script_args[0]

    if idx is None or idx == "" or idx == "None" or idx is False:
        script_args[0] = 0
        return script_args

    if isinstance(idx, bool):
        script_args[0] = int(idx)
        return script_args

    if isinstance(idx, (int, float)):
        script_args[0] = int(idx)
        return script_args

    if isinstance(idx, str):
        try:
            script_args[0] = int(idx)
            return script_args
        except ValueError:
            pass
        runner = scripts.scripts_img2img if is_img2img else scripts.scripts_txt2img
        titles = ["None"] + [s.title() for s in getattr(runner, "selectable_scripts", [])]
        try:
            script_args[0] = titles.index(idx)
        except ValueError:
            log.warning(f"[AgentScheduler] Unknown script selection {idx!r}; using 0")
            script_args[0] = 0
        return script_args

    log.warning(f"[AgentScheduler] Unexpected script index type {type(idx)}: {idx!r}; using 0")
    script_args[0] = 0
    return script_args


def default_adetailer_state(tab_index: int = 0) -> Dict[str, Any]:
    """Minimal ADetailer-Neo State dict (matches lib_adetailer.args.ADetailerArgs defaults)."""
    return {
        "ad_model": "None",
        "ad_model_classes": "",
        "ad_tab_enable": tab_index == 0,
        "ad_prompt": "",
        "ad_negative_prompt": "",
        "ad_confidence": 0.3,
        "ad_mask_filter_method": "Area",
        "ad_mask_k": 0,
        "ad_mask_min_ratio": 0.0,
        "ad_mask_max_ratio": 1.0,
        "ad_x_offset": 0,
        "ad_y_offset": 0,
        "ad_dilate_erode": 4,
        "ad_mask_merge_invert": "None",
        "ad_mask_blur": 4,
        "ad_denoising_strength": 0.4,
        "ad_inpaint_only_masked": True,
        "ad_inpaint_only_masked_padding": 32,
        "ad_use_inpaint_width_height": False,
        "ad_inpaint_width": 512,
        "ad_inpaint_height": 512,
        "ad_use_steps": False,
        "ad_steps": 20,
        "ad_use_cfg_scale": False,
        "ad_cfg_scale": 4.0,
        "ad_use_checkpoint": False,
        "ad_checkpoint": None,
        "ad_use_vae": False,
        "ad_vae": None,
        "ad_use_sampler": False,
        "ad_sampler": "Use same sampler",
        "ad_scheduler": "Use same scheduler",
        "ad_use_noise_multiplier": False,
        "ad_noise_multiplier": 1.0,
        "ad_restore_face": False,
        "ad_controlnet_model": "None",
        "ad_controlnet_module": "None",
        "ad_controlnet_weight": 1.0,
        "ad_controlnet_guidance_start_end": (0.0, 1.0),
        "is_api": (),
    }


def looks_like_adetailer_state(obj) -> bool:
    return isinstance(obj, dict) and "ad_model" in obj


def normalize_adetailer_script_args(script_args: List, is_img2img: bool = False) -> List:
    """
    ADetailer-Neo stores per-tab settings in gr.State.
    Enqueue bridge may leave those slots as null — coerce to valid State dicts
    and convert JSON list guidance ranges back to tuples.
    """
    if not script_args:
        return script_args

    script_args = list(script_args)
    ad_script = get_script_by_name("adetailer", is_img2img=is_img2img, is_always_on=True)
    if ad_script is None:
        return script_args

    start = max(int(getattr(ad_script, "args_from", 0) or 0), 0)
    end = int(getattr(ad_script, "args_to", start) or start)
    end = min(end, len(script_args))
    if end <= start:
        return script_args

    # Layout: [enable: bool, skip_img2img: bool, *states]
    enable = script_args[start] if start < end else False
    state_start = start + 2
    tab_i = 0
    models = []
    for i in range(state_start, end):
        val = script_args[i]
        if val is None or val == "" or val == "None":
            script_args[i] = default_adetailer_state(tab_i)
        elif isinstance(val, dict):
            state = dict(val)
            # JSON round-trip: tuple → list
            gse = state.get("ad_controlnet_guidance_start_end")
            if isinstance(gse, list) and len(gse) == 2:
                state["ad_controlnet_guidance_start_end"] = (gse[0], gse[1])
            # Preserve UI marker across JSON (list) vs pickle (tuple)
            if "is_api" in state and not isinstance(state["is_api"], tuple):
                state["is_api"] = ()
            if "ad_model" not in state:
                state = {**default_adetailer_state(tab_i), **state}
            script_args[i] = state
        # Only treat dict-like slots as AD states (skip unexpected types)
        if isinstance(script_args[i], dict) and "ad_model" in script_args[i]:
            models.append(script_args[i].get("ad_model"))
            tab_i += 1

    if enable and models and all(m in (None, "None", "") for m in models):
        log.warning(
            "[AgentScheduler] ADetailer enabled but all detectors are None — "
            "State was likely lost at enqueue (check enqueue-bridge ADetailer capture)"
        )

    return script_args


# Defaults for XYZ ui()/run() slots after the three axis triples (offsets 9..18).
_XYZ_SLOT_DEFAULTS = {
    9: True,  # draw_legend
    10: False,  # include_lone_images
    11: False,  # include_sub_grids
    12: False,  # no_fixed_seeds
    13: False,  # vary_seeds_x
    14: False,  # vary_seeds_y
    15: False,  # vary_seeds_z
    16: 0,  # row_count
    17: 0,  # margin_size
    18: False,  # csv_mode
}


def _xyz_dropdown_choice_labels(choices) -> Optional[List[str]]:
    if not choices:
        return None
    out = []
    for c in choices:
        if isinstance(c, (tuple, list)) and c:
            out.append(str(c[0]))
        else:
            out.append(str(c))
    return out or None


def _xyz_axis_labels(xyz_script, is_img2img: bool) -> Optional[List[str]]:
    """Labels for X/Y/Z Dropdown(type='index') — same order as xyz_grid.Script.ui()."""
    # Prefer live Gradio dropdown choices (exact index mapping used by preprocess)
    controls = getattr(xyz_script, "controls", None) or []
    if controls:
        labels = _xyz_dropdown_choice_labels(getattr(controls[0], "choices", None))
        if labels:
            return labels

    opts = getattr(xyz_script, "current_axis_options", None)
    if opts:
        return [o.label for o in opts]

    mod = inspect.getmodule(type(xyz_script))
    if mod is None or not hasattr(mod, "axis_options"):
        return None

    AxisOption = getattr(mod, "AxisOption", None)
    if AxisOption is None:
        return None

    # Match xyz_grid.Script.ui() filter (type==AxisOption OR is_img2img flag)
    filtered = [
        x
        for x in mod.axis_options
        if type(x) == AxisOption or getattr(x, "is_img2img", None) == is_img2img
    ]
    return [x.label for x in filtered]


def _coerce_index_dropdown(value, choices: Optional[List[str]], field_name: str) -> int:
    """Gradio Dropdown(type='index') may arrive as a label string when bypassing preprocess."""
    if value is None or value == "" or value == "None" or value is False:
        return 0

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, float)):
        return int(value)

    if isinstance(value, str):
        # Pure numeric index string only (avoid int("8 Steps") style accidents)
        if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
            return int(value)
        if choices:
            try:
                return choices.index(value)
            except ValueError:
                lower = [c.lower() for c in choices]
                try:
                    return lower.index(value.lower())
                except ValueError:
                    pass
        log.warning(
            f"[AgentScheduler] Unknown XYZ {field_name} {value!r} "
            f"(choices={len(choices) if choices else 0}); using 0 (Nothing)"
        )
        return 0

    log.warning(
        f"[AgentScheduler] Unexpected XYZ {field_name} type {type(value)}: {value!r}; using 0"
    )
    return 0


def normalize_xyz_grid_script_args(script_args: List, is_img2img: bool = False) -> List:
    """
    X/Y/Z plot uses Dropdown(type='index') for x_type / y_type / z_type.
    Enqueue bridge may leave those slots as axis label strings — coerce to ints
    so current_axis_options[x_type] works after queue round-trip.

    Also pads truncated trailing XYZ slots (create_submit_args / stale config can
    drop no_fixed_seeds..csv_mode) so script.run() receives a full arg list.

    Layout (relative to script.args_from):
      0:x_type, 1:x_values, 2:x_values_dropdown,
      3:y_type, 4:y_values, 5:y_values_dropdown,
      6:z_type, ...
    """
    if not script_args:
        return script_args

    xyz_script = get_script_by_name("x/y/z plot", is_img2img=is_img2img, is_always_on=False)
    if xyz_script is None:
        return script_args

    start = max(int(getattr(xyz_script, "args_from", 0) or 0), 0)
    full_end = int(getattr(xyz_script, "args_to", start) or start)
    if full_end <= start:
        return script_args

    script_args = list(script_args)

    # Pad missing trailing XYZ controls (enqueue may truncate after include_sub_grids)
    if len(script_args) < full_end:
        missing = full_end - len(script_args)
        log.warning(
            f"[AgentScheduler] XYZ script_args truncated by {missing} "
            f"(have {len(script_args)}, need args_to={full_end}); padding defaults"
        )
        while len(script_args) < full_end:
            offset = len(script_args) - start
            script_args.append(_XYZ_SLOT_DEFAULTS.get(offset))

    end = min(full_end, len(script_args))
    # Need at least through z_type (offset 6)
    if end - start < 7:
        return script_args

    labels = _xyz_axis_labels(xyz_script, is_img2img)

    for offset, name in ((0, "x_type"), (3, "y_type"), (6, "z_type")):
        i = start + offset
        if i >= end:
            break
        before = script_args[i]
        after = _coerce_index_dropdown(before, labels, name)
        script_args[i] = after
        if before != after:
            label = (
                labels[after]
                if labels and 0 <= after < len(labels)
                else "?"
            )
            log.info(
                f"[AgentScheduler] XYZ {name}: {before!r} → {after} ({label})"
            )

    return script_args


def serialize_script_args(script_args: List):
    # convert ControlNetUnit / UiControlNetUnit (and CN-shaped dicts) for pickle
    script_args = list(script_args)
    for i, a in enumerate(script_args):
        if is_controlnet_unit_type(a) or looks_like_controlnet_dict(a):
            script_args[i] = serialize_controlnet_args(a)

    return zlib.compress(pickle.dumps(script_args))


def deserialize_script_args(script_args: Union[bytes, List], UiControlNetUnit=None, is_img2img: bool = False):
    if type(script_args) is bytes:
        script_args = pickle.loads(zlib.decompress(script_args))

    script_args = normalize_script_runner_index(script_args, is_img2img=is_img2img)
    script_args = list(script_args)

    unit_cls = get_controlnet_unit_cls(UiControlNetUnit)

    for i, a in enumerate(script_args):
        if isinstance(a, dict) and (a.get("is_cnet", False) or looks_like_controlnet_dict(a)):
            unit = deserialize_controlnet_args(a)
            skip_controlnet = False
            if unit_cls is not None:
                try:
                    u = unit_cls()
                    for k, v in list(unit.items()):
                        attr = getattr(u, k, None)
                        if isinstance(attr, Enum):
                            enum_cls = attr.__class__
                            if v not in [e.value for e in enum_cls]:
                                # allow enum name as well as value
                                if isinstance(v, str) and hasattr(enum_cls, v):
                                    unit[k] = getattr(enum_cls, v)
                                else:
                                    log.error(
                                        f"Invalid enum value {v} for {k}, valid: {[e.value for e in enum_cls]}"
                                    )
                                    skip_controlnet = True
                                    break
                            else:
                                unit[k] = enum_cls(v)
                    if not skip_controlnet:
                        script_args[i] = ensure_controlnet_unit(unit, unit_cls)
                except Exception as e:
                    log.warning(f"[AgentScheduler] ControlNet deserialize fallback: {e}")
                    script_args[i] = ensure_controlnet_unit(unit, unit_cls)
            else:
                script_args[i] = unit

    # Final pass: coerce ControlNet alwayson slots (handles nulls from Gradio State bridge)
    script_args = normalize_controlnet_script_args(script_args, unit_cls, is_img2img=is_img2img)
    # ADetailer State slots (null → default dict; fix JSON list→tuple guidance)
    script_args = normalize_adetailer_script_args(script_args, is_img2img=is_img2img)
    # XYZ plot axis type Dropdown(type=index) may be label strings after enqueue
    script_args = normalize_xyz_grid_script_args(script_args, is_img2img=is_img2img)
    return script_args


def map_controlnet_args_to_api_task_args(args: Dict):
    if is_controlnet_unit_type(args):
        args = dict(args.__dict__)

    for k, v in args.items():
        if k == "image" and v is not None:
            args[k] = {
                "image": encode_image_to_base64(v["image"]),
                "mask": encode_image_to_base64(v["mask"]) if v.get("mask", None) is not None else None,
            }
        if isinstance(v, Enum):
            args[k] = v.value

    return args


def map_ui_task_args_list_to_named_args(args: List, is_img2img: bool):
    fn = (
        getattr(img2img, "img2img_create_processing", img2img.img2img)
        if is_img2img
        else getattr(txt2img, "txt2img_create_processing", txt2img.txt2img)
    )
    arg_names = inspect.getfullargspec(fn).args

    # SD WebUI 1.5.0 has new request arg
    if "request" in arg_names:
        args.insert(arg_names.index("request"), None)

    named_args = dict(zip(arg_names, args[0 : len(arg_names)]))
    script_args = normalize_script_runner_index(args[len(arg_names) :], is_img2img=is_img2img)

    override_settings_texts: List[str] = named_args.get("override_settings_texts", [])
    if override_settings_texts is None:
        override_settings_texts = []
    # add clip_skip if not exist in args (vlad fork has this arg)
    if named_args.get("clip_skip", None) is None:
        clip_skip = None
        if override_settings_texts != None:
            clip_skip = next((s for s in override_settings_texts if s.startswith("Clip skip:")), None)
        if clip_skip is None and hasattr(shared.opts, "CLIP_stop_at_last_layers"):
            override_settings_texts.append(f"Clip skip: {shared.opts.CLIP_stop_at_last_layers}")

    named_args["override_settings_texts"] = override_settings_texts

    sampler_index = named_args.get("sampler_index", None)
    if sampler_index is not None:
        available_samplers = sd_samplers.samplers_for_img2img if is_img2img else sd_samplers.samplers
        sampler_name = available_samplers[named_args["sampler_index"]].name
        named_args["sampler_name"] = sampler_name
        log.debug(f"serialize sampler index: {str(sampler_index)} as {sampler_name}")

    return (
        named_args,
        script_args,
    )


def map_named_args_to_ui_task_args_list(named_args: Dict, script_args: List, is_img2img: bool):
    fn = (
        getattr(img2img, "img2img_create_processing", img2img.img2img)
        if is_img2img
        else getattr(txt2img, "txt2img_create_processing", txt2img.txt2img)
    )
    arg_names = inspect.getfullargspec(fn).args

    sampler_name = named_args.get("sampler_name", None)
    if sampler_name is not None:
        available_samplers = sd_samplers.samplers_for_img2img if is_img2img else sd_samplers.samplers
        sampler_index = next((i for i, x in enumerate(available_samplers) if x.name == sampler_name), 0)
        named_args["sampler_index"] = sampler_index

    args = [named_args.get(name, None) for name in arg_names]
    args.extend(normalize_script_runner_index(script_args, is_img2img=is_img2img))

    return args


def map_script_args_list_to_named(script: scripts.Script, args: List):
    script_name = script.title().lower()

    if script_name == "controlnet":
        for i, cnet_args in enumerate(args):
            args[i] = map_controlnet_args_to_api_task_args(cnet_args)

        return args

    fn = script.process if script.alwayson else script.run
    inspection = inspect.getfullargspec(fn)
    arg_names = inspection.args[2:]
    named_script_args = dict(zip(arg_names, args[: len(arg_names)]))
    if inspection.varargs is not None:
        named_script_args[inspection.varargs] = args[len(arg_names) :]

    return named_script_args


def map_named_script_args_to_list(script: scripts.Script, named_args: Union[dict, list]):
    script_name = script.title().lower()

    if isinstance(named_args, dict):
        fn = script.process if script.alwayson else script.run
        inspection = inspect.getfullargspec(fn)
        arg_names = inspection.args[2:]
        args = [named_args.get(name, None) for name in arg_names]
        if inspection.varargs is not None:
            args.extend(named_args.get(inspection.varargs, []))

        return args

    if isinstance(named_args, list):
        if script_name == "controlnet":
            for i, cnet_args in enumerate(named_args):
                named_args[i] = map_controlnet_args_to_api_task_args(cnet_args)

        return named_args


def map_ui_task_args_to_api_task_args(named_args: Dict, script_args: List, is_img2img: bool):
    api_task_args: Dict = named_args.copy()

    prompt_styles = api_task_args.pop("prompt_styles", [])
    api_task_args["styles"] = prompt_styles

    sampler_index = api_task_args.pop("sampler_index", 0)
    api_task_args["sampler_name"] = sd_samplers.samplers[sampler_index].name

    override_settings_texts = api_task_args.pop("override_settings_texts", [])
    api_task_args["override_settings"] = create_override_settings_dict(override_settings_texts)

    if is_img2img:
        mode = api_task_args.pop("mode", 0)
        for arg_mode, image_args in img2img_image_args_by_mode.items():
            if mode != arg_mode:
                for keys in image_args:
                    api_task_args.pop(keys[0], None)

        # the logic below is copied from modules/img2img.py
        if mode == 0:
            image = api_task_args.pop("init_img", None)
            image = image.convert("RGB") if image else None
            mask = None
        elif mode == 1:
            sketch = api_task_args.pop("sketch", None)
            sketch_fg = api_task_args.pop("sketch_fg", None)
            if sketch and sketch_fg:
                image = Image.alpha_composite(sketch.convert("RGBA"), sketch_fg.convert("RGBA")).convert("RGB")
            else:
                image = sketch.convert("RGB") if sketch else None
            mask = None
        elif mode == 2:
            init_img_with_mask = api_task_args.pop("init_img_with_mask", None)
            init_img_with_mask_fg = api_task_args.pop("init_img_with_mask_fg", None)
            if isinstance(init_img_with_mask, dict):
                image = init_img_with_mask.get("image", None)
                image = image.convert("RGB") if image else None
                mask = init_img_with_mask.get("mask", None)
                if mask:
                    alpha_mask = (
                        ImageOps.invert(image.split()[-1]).convert("L").point(lambda x: 255 if x > 0 else 0, mode="1")
                    )
                    mask = ImageChops.lighter(alpha_mask, mask.convert("L")).convert("L")
            else:
                image = init_img_with_mask.convert("RGB") if init_img_with_mask else None
                if init_img_with_mask_fg:
                    mask = init_img_with_mask_fg.getchannel("A").convert("L")
                    mask = Image.merge("RGBA", (mask, mask, mask, Image.new("L", mask.size, 255)))
                else:
                    mask = None
        elif mode == 3:
            inpaint_color_sketch = api_task_args.pop("inpaint_color_sketch", None)
            inpaint_color_sketch_fg = api_task_args.pop("inpaint_color_sketch_fg", None)
            orig = api_task_args.pop("inpaint_color_sketch_orig", None)
            if inpaint_color_sketch_fg is not None and inpaint_color_sketch is not None:
                image = Image.alpha_composite(inpaint_color_sketch.convert("RGBA"), inpaint_color_sketch_fg.convert("RGBA")).convert("RGB")
                mask = inpaint_color_sketch_fg.getchannel("A").convert("L")
            elif inpaint_color_sketch is not None:
                orig = orig or inpaint_color_sketch
                mask_alpha = api_task_args.pop("mask_alpha", 0)
                mask_blur = api_task_args.get("mask_blur", 4)
                pred = np.any(np.array(inpaint_color_sketch) != np.array(orig), axis=-1)
                mask = Image.fromarray(pred.astype(np.uint8) * 255, "L")
                mask = ImageEnhance.Brightness(mask).enhance(1 - mask_alpha / 100)
                blur = ImageFilter.GaussianBlur(mask_blur)
                image = Image.composite(inpaint_color_sketch.filter(blur), orig, mask.filter(blur))
                image = image.convert("RGB")
            else:
                image = None
                mask = None
        elif mode == 4:
            image = api_task_args.pop("init_img_inpaint", None)
            mask = api_task_args.pop("init_mask_inpaint", None)
        else:
            raise Exception(f"Batch mode is not supported yet")

        image = ImageOps.exif_transpose(image) if image else None
        api_task_args["init_images"] = [encode_image_to_base64(image)] if image else []
        api_task_args["mask"] = encode_image_to_base64(mask) if mask else None

        selected_scale_tab = api_task_args.pop("selected_scale_tab", 0)
        scale_by = api_task_args.get("scale_by", 1)
        if selected_scale_tab == 1 and image:
            api_task_args["width"] = int(image.width * scale_by)
            api_task_args["height"] = int(image.height * scale_by)
    else:
        hr_sampler_index = api_task_args.pop("hr_sampler_index", 0)
        api_task_args["hr_sampler_name"] = (
            sd_samplers.samplers_for_img2img[hr_sampler_index - 1].name if hr_sampler_index != 0 else None
        )

    # script
    script_runner = scripts.scripts_img2img if is_img2img else scripts.scripts_txt2img
    script_id = script_args[0]
    if script_id == 0:
        api_task_args["script_name"] = None
        api_task_args["script_args"] = []
    else:
        script: scripts.Script = script_runner.selectable_scripts[script_id - 1]
        api_task_args["script_name"] = script.title().lower()
        current_script_args = script_args[script.args_from : script.args_to]
        api_task_args["script_args"] = map_script_args_list_to_named(script, current_script_args)

    # alwayson scripts
    alwayson_scripts = api_task_args.get("alwayson_scripts", None)
    if not alwayson_scripts:
        api_task_args["alwayson_scripts"] = {}
        alwayson_scripts = api_task_args["alwayson_scripts"]

    for script in script_runner.alwayson_scripts:
        alwayson_script_args = script_args[script.args_from : script.args_to]
        script_name = script.title().lower()
        if script_name != "agent scheduler":
            named_script_args = map_script_args_list_to_named(script, alwayson_script_args)
            alwayson_scripts[script_name] = {"args": named_script_args}

    return api_task_args


def serialize_api_task_args(
    params: Dict,
    is_img2img: bool,
    checkpoint: str = None,
    vae: str = None,
) -> Dict:
    # handle named script args
    script_name = params.get("script_name", None)
    if script_name is not None and script_name != "":
        script = get_script_by_name(script_name, is_img2img)
        if script is None:
            raise Exception(f"Not found script {script_name}")

        script_args = params.get("script_args", {})
        params["script_args"] = map_named_script_args_to_list(script, script_args)

    # handle named alwayson script args
    alwayson_scripts = get_dict_attribute(params, "alwayson_scripts", {})
    assert type(alwayson_scripts) is dict

    script_runner = scripts.scripts_img2img if is_img2img else scripts.scripts_txt2img
    allowed_alwayson_scripts = {s.title().lower(): s for s in script_runner.alwayson_scripts}

    valid_alwayson_scripts = {}
    for script_name, script_args in alwayson_scripts.items():
        if script_name.lower() == "agent scheduler":
            continue

        if script_name.lower() not in allowed_alwayson_scripts:
            log.warning(f"Script {script_name} is not in script_runner.alwayson_scripts")
            continue

        script = allowed_alwayson_scripts[script_name.lower()]
        script_args = get_dict_attribute(script_args, "args", [])
        arg_list = map_named_script_args_to_list(script, script_args)
        valid_alwayson_scripts[script_name] = {"args": arg_list}

    params["alwayson_scripts"] = valid_alwayson_scripts

    args = (
        StableDiffusionImg2ImgProcessingAPI(**params) if is_img2img else StableDiffusionTxt2ImgProcessingAPI(**params)
    )

    if args.override_settings is None:
        args.override_settings = {}

    if checkpoint is not None:
        checkpoint_info: CheckpointInfo = get_closet_checkpoint_match(checkpoint)
        if not checkpoint_info:
            log.warning(f"Checkpoint {checkpoint} not found, use current system model")
        else:
            args.override_settings["sd_model_checkpoint"] = checkpoint_info.title

    if vae is not None:
        if vae not in sd_vae.vae_dict:
            log.warning(f"VAE {vae} not found, use current system vae")
        else:
            args.override_settings["sd_vae"] = vae

    # load images from url or file if needed
    if is_img2img:
        init_images = args.init_images
        if len(init_images) == 0:
            raise Exception("At least one init image is required")

        for i, image in enumerate(init_images):
            init_images[i] = encode_image_to_base64(image)

        args.mask = encode_image_to_base64(args.mask)
        if len(init_images) > 1:
            args.batch_size = len(init_images)

    return args.model_dump() if hasattr(args, "model_dump") else args.dict()


def _looks_like_geninfo(obj: Any) -> bool:
    return isinstance(obj, dict) and ("infotexts" in obj or "all_prompts" in obj or "seed" in obj)


def _parse_geninfo_candidate(value: Any) -> Optional[dict]:
    if value is None:
        return None
    if isinstance(value, dict):
        # Gradio Update payload — not generation info
        if value.get("__type__") == "update":
            return None
        if _looks_like_geninfo(value):
            return value
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or not text.startswith("{"):
            return None
        try:
            parsed = json.loads(text)
        except Exception:
            return None
        if isinstance(parsed, dict) and parsed.get("__type__") == "update":
            return None
        if _looks_like_geninfo(parsed):
            return parsed
    return None


def extract_generation_info(result: Any) -> Optional[dict]:
    """
    Pull generation_info from txt2img/img2img return values.

    Forge Neo returns: (gallery_update, video_update, generation_info_js, html_info, html_comments)
    A1111 returns:     (images, generation_info_js, html_info, html_comments)
    """
    if isinstance(result, dict):
        return _parse_geninfo_candidate(result)

    if isinstance(result, str):
        return _parse_geninfo_candidate(result)

    if not isinstance(result, (list, tuple)):
        return None

    # Prefer known geninfo slots (Forge Neo index 2, A1111 index 1)
    for idx in (2, 1, 0, 3):
        if idx < len(result):
            parsed = _parse_geninfo_candidate(result[idx])
            if parsed is not None:
                return parsed

    for part in result:
        parsed = _parse_geninfo_candidate(part)
        if parsed is not None:
            return parsed

    return None


def extract_ui_task_error(result: Any, last_exception: Optional[str] = None) -> Optional[str]:
    """
    Recover the real failure message when Forge Neo swallows exceptions.

    modules_forge.main_thread catches exceptions, sets last_exception, and returns None.
    wrap_gradio_call then turns that into an HTML error div — Agent Scheduler previously
    saw extract_generation_info(...) == None and logged "failed: None".
    """
    if last_exception:
        msg = str(last_exception).strip()
        if msg:
            return msg

    if not isinstance(result, (list, tuple)):
        return None

    import re
    from html import unescape

    for part in result:
        if not isinstance(part, str):
            continue
        if "class='error'" not in part and 'class="error"' not in part:
            continue
        text = unescape(re.sub(r"<[^>]+>", " ", part))
        text = re.sub(r"\s+", " ", text).strip()
        # Drop Gradio performance footer noise if present
        text = re.split(r"Time taken:", text, maxsplit=1)[0].strip()
        if text and text.lower() not in ("none", "null"):
            return text
    return None


def normalize_saved_image_paths(paths: List[str]) -> List[str]:
    normalized = []
    for path in paths:
        if not path:
            continue
        abs_path = os.path.abspath(path)
        if abs_path not in normalized:
            normalized.append(abs_path)
    return normalized


def geninfo_from_saved_images(image_paths: List[str]) -> Optional[dict]:
    """Rebuild geninfo from PNG metadata when UI return value was a Gradio Update."""
    infotexts = []
    for path in image_paths:
        if not path or not os.path.isfile(path):
            continue
        try:
            with Image.open(path) as image:
                geninfo, _ = images.read_info_from_image(image)
            if geninfo:
                infotexts.append(geninfo)
            else:
                infotexts.append("")
        except Exception as e:
            log.warning(f"[AgentScheduler] Failed to read geninfo from {path}: {e}")
            infotexts.append("")

    if not any(infotexts):
        return None

    return {
        "infotexts": infotexts,
        "all_prompts": [],
        "all_seeds": [],
        "index_of_first_image": 0,
    }
