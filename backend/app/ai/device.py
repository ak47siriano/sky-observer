"""Cross-Platform GPU/CPU Device Detection (NVIDIA CUDA, AMD ROCm, AMD/Intel DirectML, Apple MPS, CPU)."""

from __future__ import annotations

import platform
import shutil
import subprocess
from typing import Any

import psutil
import torch

HARDWARE_PROFILES: dict[str, dict[str, Any]] = {
    "low_end": {
        "id": "low_end",
        "device": "cpu",
        "inference_resolution": 256,
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
        "max_gpu_memory_fraction": 0.60,
        "mixed_precision": False,
        "cpu_threads": 2,
    },
    "balanced": {
        "id": "balanced",
        "device": "auto",
        "inference_resolution": 384,
        "batch_size": 2,
        "gradient_accumulation_steps": 2,
        "max_gpu_memory_fraction": 0.75,
        "mixed_precision": True,
        "cpu_threads": 4,
    },
    "high_end": {
        "id": "high_end",
        "device": "auto",
        "inference_resolution": 512,
        "batch_size": 4,
        "gradient_accumulation_steps": 1,
        "max_gpu_memory_fraction": 0.85,
        "mixed_precision": True,
        "cpu_threads": 0,
    },
    "ultra": {
        "id": "ultra",
        "device": "auto",
        "inference_resolution": 768,
        "batch_size": 8,
        "gradient_accumulation_steps": 1,
        "max_gpu_memory_fraction": 0.90,
        "mixed_precision": True,
        "cpu_threads": 0,
    },
}

MINIMUM_HARDWARE_SPECS: dict[str, Any] = {
    "os": "Linux (x86_64 / ARM64), Windows 10/11, macOS 12+",
    "cpu_cores": 2,
    "ram_gb": 4.0,
    "disk_free_gb": 2.0,
    "gpu": "Not required (runs 100% on CPU at 256x256 - 384x384)",
}

RECOMMENDED_HARDWARE_SPECS: dict[str, Any] = {
    "os": "Linux (Ubuntu/Debian) or Windows 10/11 (64-bit)",
    "cpu_cores": 4,
    "ram_gb": 8.0,
    "disk_free_gb": 10.0,
    "gpu": "Optional NVIDIA CUDA, AMD ROCm (Linux), AMD/Intel DirectML (Windows), or Apple MPS (>= 4 GB VRAM)",
}


def is_rocm_available() -> bool:
    """True when PyTorch is running on an AMD ROCm (HIP) GPU via the torch.cuda interface."""
    try:
        return bool(torch.cuda.is_available() and getattr(torch.version, "hip", None))
    except Exception:
        return False


def is_mps_available() -> bool:
    """Check if Apple Silicon Metal Performance Shaders (MPS) backend is available."""
    try:
        return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    except Exception:
        return False


def is_directml_available() -> bool:
    """Check if Microsoft DirectML (torch-directml) is installed for AMD Radeon / Intel GPUs on Windows."""
    try:
        import torch_directml  # type: ignore[import-not-found]

        return bool(torch_directml.is_available())
    except Exception:
        return False


def get_directml_device_name() -> str:
    try:
        import torch_directml  # type: ignore[import-not-found]

        return str(torch_directml.device_name(0)).replace("\x00", "").strip()
    except Exception:
        return "AMD / Intel DirectML GPU"


def to_torch_device(device: str | torch.device) -> Any:
    """Convert a logical device string ('cuda', 'rocm', 'directml', 'mps', 'cpu') into a PyTorch device object."""
    if not isinstance(device, str):
        return device
    dev = device.lower().strip()
    if dev == "directml":
        try:
            import torch_directml  # type: ignore[import-not-found]

            if torch_directml.is_available():
                return torch_directml.device()
        except Exception:
            pass
        return torch.device("cpu")
    if dev == "rocm":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev == "mps":
        return torch.device("mps" if is_mps_available() else "cpu")
    return torch.device("cpu")


def apply_cpu_thread_limit(cpu_threads: int = 0) -> int:
    """Configure PyTorch intra-op CPU threads to prevent saturating low-end dual/quad-core machines."""
    logical = psutil.cpu_count(logical=True) or 2
    target = cpu_threads if cpu_threads > 0 else max(1, min(8, logical // 2 if logical >= 4 else logical))
    try:
        torch.set_num_threads(max(1, int(target)))
    except Exception:
        pass
    return max(1, int(target))


def resolve_device(preferred: str = "auto", max_gpu_memory_fraction: float = 0.85, cpu_threads: int = 0) -> str:
    """
    Resolve logical PyTorch device ('cuda', 'directml', 'mps', or 'cpu') across NVIDIA CUDA,
    AMD ROCm (Linux via torch.cuda/HIP), AMD/Intel DirectML (Windows DX12), Apple MPS, and CPU.
    """
    apply_cpu_thread_limit(cpu_threads)
    pref = (preferred or "auto").lower().strip()
    if pref == "cpu":
        return "cpu"
    if pref in {"auto", "cuda", "rocm"} and torch.cuda.is_available():
        if pref == "rocm" and not is_rocm_available():
            # Fall through cleanly if ROCm specifically requested on non-ROCm CUDA
            pass
        try:
            torch.cuda.set_per_process_memory_fraction(max(0.1, min(0.98, max_gpu_memory_fraction)), 0)
        except Exception:
            pass
        return "cuda"
    if pref in {"auto", "directml"} and is_directml_available():
        return "directml"
    if pref in {"auto", "mps"} and is_mps_available():
        return "mps"
    return "cpu"


def recommend_safe_batch_size(
    device: str,
    inference_resolution: int = 512,
    total_vram_gb: float = 0.0,
    ram_total_gb: float = 8.0,
) -> int:
    """Suggest a safe training batch size scaled dynamically to the host's actual CPU RAM or GPU VRAM."""
    if device == "cpu":
        if ram_total_gb < 6.0 or inference_resolution >= 512:
            return 1
        if ram_total_gb < 12.0:
            return 2
        return 4
    if device in {"mps", "directml"}:
        return 2 if inference_resolution >= 512 else 4
    if total_vram_gb <= 3.5:
        return 1
    if total_vram_gb <= 5.5:
        return 2 if inference_resolution <= 512 else 1
    if total_vram_gb <= 9.0:
        return 2 if inference_resolution >= 640 else 4
    return 8


def detect_hardware_tier(
    cuda_available: bool,
    mps_available: bool,
    directml_available: bool,
    vram_total_gb: float,
    ram_total_gb: float,
    cpu_cores: int,
) -> str:
    """Classify the host machine into 'low_end', 'balanced', 'high_end', or 'ultra'."""
    if cuda_available and vram_total_gb >= 11.5 and ram_total_gb >= 15.0:
        return "ultra"
    if (
        (cuda_available and vram_total_gb >= 5.5 and ram_total_gb >= 11.0)
        or (mps_available and ram_total_gb >= 15.0)
        or (directml_available and ram_total_gb >= 15.0)
    ):
        return "high_end"
    if ram_total_gb < 6.5 or cpu_cores <= 2:
        return "low_end"
    return "balanced"


_cached_system_gpus: list[str] | None = None


def query_system_gpus() -> list[str]:
    """Detect physical GPU names on Windows (WMI/PowerShell) or Linux (lspci), including AMD Radeon & Intel."""
    global _cached_system_gpus
    if _cached_system_gpus is not None:
        return _cached_system_gpus
    names: list[str] = []
    try:
        if platform.system() == "Windows":
            out = subprocess.check_output(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "(Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name)",
                ],
                timeout=4,
                text=True,
            )
            for line in out.splitlines():
                s = line.strip()
                if s and "microsoft basic" not in s.lower() and "virtual" not in s.lower():
                    names.append(s)
        elif shutil.which("lspci"):
            out = subprocess.check_output(["lspci"], timeout=3, text=True)
            for line in out.splitlines():
                low = line.lower()
                if "vga" in low or "3d controller" in low or "display controller" in low:
                    parts = line.split(":", 2)
                    names.append(parts[-1].strip() if len(parts) >= 3 else line.strip())
    except Exception:
        pass
    _cached_system_gpus = names
    return names


def query_nvidia_smi() -> dict[str, Any] | None:
    """Query local nvidia-smi if present on the host or inside NVIDIA Container Toolkit."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.check_output(
            [
                exe,
                "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            timeout=3,
            text=True,
        ).strip()
        if not out:
            return None
        first = out.splitlines()[0].split(",")
        if len(first) < 6:
            return None
        return {
            "name": first[0].strip(),
            "vram_total_mb": float(first[1].strip()),
            "vram_used_mb": float(first[2].strip()),
            "vram_free_mb": float(first[3].strip()),
            "gpu_utilization_pct": float(first[4].strip()),
            "temperature_c": float(first[5].strip()),
        }
    except Exception:
        return None


def get_hardware_status(
    preferred_device: str = "auto",
    max_gpu_fraction: float = 0.85,
    resolution: int = 512,
    hardware_profile: str = "auto",
    cpu_threads: int = 0,
) -> dict[str, Any]:
    """Return full cross-platform OS, CPU, RAM, CUDA/ROCm/DirectML/MPS status, and compatibility report."""
    cuda_available = bool(torch.cuda.is_available())
    rocm_available = is_rocm_available()
    directml_available = is_directml_available()
    mps_available = is_mps_available()
    active_device = resolve_device(preferred_device, max_gpu_fraction, cpu_threads=cpu_threads)

    vm = psutil.virtual_memory()
    cpu_pct = float(psutil.cpu_percent(interval=None))
    logical_cores = psutil.cpu_count(logical=True) or 1
    physical_cores = psutil.cpu_count(logical=False) or logical_cores
    ram_total_gb = round(vm.total / (1024**3), 2)
    ram_avail_gb = round(vm.available / (1024**3), 2)

    os_name = platform.system() or "Unknown"
    arch = platform.machine() or "x86_64"
    cpu_label = f"{os_name} {arch} CPU ({physical_cores}C/{logical_cores}T)"

    if cuda_available and active_device == "cuda":
        raw_name = torch.cuda.get_device_name(0)
        dev_name = f"{raw_name} (AMD ROCm)" if rocm_available else raw_name
        gpu_backend = "rocm" if rocm_available else "cuda"
    elif directml_available and active_device == "directml":
        dev_name = f"{get_directml_device_name()} (DirectML)"
        gpu_backend = "directml"
    elif mps_available and active_device == "mps":
        dev_name = f"Apple Silicon MPS ({arch})"
        gpu_backend = "mps"
    else:
        sys_gpus = query_system_gpus()
        detected_hint = f" · GPU: {sys_gpus[0]} (install torch-directml / ROCm)" if sys_gpus else ""
        dev_name = f"{cpu_label}{detected_hint}" if preferred_device in {"cpu", "auto"} else f"{cpu_label} [Fallback]{detected_hint}"
        gpu_backend = "cpu"

    gpu_info: dict[str, Any] = {
        "cuda_available": cuda_available,
        "rocm_available": rocm_available,
        "directml_available": directml_available,
        "mps_available": mps_available,
        "gpu_backend": gpu_backend,
        "preferred_device": preferred_device,
        "active_device": active_device,
        "fallback_active": active_device == "cpu" and preferred_device in {"cuda", "rocm", "directml", "mps"},
        "device_name": dev_name,
        "vram_total_gb": 0.0,
        "vram_allocated_mb": 0.0,
        "vram_reserved_mb": 0.0,
        "gpu_utilization_pct": None,
        "recommended_batch_size": recommend_safe_batch_size(active_device, resolution, 0.0, ram_total_gb),
    }

    if cuda_available:
        props = torch.cuda.get_device_properties(0)
        total_gb = round(props.total_memory / (1024**3), 2)
        gpu_info["vram_total_gb"] = total_gb
        gpu_info["vram_allocated_mb"] = round(torch.cuda.memory_allocated(0) / (1024**2), 2)
        gpu_info["vram_reserved_mb"] = round(torch.cuda.memory_reserved(0) / (1024**2), 2)
        gpu_info["recommended_batch_size"] = recommend_safe_batch_size(active_device, resolution, total_gb, ram_total_gb)

    smi = query_nvidia_smi()
    if smi:
        gpu_info["nvidia_smi"] = smi
        gpu_info["gpu_utilization_pct"] = smi.get("gpu_utilization_pct")
        if gpu_info["vram_total_gb"] == 0.0:
            gpu_info["vram_total_gb"] = round(smi["vram_total_mb"] / 1024.0, 2)

    detected_tier = detect_hardware_tier(
        cuda_available=cuda_available,
        mps_available=mps_available,
        directml_available=directml_available,
        vram_total_gb=float(gpu_info["vram_total_gb"]),
        ram_total_gb=ram_total_gb,
        cpu_cores=logical_cores,
    )
    recommended_preset = HARDWARE_PROFILES[detected_tier]
    meets_min = logical_cores >= MINIMUM_HARDWARE_SPECS["cpu_cores"] and ram_total_gb >= 3.5
    meets_rec = logical_cores >= RECOMMENDED_HARDWARE_SPECS["cpu_cores"] and ram_total_gb >= 7.5

    available_devices = [
        {"id": "auto", "label": "Auto (Best Available: NVIDIA / AMD / Apple → CPU)", "available": True},
        {"id": "cpu", "label": f"CPU Only ({physical_cores}C/{logical_cores}T Universal)", "available": True},
        {
            "id": "cuda",
            "label": f"NVIDIA CUDA GPU ({gpu_info['device_name']} · {gpu_info['vram_total_gb']} GB)"
            if (cuda_available and not rocm_available)
            else "NVIDIA CUDA GPU (Not detected — falls back to CPU)",
            "available": cuda_available and not rocm_available,
        },
        {
            "id": "rocm",
            "label": f"AMD ROCm GPU ({gpu_info['device_name']} · {gpu_info['vram_total_gb']} GB)"
            if rocm_available
            else "AMD ROCm GPU (Linux — install PyTorch ROCm)",
            "available": rocm_available,
        },
        {
            "id": "directml",
            "label": f"AMD / Intel DirectML ({get_directml_device_name()})"
            if directml_available
            else "AMD / Intel DirectML (Windows DX12 — pip install torch-directml)",
            "available": directml_available,
        },
    ]
    if mps_available or os_name == "Darwin":
        available_devices.append({"id": "mps", "label": "Apple Silicon Metal (MPS)", "available": mps_available})

    return {
        "os_name": os_name,
        "os_release": platform.release(),
        "architecture": arch,
        "cpu_percent": round(cpu_pct, 1),
        "cpu_cores": logical_cores,
        "cpu_physical_cores": physical_cores,
        "cpu_threads_limit": torch.get_num_threads(),
        "ram_total_gb": ram_total_gb,
        "ram_used_gb": round(vm.used / (1024**3), 2),
        "ram_available_gb": ram_avail_gb,
        "ram_percent": round(vm.percent, 1),
        "detected_tier": detected_tier,
        "active_profile": hardware_profile,
        "recommended_profile": detected_tier,
        "recommended_settings": recommended_preset,
        "meets_minimum_requirements": meets_min,
        "meets_recommended_requirements": meets_rec,
        "minimum_requirements": MINIMUM_HARDWARE_SPECS,
        "recommended_requirements": RECOMMENDED_HARDWARE_SPECS,
        "hardware_profiles": HARDWARE_PROFILES,
        "available_devices": available_devices,
        "gpu": gpu_info,
    }
