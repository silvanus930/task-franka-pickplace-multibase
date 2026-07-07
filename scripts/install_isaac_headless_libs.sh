#!/usr/bin/env bash
# System libraries for Isaac Sim / Isaac Lab in headless Linux containers.
#
# Fixes common startup errors:
#   libGLU.so.1 / libXt.so.6  — RTX/neuray material loading (training hang)
#   libvulkan1                — base Vulkan loader (video / offscreen render)
#
# NOTE: Video recording also needs a working NVIDIA *graphics* stack (Vulkan ICD).
# CUDA-only containers often have compute libs but no Vulkan driver — see
# scripts/debug_record_video.sh header for details.

set -euo pipefail

PACKAGES=(
  libglu1-mesa
  libxt6
  libx11-6
  libxext6
  libsm6
  libvulkan1
  vulkan-tools
)

if dpkg --audit 2>/dev/null | grep -q .; then
  echo "[WARN] Broken apt packages detected (often from a failed libnvidia-gl install in GPU containers)."
  echo "       Attempting cleanup of partial NVIDIA user-space packages..."
  DEBIAN_FRONTEND=noninteractive dpkg --remove --force-remove-reinstreq \
    nvidia-kernel-common-580 libnvidia-common-580 libnvidia-egl-wayland1 libnvidia-gl-580 \
    libnvidia-compute-580 nvidia-firmware-580-580.159.03 2>/dev/null || true
  DEBIAN_FRONTEND=noninteractive dpkg --configure -a 2>/dev/null || true
  echo "       Do NOT apt-install libnvidia-gl-* inside nvidia-container mounts — use graphics caps instead."
  echo ""
fi

echo "Installing Isaac headless dependencies..."
DEBIAN_FRONTEND=noninteractive apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y "${PACKAGES[@]}"

echo ""
echo "Installed. Verify:"
ldconfig -p | grep -E 'libGLU|libXt' || true

echo ""
echo "Vulkan check (video recording needs this to succeed):"
if [[ -f /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0 ]] && [[ ! -f /usr/share/vulkan/icd.d/nvidia_icd.json ]]; then
  echo "Creating NVIDIA Vulkan ICD (libGLX_nvidia.so.0)..."
  cat > /usr/share/vulkan/icd.d/nvidia_icd.json <<'EOF'
{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "libGLX_nvidia.so.0",
        "api_version": "1.3.0"
    }
}
EOF
fi

CAPS="${NVIDIA_DRIVER_CAPABILITIES:-}"
if [[ -z "${CAPS}" || ( "${CAPS}" != *graphics* && "${CAPS}" != "all" ) ]]; then
  echo ""
  if [[ -z "${CAPS}" ]]; then
    echo "[WARN] NVIDIA_DRIVER_CAPABILITIES is unset."
    echo "       Many GPU containers default to compute,utility only (no Vulkan/graphics)."
  else
    echo "[WARN] NVIDIA_DRIVER_CAPABILITIES=${CAPS}"
  fi
  echo "       Headless video needs graphics. Restart the container with e.g.:"
  echo "         -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics"
  echo "       or: -e NVIDIA_DRIVER_CAPABILITIES=all"
fi

if vulkaninfo --summary 2>&1 | head -5; then
  echo "Vulkan OK — headless video may work with --enable_cameras --video"
else
  echo "Vulkan FAILED — physics/training still works; video recording likely unavailable"
  echo "  Common causes in GPU containers:"
  echo "    1) NVIDIA_DRIVER_CAPABILITIES missing 'graphics' (see warning above)"
  echo "    2) Only CUDA compute libs mounted, no Vulkan ICD / GL stack"
  echo "  Options:"
  echo "    1) Restart container with NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics"
  echo "    2) Run video on a full Isaac Sim workstation (local GPU + display)"
  echo "    3) Debug without video: eval logs, play.py without --video/--enable_cameras"
fi
