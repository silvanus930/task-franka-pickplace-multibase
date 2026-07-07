#!/usr/bin/env bash
# System libraries for Isaac Sim / Isaac Lab in headless Linux containers (incl. Lium pods).
#
# Fixes common startup errors:
#   libGLU.so.1 / libXt.so.6  — RTX/neuray material loading (training hang)
#   libvulkan1 / libegl1      — headless Vulkan / offscreen render for --video
#
# Lium / nvidia-container: set on pod create (not inside the container):
#   NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
#   or NVIDIA_DRIVER_CAPABILITIES=all

set -euo pipefail

PACKAGES=(
  libglu1-mesa
  libxt6
  libx11-6
  libxext6
  libsm6
  libvulkan1
  vulkan-tools
  libegl1
)

# leatherback-stack .venv is Python 3.11; many Lium images only ship 3.12.
if [[ -x /root/leatherback-stack/.venv/bin/python3.11 ]] && ! /root/leatherback-stack/.venv/bin/python3.11 -c "import encodings" 2>/dev/null; then
  echo "[INFO] Installing Python 3.11 (required by leatherback-stack/.venv on this image)..."
  PACKAGES+=(python3.11 python3.11-venv)
fi

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

# Headless Vulkan on nvidia-container (no X11): use EGL ICD, not GLX.
mkdir -p /usr/share/glvnd/egl_vendor.d /usr/share/vulkan/icd.d
if [[ -f /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0 ]]; then
  cat > /usr/share/glvnd/egl_vendor.d/10_nvidia.json <<'EOF'
{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "libEGL_nvidia.so.0",
        "api_version": "1.3.0"
    }
}
EOF
  echo "Configured headless EGL Vulkan ICD: /usr/share/glvnd/egl_vendor.d/10_nvidia.json"
  echo "Add to your shell or Lium template environment:"
  echo "  VK_ICD_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
fi

echo ""
echo "Installed. Verify:"
ldconfig -p | grep -E 'libGLU|libXt|libEGL' || true

CAPS="${NVIDIA_DRIVER_CAPABILITIES:-}"
if [[ -z "${CAPS}" || ( "${CAPS}" != *graphics* && "${CAPS}" != "all" ) ]]; then
  echo ""
  if [[ -z "${CAPS}" ]]; then
    echo "[WARN] NVIDIA_DRIVER_CAPABILITIES is unset."
    echo "       Many GPU containers default to compute,utility only (no Vulkan/graphics)."
  else
    echo "[WARN] NVIDIA_DRIVER_CAPABILITIES=${CAPS}"
  fi
  echo "       Set on Lium pod create: -e NVIDIA_DRIVER_CAPABILITIES=all"
fi

echo ""
echo "Vulkan check (export VK_ICD_FILENAMES for headless EGL ICD first):"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"
if vulkaninfo --summary 2>&1 | grep -q "deviceName.*NVIDIA"; then
  vulkaninfo --summary 2>&1 | grep "deviceName" || true
  echo "Vulkan OK — headless video may work with --enable_cameras --video"
else
  echo "Vulkan FAILED — physics/training still works; video recording likely unavailable"
  echo "  Options:"
  echo "    1) Lium: redeploy with NVIDIA_DRIVER_CAPABILITIES=all"
  echo "    2) export VK_ICD_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
  echo "    3) Debug without video: play.py without --video/--enable_cameras"
fi
