# One-time Jetson setup (Orin Nano Super dev kit, JetPack 6.x)

## 1. Flash JetPack
- Download the latest JetPack 6.x SD-card image from NVIDIA's Jetson Orin Nano
  "Getting Started" page and write it to a 64 GB+ microSD (A2 speed class) with Balena Etcher.
- Older kits may need a firmware (QSPI) update before JetPack 6 boots. NVIDIA's getting-started
  guide covers this; if the screen stays blank on first boot, that's usually the cause.
- Better: put the OS on an NVMe SSD. Engine builds and model loading are much faster.

## 2. First boot
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip
sudo pip3 install jetson-stats      # gives you `jtop`
sudo reboot
```

## 3. Max performance mode (do this before every benchmark)
```bash
sudo nvpmodel -q          # show current mode
sudo nvpmodel -m 0        # highest mode (MAXN SUPER on Super-capable kits)
sudo jetson_clocks        # lock clocks at max
```
Record the power mode name. Benchmarks are only comparable at the same mode.

## 4. Docker
JetPack 6 ships with Docker and the NVIDIA container runtime. Check:
```bash
sudo docker run --rm --runtime=nvidia ultralytics/ultralytics:latest-jetson-jetpack6 \
  python3 -c "import torch; print(torch.cuda.is_available())"
```
It should print `True`. Using this container avoids the hardest part of Jetson work:
matching PyTorch/TensorRT wheels to your JetPack version.

## 5. Sanity check
Run `jtop` in a second terminal while benchmarking. It shows GPU load, temps and power draw.
If the board throttles (temps > ~85 °C), add a fan or lower the power mode.
