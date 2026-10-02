# selectbadge-llm

A tiny language model running **entirely on a physical [SELECT badge](https://badge.select/make)**. This ports the full FP32 TinyStories 260K transformer from [imac-llm](https://github.com/maddiedreese/imac-llm) to the [Pimoroni Tufty 2350](https://shop.pimoroni.com/products/tufty-2350), with an on-screen keyboard and a Supabase-inspired dark interface.

The [prebuilt firmware](firmware/selectbadge-llm.uf2) has been confirmed working on a physical SELECT badge. No phone, Wi-Fi, server, or cloud inference is involved.

## Flash it

1. Back up any apps or files on the badge that you want to keep. Flashing this UF2 replaces the running firmware.
2. Download [selectbadge-llm.uf2](firmware/selectbadge-llm.uf2).
3. Connect the badge over USB-C. Hold **BOOT/Home** and tap **RESET**. Your computer should show an `RP2350` drive.
4. Copy the UF2 to that drive. The badge reboots into the model app.

Pimoroni provides the [stock Tufty firmware](https://github.com/pimoroni/tufty2350/releases/latest) if you want to restore Badgeware later.

## Use it

The five front buttons drive the keyboard:

| Button | Keyboard | After generation |
| --- | --- | --- |
| A / B | Move left / right | B clears the prompt |
| Up / Down | Change rows | — |
| C | Select the highlighted key | Return to editing |
| Home | — | Stop generation and return to editing |

The bottom row has **SPACE**, **SHIFT**, **DEL**, and **SEND**. SHIFT applies to the next letter. Prompts are limited to 64 characters and 32 model tokens; the app generates up to 24 new tokens. This is a story continuation model, not an instruction-following assistant.

## What runs on the badge

- RP2350B Arm Cortex-M33 cores, 520 KB SRAM, 16 MB flash, 8 MB PSRAM, and a 320 × 240 colour display, as specified by [Pimoroni](https://shop.pimoroni.com/products/tufty-2350).
- The original 1,056,540-byte FP32 checkpoint and 6,227-byte tokenizer are embedded in the UF2 and read from execute-in-place flash.
- The transformer uses full RoPE, grouped-query attention, softmax, RMSNorm, and SwiGLU. Activations and the 64-position KV cache use SRAM.
- The native UI is built on [Pimoroni's badgeware-cpp framework](https://github.com/pimoroni/badgeware-cpp). Its Supabase brand green is `#3ECF8E`, alongside charcoal panels and light text.

The compiled firmware occupies about 1.2 MB of flash. The build has about 409 KB of static BSS and fits the badge's SRAM; actual free heap and stack space vary with runtime use.

## Build from source

You need CMake, an Arm GNU embedded toolchain **with newlib**, the [Pico SDK](https://github.com/raspberrypi/pico-sdk), and `picotool`. Set `PICO_SDK_PATH` to your SDK checkout, then run:

```sh
git clone https://github.com/maddiedreese/selectbadge-llm.git
cd selectbadge-llm
export PICO_SDK_PATH=/path/to/pico-sdk
./scripts/build.sh
```

The script checks out `pimoroni/badgeware-cpp` at commit `7d1ef0882c6e9dcf374542528f6f211e287b5afb`, initializes its PicoVector submodule, installs this app under `demos/badge_llm`, and builds `build/selectbadge-llm.uf2`. It does not modify the prebuilt UF2 in `firmware/`.

To integrate manually, use `python3 install.py /path/to/badgeware-cpp` and build the `badge_llm` target. The model assets in `demos/badge_llm/assets/` are the original little-endian files; no download or conversion is needed.

## Verification

- Cross-compiled as a native Tufty 2350 UF2 for the `rp2350-arm-s` target.
- Confirmed model and tokenizer shapes and checkpoint layout during app startup.
- Ran the model code with the embedded assets on a host computer; a prompt beginning “Once upon a time” generated a coherent continuation.
- Confirmed by the badge owner to work on physical SELECT hardware.

The compiled UF2 has SHA-256 `a04605cbdf9eed79df2096e44ec107cba3ffd4f714fa9022a4632b93c5100fc7`.

## Credits and licenses

Inference was adapted from [imac-llm](https://github.com/maddiedreese/imac-llm) and [llama2.c](https://github.com/karpathy/llama2.c). The checkpoint and tokenizer come from [karpathy/tinyllamas](https://huggingface.co/karpathy/tinyllamas) and were also used in [gbc-transformer](https://github.com/maddiedreese/gbc-transformer). The app uses [Pimoroni's badgeware-cpp](https://github.com/pimoroni/badgeware-cpp) at build time. The source and model lineage use MIT licensing; see [LICENSE](LICENSE), [BADGEWARE_LICENSE](BADGEWARE_LICENSE), and [KARPATHY_LICENSE](KARPATHY_LICENSE).
