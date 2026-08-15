#!/usr/bin/env python3
"""Cosine-normalized wrist-only SurfEmb control; preserves the raw-dot runner."""

import sys

import train_surfemb_resnet_wrist_only_lnd_prerefine_augfix as augfix


def _append_default(flag, value):
    if flag not in sys.argv:
        sys.argv.extend((flag, value))


def main():
    _append_default(
        "--name",
        "surfemb_resnet_wristonly_lnd_prerefine_augfix_cosine_t01_p1024_b56_gpu0123",
    )
    _append_default("--surfemb_similarity", "cosine")
    _append_default("--surfemb_temperature", "0.1")
    augfix.main()


if __name__ == "__main__":
    main()
