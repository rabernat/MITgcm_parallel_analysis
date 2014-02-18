#!/usr/bin/env bash
convert -shade 90x10 -normalize gray.tiff gray_shaded.tiff
convert gray_shaded.tiff color.tiff -compose overlay -composite theta_overlay.tiff

