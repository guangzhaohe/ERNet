# Retained code provenance

This inference reduction preserves the following attributions from the
original source used by ERNet. It does not assign a new license to upstream
code, model weights, or the supplied bear data.

- The triplane point encoder is adapted from ConvONet's LocalPoolPointnet:
  https://github.com/autonomousvision/convolutional_occupancy_networks/blob/838bea5b2f1314f2edbb68d05ebb0db49f1f3bd2/src/encoder/pointnet.py
- The convolutional U-Net is adapted from jaxony/unet-pytorch:
  https://github.com/jaxony/unet-pytorch/blob/master/model.py
- Transformer and bilinear sampling code retains the original Meta Platforms
  copyright attribution. The original source cites CoTracker:
  https://github.com/facebookresearch/co-tracker/blob/8d364031971f6b3efec945dd15c468a183e58212/cotracker/models/core/model_utils.py
- DT4D binary animation parsing and camera conventions follow the source's
  DeformingThings4D reference:
  https://github.com/rabbityl/DeformingThings4D/blob/7cb946173968d88419b7432139f4be682cf61622/code/anime_renderer.py
- KNN_CUDA is installed from a pinned community re-upload because the original
  unlimblue repository is unavailable. Its source and upstream attribution are
  maintained separately: https://github.com/altaykacan/KNN_CUDA_reborn

The original repository's `LEGAL.md` is retained unchanged.
