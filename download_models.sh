#!/usr/bin/env bash
# Descarga los modelos de deteccion (YuNet) y embeddings (SFace) desde el
# repo oficial de OpenCV Zoo. Son modelos livianos (unos cientos de KB a
# pocos MB), pensados para correr en CPU.
set -e

mkdir -p models
cd models

echo "Descargando YuNet (detector de rostros)..."
curl -L -o face_detection_yunet_2023mar.onnx \
  "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

echo "Descargando SFace (embeddings de rostros)..."
curl -L -o face_recognition_sface_2021dec.onnx \
  "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

echo "Listo. Modelos en $(pwd)"
