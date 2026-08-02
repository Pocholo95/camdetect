#!/usr/bin/env bash
# Descarga el modelo de deteccion (YuNet, repo oficial de OpenCV Zoo) y el
# de embeddings (EdgeFace XS gamma-06, exportado a ONNX por yakhyo/edgeface-onnx
# a partir de los pesos oficiales de otroshi/edgeface). Son modelos livianos
# (unos cientos de KB a pocos MB), pensados para correr en CPU.
set -e

mkdir -p models
cd models

echo "Descargando YuNet (detector de rostros)..."
curl -L -o face_detection_yunet_2023mar.onnx \
  "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

echo "Descargando EdgeFace XS gamma-06 (embeddings de rostros)..."
curl -L -o edgeface_xs_gamma_06.onnx \
  "https://github.com/yakhyo/edgeface-onnx/releases/download/weights/edgeface_xs_gamma_06.onnx"

echo "Listo. Modelos en $(pwd)"
