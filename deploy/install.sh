#!/usr/bin/env bash
# Instala face-presence en un LXC Debian/Ubuntu usando venv + systemd (sin Docker).
set -euo pipefail

APP_DIR="/opt/face-presence"
SERVICE_USER="face-presence"

if [ "$(id -u)" -ne 0 ]; then
  echo "Ejecuta este script como root (sudo)." >&2
  exit 1
fi

echo "==> Instalando dependencias del sistema"
apt-get update
apt-get install -y --no-install-recommends \
  python3 python3-venv python3-pip \
  libgl1 libglib2.0-0 ffmpeg git

if ! id "$SERVICE_USER" >/dev/null 2>&1; then
  echo "==> Creando usuario de servicio $SERVICE_USER"
  useradd --system --home-dir "$APP_DIR" --shell /usr/sbin/nologin "$SERVICE_USER"
fi

echo "==> Copiando proyecto a $APP_DIR"
mkdir -p "$APP_DIR"
rsync -a --exclude 'data' --exclude '.git' --exclude 'venv' ./ "$APP_DIR/"

echo "==> Creando entorno virtual"
python3 -m venv "$APP_DIR/venv"
"$APP_DIR/venv/bin/pip" install --upgrade pip
"$APP_DIR/venv/bin/pip" install -r "$APP_DIR/requirements.txt"

mkdir -p "$APP_DIR/data/media" "$APP_DIR/data/models"

if [ ! -f "$APP_DIR/.env" ]; then
  echo "==> Creando .env desde .env.example (edita RTSP_URL y credenciales de Telegram)"
  cp "$APP_DIR/.env.example" "$APP_DIR/.env"
fi

chown -R "$SERVICE_USER:$SERVICE_USER" "$APP_DIR"

echo "==> Precargando modelo buffalo_sc (InsightFace)"
sudo -u "$SERVICE_USER" "$APP_DIR/venv/bin/python" - <<'PYEOF'
from insightface.app import FaceAnalysis
fa = FaceAnalysis(name="buffalo_sc", root="/opt/face-presence/data/models")
fa.prepare(ctx_id=-1)
PYEOF

echo "==> Inicializando base de datos"
sudo -u "$SERVICE_USER" "$APP_DIR/venv/bin/python" "$APP_DIR/scripts/init_db.py"

echo "==> Instalando servicio systemd"
cp "$APP_DIR/deploy/face-presence.service" /etc/systemd/system/face-presence.service
systemctl daemon-reload
systemctl enable face-presence

echo ""
echo "Listo. Antes de arrancar:"
echo "  1. Edita $APP_DIR/.env (RTSP_URL, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)"
echo "  2. Prueba el stream: sudo -u $SERVICE_USER $APP_DIR/venv/bin/python $APP_DIR/scripts/test_rtsp.py"
echo "  3. Arranca: systemctl start face-presence"
echo "  4. Logs: journalctl -u face-presence -f"
