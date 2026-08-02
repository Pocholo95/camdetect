# face-presence

Sistema de presencia domestico: detecta caras en un stream RTSP, las identifica
contra una base de personas conocidas, registra rangos de presencia y notifica
por Telegram sin spamear.

Despliegue objetivo: contenedor LXC (Debian/Ubuntu), solo CPU, con **venv +
systemd** (no requiere Docker).

## Instalacion (LXC, venv + systemd)

Instalador automatico (crea usuario de servicio, venv, precarga el modelo,
inicializa la DB e instala el unit de systemd):

```bash
sudo bash deploy/install.sh
```

Luego:

```bash
# 1. Edita /opt/face-presence/.env (RTSP_URL, credenciales de Telegram)
sudo nano /opt/face-presence/.env

# 2. Verifica que el stream se lee antes de arrancar nada mas
sudo -u face-presence /opt/face-presence/venv/bin/python /opt/face-presence/scripts/test_rtsp.py

# 3. Arranca el servicio
sudo systemctl start face-presence
sudo systemctl status face-presence

# 4. Logs en vivo
sudo journalctl -u face-presence -f
```

La WebUI queda disponible en `http://<ip-del-lxc>:8000`.

### Instalacion manual (sin el script)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edita .env con tu RTSP_URL y credenciales de Telegram

python scripts/test_rtsp.py
python scripts/init_db.py

python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Para produccion, copia `deploy/face-presence.service` a
`/etc/systemd/system/`, ajusta las rutas (`WorkingDirectory`, `ExecStart`,
`EnvironmentFile`) y habilita el servicio con `systemctl enable --now
face-presence`.

## Docker (alternativa opcional)

El proyecto tambien incluye `Dockerfile` y `docker-compose.yml` por si se
prefiere ese camino, pero **no es el metodo de despliegue soportado en este
entorno** (LXC + systemd es el metodo principal).

## Configuracion

Ver `.env.example` para todas las variables (umbrales de matching, FPS,
cooldown de notificaciones, etc). La mayoria son editables en caliente desde
`/settings` en la WebUI sin reiniciar el servicio; cambiar `RTSP_URL` si
requiere reinicio del servicio.

## Estructura del proyecto

```
app/
├── main.py           # FastAPI, arranque, lifespan
├── config.py         # pydantic-settings
├── db.py             # esquema, migraciones, helpers
├── capture.py        # hilo RTSP + pipeline completo
├── quality.py         # filtros de calidad
├── recognizer.py     # InsightFace, cache de matriz, matching
├── tracker.py        # tracker IoU + votacion
├── presence.py        # maquina de estados, presence_logs
├── unknowns.py         # clustering de desconocidos
├── notifier.py         # Telegram: cooldown, batching, reintentos
├── routes/
├── templates/
└── static/
data/                 # gitignored: db, media, modelos
deploy/               # systemd unit + script de instalacion
scripts/
├── init_db.py
└── test_rtsp.py      # verifica el stream sin arrancar todo
tests/
```

## Pruebas

```bash
pip install -r requirements.txt
pytest
```

## El requisito mas importante: no spamear notificaciones

La unidad de notificacion es el **track**, nunca el frame. Una persona parada
frente a la camara 10 minutos genera exactamente 1 notificacion (ver
`tests/` para los casos minimos exigidos: antispam, estabilidad de
desconocidos, regla de margen, filtro de calidad y ciclo de presencia).
