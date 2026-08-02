# camdetect

Detector de personas por rostro (sin servicios de IA en la nube, todo local
y en CPU) que notifica por Telegram. Usa
[YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
(via `opencv-contrib-python`) para detectar rostros y
[EdgeFace](https://github.com/otroshi/edgeface) (variante XS gamma-06, via
ONNX Runtime) para generar los embeddings — ambos modelos livianos pensados
para CPU, sin dependencias de GPU ni llamadas externas.

## Cómo funciona

1. **`pipeline.py`** lee la cámara RTSP, detecta rostros 1-2 veces por
   segundo, y compara cada uno contra la base de conocidos.
   - Si matchea a alguien conocido → notifica por Telegram y loguea el
     evento.
   - Si no matchea a nadie → lo guarda (con dedupe, para no llenar el
     disco de fotos casi iguales) en el pool de "pendientes", y también
     avisa "desconocido detectado" (con su propio cooldown).
2. **`cluster_pending.py`** agrupa por similitud los rostros pendientes
   (probablemente la misma persona aparece varias veces). Se puede correr
   a mano o dejar con el timer de systemd cada 15 min.
3. **`review_app.py`** es una web local (Flask) donde ves cada grupo,
   le ponés nombre (pasa a la base de conocidos) o lo descartás.
4. **`enroll.py`** es opcional, por si preferís cargar a alguien a mano
   desde fotos ya existentes en vez de esperar a que el sistema lo detecte.

## Instalación en el LXC (Debian/Ubuntu)

```bash
apt update && apt install -y python3 python3-venv python3-pip ffmpeg libgl1

mkdir -p /opt/camdetect
# copiar todo el contenido de este proyecto a /opt/camdetect

cd /opt/camdetect
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

bash download_models.sh   # baja YuNet y EdgeFace a models/
```

Editá `config.yaml`:
- `rtsp.url`: la URL RTSP de tu cámara.
- `notify.telegram_bot_token` / `telegram_chat_id`: creá un bot con
  [@BotFather](https://t.me/BotFather), y conseguí tu chat_id hablándole
  al bot y consultando `https://api.telegram.org/bot<TOKEN>/getUpdates`.

### Usuario dedicado (recomendado)

```bash
useradd -r -s /usr/sbin/nologin -d /opt/camdetect camdetect
chown -R camdetect:camdetect /opt/camdetect
```

### Servicios systemd

```bash
cp systemd/*.service systemd/*.timer /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now camdetect-pipeline.service
systemctl enable --now camdetect-review.service
systemctl enable --now camdetect-cluster.timer
```

- Pipeline: corre siempre, mirando la cámara.
- Review app: `http://IP_DEL_LXC:5000` para etiquetar clusters.
- Cluster timer: agrupa los pendientes cada 15 min automáticamente.

Ver logs:
```bash
journalctl -u camdetect-pipeline -f
journalctl -u camdetect-review -f
```

## Migrar de un modelo de embeddings a otro

Si ya tenías `camdetect` corriendo (ej. con SFace) y actualizás el código a
una versión que cambia el modelo de reconocimiento (ej. a EdgeFace), los
embeddings guardados en la base quedan en un espacio vectorial distinto y
ya no son comparables — hay que recalcularlos, si no vas a tener matches
raros o gente conocida que deja de reconocerse.

```bash
systemctl stop camdetect-pipeline camdetect-review   # para que no escriba mientras migrás
bash download_models.sh                              # baja el modelo nuevo
python src/migrate_embeddings.py
systemctl start camdetect-pipeline camdetect-review
```

El script recalcula los embeddings de `known_faces` y `pending_faces` a
partir de las fotos ya guardadas en disco (no hace falta re-enrolar a
mano). Si a alguien no se le puede recalcular (falta el archivo o ya no se
detecta un rostro adentro), se descarta y el script te dice a quién hay que
reenrolar. Los clusters pendientes sin revisar quedan reseteados — corré
`cluster_pending.py` de nuevo cuando quieras volver a ver la webUI de
revisión.

## Uso del día a día

1. El sistema detecta gente nueva sola, sin que hagas nada.
2. Cada tanto (o cuando te llega el aviso de "desconocido detectado"),
   entrás a `http://IP_DEL_LXC:5000`, ves las fotos agrupadas, y le ponés
   nombre a la persona (o la descartás si no te interesa, ej. un repartidor
   ocasional).
3. La próxima vez que esa persona pase por la cámara, ya la va a reconocer
   y notificar por su nombre.

## Ajuste de umbrales

Todo se ajusta en `config.yaml`:
- `matching.known_threshold`: subilo si te tira falsos positivos
  (confunde personas), bajalo si no reconoce bien a la misma persona.
- `pending.dedupe_min_distance`: subilo si guarda muy pocas variantes de
  la misma persona, bajalo si te llena el pool de fotos casi iguales.
- `clustering.eps`: si dos apariciones de la misma persona quedan en
  clusters separados, subilo un poco. Si te agrupa a dos personas
  distintas en un mismo cluster, bajalo.

## Recursos recomendados para el LXC

Para 1 cámara procesando ~1 frame/seg:
- CPU: 2 núcleos (containers unprivileged están bien, no hace falta pasar
  hardware, todo es por red vía RTSP)
- RAM: 2 GB (subir a 3-4 GB si vas a acumular muchos snapshots)
- Disco: 5-10 GB con limpieza periódica de `data/snapshots/` y
  `data/crops/pending/`

## Estructura

```
camdetect/
├── config.yaml
├── download_models.sh
├── requirements.txt
├── models/                  # YuNet + EdgeFace (.onnx), se descargan aparte
├── src/
│   ├── config.py
│   ├── db.py                 # SQLite: known_faces, pending_faces, clusters, events
│   ├── face_engine.py        # YuNet (deteccion) + EdgeFace (embeddings) wrapper
│   ├── capture.py            # lector RTSP en hilo separado
│   ├── telegram_notify.py
│   ├── pipeline.py           # loop principal
│   ├── cluster_pending.py    # DBSCAN sobre pendientes
│   ├── review_app.py         # Flask: etiquetar/descartar clusters
│   ├── enroll.py             # enroll manual opcional desde fotos
│   └── migrate_embeddings.py # recalcula embeddings tras cambiar de modelo
├── templates/                # HTML de la review app
├── systemd/                  # units para pipeline, review app y timer de clustering
└── data/                     # DB sqlite, crops, snapshots (se crea solo)
```
