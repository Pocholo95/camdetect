"""Envio de notificaciones a Telegram usando la API HTTP directa (sin
libreria pesada). Requiere un bot creado con @BotFather y el chat_id
al que se le quiere avisar (puede ser tu chat personal o un grupo)."""
import requests

API_BASE = "https://api.telegram.org/bot{token}/{method}"


class TelegramNotifier:
    def __init__(self, token: str, chat_id: str, send_photo: bool = True, timeout: float = 10.0):
        self.token = token
        self.chat_id = chat_id
        self.send_photo_enabled = send_photo
        self.timeout = timeout

    def _url(self, method: str) -> str:
        return API_BASE.format(token=self.token, method=method)

    def send_text(self, text: str) -> bool:
        try:
            r = requests.post(
                self._url("sendMessage"),
                data={"chat_id": self.chat_id, "text": text},
                timeout=self.timeout,
            )
            return r.ok
        except requests.RequestException as e:
            print(f"[telegram] error enviando texto: {e}")
            return False

    def send_photo(self, image_path: str, caption: str = "") -> bool:
        if not self.send_photo_enabled:
            return self.send_text(caption)
        try:
            with open(image_path, "rb") as f:
                r = requests.post(
                    self._url("sendPhoto"),
                    data={"chat_id": self.chat_id, "caption": caption},
                    files={"photo": f},
                    timeout=self.timeout,
                )
            return r.ok
        except (requests.RequestException, OSError) as e:
            print(f"[telegram] error enviando foto: {e}")
            return self.send_text(caption)
