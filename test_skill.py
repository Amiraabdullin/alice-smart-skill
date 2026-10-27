import json
from main import app

def test_single_shot():
    client = app.test_client()
    
    # 1. Проверяем healthcheck
    res = client.get("/")
    print("Healthcheck status:", res.status_code, res.get_json())
    
    # 2. Имитируем запрос от Алисы: «Алиса, спроси у Джина, почему трава зеленая»
    mock_payload = {
        "meta": {
            "locale": "ru-RU",
            "timezone": "Europe/Moscow",
            "client_id": "ru.yandex.quasar.app/1.0"
        },
        "request": {
            "command": "почему трава зеленая",
            "original_utterance": "почему трава зеленая",
            "type": "SimpleUtterance"
        },
        "session": {
            "new": True,
            "message_id": 0,
            "session_id": "test-session-123",
            "user_id": "test-user-456"
        },
        "version": "1.0"
    }

    print("\nОтправка тестового запроса от Яндекс Станции...")
    res = client.post("/webhook", json=mock_payload)
    print("Код ответа:", res.status_code)
    data = res.get_json()
    print("Ответ для Алисы:")
    print(json.dumps(data, ensure_ascii=False, indent=2))
    
    assert data["response"]["end_session"] is True, "end_session должен быть True!"
    print("\n[OK] Тест пройден успешно! Сессия корректно завершается для освобождения музыки.")

if __name__ == "__main__":
    test_single_shot()
