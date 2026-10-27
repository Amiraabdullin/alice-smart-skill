import os
import re
import time
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

SYSTEM_PROMPT = (
    "Ты голосовой помощник 'Супер мозг' в умной колонке Яндекс Станция. "
    "Отвечай кратко, понятно и по делу: максимум 1-3 предложения (до 45 слов). "
    "Твой ответ будет озвучен Алисой вслух. "
    "Категорически запрещено использовать: markdown-разметку (никаких звездочек *, решеток #, "
    "тире, списков, кода), смайлики, таблицы и спецсимволы. "
    "Используй только грамотный живой разговорный русский язык."
)

ACTIVATION_PHRASES = sorted([
    "запусти навык супер мозг и спроси",
    "запусти навык супер мозг и скажи",
    "запусти навык супер мозг",
    "открой навык супер мозг и спроси",
    "открой навык супер мозг",
    "включи навык супер мозг и спроси",
    "включи навык супер мозг",
    "спроси у навыка супер мозг",
    "попроси навык супер мозг",
    "спроси у супер мозга",
    "спроси супер мозг",
    "попроси супер мозг",
    "скажи супер мозгу",
    "навык супер мозг",
    "запусти супер мозг",
    "открой супер мозг",
    "включи супер мозг",
    "супер мозг",
    "запусти навык чат гпт и спроси",
    "запусти навык чат гпт",
    "открой навык чат гпт",
    "включи навык чат гпт",
    "запусти чат гпт",
    "открой чат гпт",
    "включи чат гпт",
    "попроси чат гпт",
    "спроси у чат гпт",
    "чат гпт",
    "запусти навык",
    "открой навык",
    "включи навык",
    "поговори с супер мозгом",
    "позови супер мозг",
    "позови чат гпт",
], key=len, reverse=True)

EXIT_WORDS = {
    "стоп", "хватит", "выход", "отмена", "пока", "закройся", "отбой", "выключись"
}

def clean_activation_prefix(text: str) -> str:
    cleaned = (text or "").strip()
    low = cleaned.lower()
    for phrase in ACTIVATION_PHRASES:
        if low.startswith(phrase):
            cleaned = cleaned[len(phrase):].strip(" ,?!")
            low = cleaned.lower()
            if low.startswith("и "):
                cleaned = cleaned[2:].strip(" ,?!")
            break
    return cleaned

def clean_speech_text(text: str) -> str:
    """Удаляет markdown, спецсимволы и форматирует текст для красивой начитки Алисой."""
    if not text:
        return ""
    # Удаляем маркеры заголовков, жирности, курсива, кода
    text = re.sub(r'#+\s*', '', text)
    text = re.sub(r'\*{1,3}', '', text)
    text = re.sub(r'_{1,3}', '', text)
    text = re.sub(r'`{1,3}[^`]*`{1,3}', '', text)
    text = re.sub(r'`', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # [текст](ссылка) -> текст
    text = re.sub(r'[\r\n]+', ' ', text)  # переносы строк в пробелы
    text = re.sub(r'\s{2,}', ' ', text)   # двойные пробелы
    text = text.strip()

    # Ограничение Яндекс Диалогов: максимум 1024 символа
    if len(text) > 950:
        text = text[:940].rsplit(' ', 1)[0] + '...'
    return text

def ask_groq(prompt: str, api_key: str) -> str:
    """Groq API — сверхбыстрый инференс (отклик 200-400 мс)."""
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 150,
            "temperature": 0.7
        },
        timeout=3.0
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

def ask_gemini(prompt: str, api_key: str) -> str:
    """Google Gemini Flash Latest."""
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent"
    full_prompt = (
        f"Ты голосовой помощник Алиса. Ответь на вопрос кратко, простыми словами, "
        f"максимум 1-2 предложения, без markdown, звездочек и списков: {prompt}"
    )
    resp = requests.post(
        url,
        headers={
            "Content-Type": "application/json",
            "X-goog-api-key": api_key
        },
        json={
            "contents": [
                {
                    "parts": [{"text": full_prompt}]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": 120,
                "temperature": 0.7
            }
        },
        timeout=3.0
    )
    if resp.status_code == 200:
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    else:
        err_json = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
        msg = err_json.get("error", {}).get("message", resp.text[:150])
        raise RuntimeError(f"Gemini ({resp.status_code}): {msg}")

def ask_openai_compatible(prompt: str, api_key: str, base_url: str, model: str) -> str:
    """Универсальный клиент для OpenAI, DeepSeek, OpenRouter."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 150,
            "temperature": 0.7
        },
        timeout=3.0
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

def generate_ai_response(user_query: str) -> str:
    """Опрашивает доступную нейросеть с тайм-аутом, безопасным для Алисы."""
    deepseek_key = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
    gemini_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    groq_key = (os.getenv("GROQ_API_KEY") or "").strip()
    openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    hf_key = (os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN") or "").strip()

    try:
        # Приоритет 1: DeepSeek (пополненный баланс, топовая модель DeepSeek-V3, сверхбыстрый отклик)
        if deepseek_key:
            return ask_openai_compatible(user_query, deepseek_key, "https://api.deepseek.com", "deepseek-chat")

        if gemini_key:
            return ask_gemini(user_query, gemini_key)
        elif groq_key:
            return ask_groq(user_query, groq_key)
        elif hf_key:
            return ask_openai_compatible(user_query, hf_key, "https://router.huggingface.co/v1", "Qwen/Qwen2.5-72B-Instruct")
        elif openai_key:
            return ask_openai_compatible(user_query, openai_key, "https://api.openai.com/v1", "gpt-4o-mini")
        else:
            # Автоматический шлюз без ключей (на сервере Vercel работает без VPN и без регистрации)
            try:
                resp = requests.post(
                    "https://text.pollinations.ai/",
                    json={
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_query}
                        ],
                        "model": "openai"
                    },
                    timeout=2.6
                )
                if resp.status_code == 200 and resp.text.strip():
                    return resp.text.strip()
            except Exception as ex:
                print(f"Zero-key gateway error: {ex}")
            
            return "Я на связи! Навык работает, но бесплатный сервер сейчас немного перегружен. Задайте вопрос еще раз."
    except requests.exceptions.Timeout:
        return "Нейросеть не успела ответить за пару секунд. Пожалуйста, повторите вопрос."
    except Exception as e:
        err_msg = str(e)
        if hasattr(e, 'response') and e.response is not None:
            err_msg += f": {e.response.text[:200]}"
        print(f"Error querying LLM: {err_msg}")
        if "402" in err_msg or "Insufficient Balance" in err_msg:
            return "На аккаунте DeepSeek нулевой баланс. Требуется пополнить баланс."
        if "401" in err_msg:
            return "Ошибка авторизации: ключ указан неверно."
        return "Произошла временная ошибка при обращении к нейросети. Попробуйте еще раз."

@app.route("/", methods=["GET"])
def healthcheck():
    return jsonify({
        "status": "ok",
        "service": "Alice Smart Skill Webhook",
        "llm_configured": bool(
            os.getenv("GROQ_API_KEY") or os.getenv("GEMINI_API_KEY") or 
            os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        )
    })

@app.route("/", methods=["POST"])
@app.route("/webhook", methods=["POST"])
def yandex_dialog():
    req_data = request.get_json(silent=True) or {}
    
    req = req_data.get("request", {})
    session = req_data.get("session", {})
    is_new_session = session.get("new", False)

    command = req.get("command", "").strip()
    original = req.get("original_utterance", "").strip()

    # Очищаем активационные префиксы
    query_from_command = clean_activation_prefix(command)
    query_from_original = clean_activation_prefix(original)
    user_query = query_from_command or query_from_original

    # СЦЕНАРИЙ 1: Запуск навыка ("Алиса, запусти навык супер мозг")
    # Возвращаем мгновенное приветствие с end_session: False
    if is_new_session and not user_query:
        return jsonify({
            "response": {
                "text": "Я на связи! Задайте мне любой вопрос, и я отвечу с помощью нейросети.",
                "tts": "Я на связи! Задайте мне любой вопрос, и я отвечу с помощью нейросети.",
                "end_session": False
            },
            "version": "1.0"
        })

    # СЦЕНАРИЙ 2: Команда выхода
    if user_query.lower() in EXIT_WORDS:
        return jsonify({
            "response": {
                "text": "До связи!",
                "tts": "До связи!",
                "end_session": True
            },
            "version": "1.0"
        })

    # Пустая фраза при продолжении диалога (пауза/тишина)
    if not user_query:
        return jsonify({
            "response": {
                "text": "Я вас слушаю. Задайте вопрос или скажите 'стоп'.",
                "tts": "Я вас слушаю. Задайте вопрос или скажите стоп.",
                "end_session": False
            },
            "version": "1.0"
        })

    # СЦЕНАРИЙ 3: Запрос к нейросети
    raw_answer = generate_ai_response(user_query)
    clean_answer = clean_speech_text(raw_answer)

    # ЛОГИКА end_session:
    # 1) Одноразовый вопрос ("Алиса, спроси у супер мозга...") -> is_new_session: True -> end_session: True (сразу освобождает колонку для музыки/таймеров)
    # 2) Внутри диалога ("Алиса, запусти навык...") -> is_new_session: False -> end_session: False (держит сессию открытой для продолжения беседы)
    should_end = True if is_new_session else False

    return jsonify({
        "response": {
            "text": clean_answer,
            "tts": clean_answer,
            "end_session": should_end
        },
        "version": "1.0"
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
