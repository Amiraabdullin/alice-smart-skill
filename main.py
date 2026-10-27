import os
import re
import time
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

SYSTEM_PROMPT = (
    "Ты умный голосовой ассистент для умной колонки Яндекс Алиса. "
    "Отвечай кратко, емко, естественным разговорным языком. "
    "Твой ответ будет озвучен голосом, поэтому КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО использовать: "
    "markdown-разметку (звездочки, решетки, тире, ссылки), списки с цифрами, "
    "смайлики, таблицы и любые спецсимволы. "
    "Длина ответа — максимум 2-3 предложения, если только тебя прямо не попросили рассказать подробно."
)

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
    """Groq API — сверхбыстрый бесплатный инференс (отклик 200-400 мс)."""
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 250,
            "temperature": 0.7
        },
        timeout=2.8
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

def ask_gemini(prompt: str, api_key: str) -> str:
    """Google Gemini API (быстрый бесплатный тариф)."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    resp = requests.post(
        url,
        json={
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": 250, "temperature": 0.7}
        },
        timeout=2.8
    )
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]

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
            "max_tokens": 250,
            "temperature": 0.7
        },
        timeout=2.8
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

def generate_ai_response(user_query: str) -> str:
    """Опрашивает доступную нейросеть с тайм-аутом, безопасным для Алисы."""
    groq_key = (os.getenv("GROQ_API_KEY") or "").strip()
    gemini_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    deepseek_key = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
    openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()

    try:
        if groq_key:
            return ask_groq(user_query, groq_key)
        elif gemini_key:
            return ask_gemini(user_query, gemini_key)
        elif deepseek_key:
            return ask_openai_compatible(user_query, deepseek_key, "https://api.deepseek.com", "deepseek-chat")
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
                    timeout=2.8
                )
                if resp.status_code == 200 and resp.text.strip():
                    return resp.text.strip()
            except Exception as ex:
                print(f"Zero-key gateway error: {ex}")
            
            return "Я на связи! Навык работает, но бесплатный сервер сейчас немного перегружен. Задайте вопрос еще раз."
    except requests.exceptions.Timeout:
        return "Нейросеть задумалась слишком долго. Пожалуйста, задайте вопрос ещё раз."
    except Exception as e:
        err_msg = str(e)
        if hasattr(e, 'response') and e.response is not None:
            err_msg += f": {e.response.text[:200]}"
        print(f"Error querying LLM: {err_msg}")
        # Если у DeepSeek закончился баланс или неверный ключ
        if "402" in err_msg or "Insufficient Balance" in err_msg:
            return "На аккаунте DeepSeek нулевой баланс. Для работы API требуется пополнить баланс или использовать другой ключ."
        if "401" in err_msg:
            return "Ошибка авторизации: ключ указан неверно. Проверьте правильность ключа в настройках."
        return f"Ошибка обращения: {err_msg[:200]}"

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
    user_query = req.get("command", "").strip() or req.get("original_utterance", "").strip()
    is_new_session = session.get("new", False)

    # Если пользователь просто открыл навык без вопроса ("Алиса, включи навык Умник")
    if is_new_session and not user_query:
        return jsonify({
            "response": {
                "text": "Я на связи! Задайте мне любой вопрос, и я отвечу с помощью нейросети.",
                "end_session": False
            },
            "version": "1.0"
        })

    # Если пользователь сказал команду выхода
    if user_query.lower() in ["стоп", "хватит", "выход", "отмена", "пока"]:
        return jsonify({
            "response": {
                "text": "До связи!",
                "end_session": True
            },
            "version": "1.0"
        })

    # Отправляем вопрос в LLM
    raw_answer = generate_ai_response(user_query)
    clean_answer = clean_speech_text(raw_answer)

    # end_session: True — главное правило!
    # Сессия сразу завершается, и колонка свободна для музыки и обычных дел.
    return jsonify({
        "response": {
            "text": clean_answer,
            "end_session": True
        },
        "version": "1.0"
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
