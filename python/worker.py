import torch
from flask import Flask, request, jsonify
import soundfile as sf
import io
from aksharamukha import transliterate
import base64
from threading import Lock
from pysentimiento import create_analyzer

app = Flask(__name__)

# Dictionary to cache loaded TTS model and some constants to create Speech
models_cache = {}
models_lock = Lock()

sample_rate = 48000
put_accent = True
put_yo = True

# Dictionary to cache sentiment analyzer
analyzer_cache = {}
analyzer_lock = Lock()

#Function to Load model and caching that model for TTS
def load_model_TTS(language, model_id):
    key = f"{language}_{model_id}"
    if key not in models_cache:
        with models_lock:
            if key not in models_cache:  # Double-checked locking
                model, example_text = torch.hub.load(
                    repo_or_dir='snakers4/silero-models',
                    model='silero_tts',
                    language=language,
                    speaker=model_id)
                model.to(torch.device('cpu'))
                models_cache[key] = model
    return models_cache[key]

def get_analyzer(task, language):
    key = f"{task}_{language}"
    if key not in analyzer_cache:
        with analyzer_lock:
            if key not in analyzer_cache:   # Double-checked locking
                analyzer_cache[key] = create_analyzer(task=task, lang=language)
    return analyzer_cache[key]

@app.route('/generate-audio', methods=['POST'])
def generate_audio():
    data = request.json
    input_text = data.get("input_text", "")
    speaker = data.get("speaker", "en_99")
    model_id = data.get("model_id", "v3_en")
    language = data.get("language", "en")
    indic_lang = data.get("indic_lang", "")
    
    model = load_model_TTS(language, model_id)
    
    if language == 'indic':
        if indic_lang == "hindi":
            input_text= transliterate.process('Devanagari', 'ISO', input_text)
        elif indic_lang == "urdu":
            input_text = transliterate.process('Urdu', 'Latn', input_text)

    audio, _ = generate_tts_audio(model, input_text, speaker)
    
    # Save to a memory buffer
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format='WAV')
    buffer.seek(0)
    
    # Convert to base64
    audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    
    return jsonify({
        'status': 200,
        'audio': audio_base64
    })

def generate_tts_audio(model, input_text, speaker):
    audio = model.apply_tts(text=input_text,
                            speaker=speaker,
                            sample_rate=sample_rate,
                            put_accent=put_accent,
                            put_yo=put_yo)
    return audio, sample_rate

@app.route('/sentiment-analysis', methods=['POST'])
def sentiment_analysis():
    data = request.json
    language = data.get("language", "")
    text = data.get("text", "")
    type = data.get("type", "")

    if type == "sentiment":
        analyzer = get_analyzer("sentiment", language)
    elif type == "emotion":
        analyzer = get_analyzer("emotion", language)
    elif type == "hate_speech":
        analyzer = get_analyzer("hate_speech", language)

    result = analyzer.predict(text)
    
    return jsonify({
        'status': 200,
        'result': {
            'output':result.output,
            'probas':result.probas
        }
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)