import torch
from flask import Flask, request, jsonify
import soundfile as sf
import io
from aksharamukha import transliterate
import base64
from threading import Lock
from pysentimiento import create_analyzer
from transformers import pipeline, MBartForConditionalGeneration, MBart50TokenizerFast, T5Tokenizer, T5ForConditionalGeneration
from language_mapping import LANGUAGE_NAME_MAPPING, LANGUAGE_CODES
from functools import lru_cache
import base64
import threading
import time
from diffusers import StableDiffusionPipeline
from flask_socketio import SocketIO, emit


app = Flask(__name__)
socketio = SocketIO(app, async_mode="threading")  # Enable asynchronous communication

# Detect if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dictionary to cache loaded TTS model and some constants to create Speech
models_cache = {}
models_lock = Lock()

sample_rate = 48000
put_accent = True
put_yo = True

# Dictionary to cache sentiment analyzer
analyzer_cache = {}
analyzer_lock = Lock()

# Dictionary to cache language detection model
language_detection_cache = {}
language_detection_lock = Lock()

# Dictionary to cache summarizer model
summarizer_cache = {}
summarizer_lock = Lock()

# Dictionary to cache translation model
translation_model_cache = {}
translation_model_lock = Lock()

keywords_model_cache = {}
keywords_model_lock = Lock()

# Cache Stable Diffusion model for reuse
stable_diffusion_cache = {}
stable_diffusion_lock = threading.Lock()

# Dictionary to store ongoing task results
task_results = {}
task_results_lock = threading.Lock()

##FUNCTION TO LOAD THE MODELS IN THE CACHE

# Function to Load model and caching that model for TTS
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
                model.to(device)  # Move the model to the GPU if available
                models_cache[key] = model
    return models_cache[key]

def get_analyzer(task, language):
    key = f"{task}_{language}"
    if key not in analyzer_cache:
        with analyzer_lock:
            if key not in analyzer_cache:   # Double-checked locking
                analyzer_cache[key] = create_analyzer(task=task, lang=language)
    return analyzer_cache[key]

def load_language_detection_model():
    key = "language_detection"
    if key not in language_detection_cache:
        with language_detection_lock:
            if key not in language_detection_cache:  # Double-checked locking
                model_ckpt = "papluca/xlm-roberta-base-language-detection"
                language_detection_cache[key] = pipeline(
                    "text-classification",
                    model=model_ckpt,
                    device=0 if device.type == "cuda" else -1,
                    batch_size=1  # Smaller batch size to manage memory
                )
    return language_detection_cache[key]

def load_summarizer_model():
    key = "summarization"
    if key not in summarizer_cache:
        with summarizer_lock:
            if key not in summarizer_cache:  # Double-checked locking
                summarizer_cache[key] = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",
                    device=0 if device.type == "cuda" else -1,
                    batch_size=1  # Smaller batch size for memory efficiency
                )
    return summarizer_cache[key]

def generate_tts_audio(model, input_text, speaker):
    audio = model.apply_tts(
        text=input_text,
        speaker=speaker,
        sample_rate=sample_rate,
        put_accent=put_accent,
        put_yo=put_yo
    )
    return audio, sample_rate


# Load and cache the translation model
def load_translation_model():
    key = "translation"
    if key not in translation_model_cache:
        with translation_model_lock:
            if key not in translation_model_cache:
                model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt").to(device)
                tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
                translation_model_cache[key] = (model, tokenizer)
    return translation_model_cache[key]

# Translation function
def translate_text(article, src_lang, tgt_lang):
    model, tokenizer = load_translation_model()
    tokenizer.src_lang = src_lang

    # Encode the text and move tensors to CPU
    encoded_article = tokenizer(article, return_tensors="pt").to(device)
    
    # Temporarily move model to GPU if available
    if torch.cuda.is_available():
        model.to("cuda")
        encoded_article = encoded_article.to("cuda")

    # Generate the translation
    generated_tokens = model.generate(
        **encoded_article,
        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang]
    )

    # Move model back to CPU to save GPU memory
    model.to("cpu")

    # Decode the generated tokens
    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return translation[0]


def load_keywords_model():
    key = "keywords"
    if key not in keywords_model_cache:
        with keywords_model_lock:
            if key not in keywords_model_cache:  # Double-checked locking
                model = T5ForConditionalGeneration.from_pretrained("Voicelab/vlt5-base-keywords").to(device)
                tokenizer = T5Tokenizer.from_pretrained("Voicelab/vlt5-base-keywords", legacy=False)
                keywords_model_cache[key] = (model, tokenizer)
    return keywords_model_cache[key]

def load_stable_diffusion_model():
    key = "stable_diffusion"
    if key not in stable_diffusion_cache:
        with stable_diffusion_lock:
            if key not in stable_diffusion_cache:  # Double-checked locking
                model = StableDiffusionPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-2-1-base", torch_dtype=torch.float16
                )
                model.enable_model_cpu_offload()  # Enable CPU offloading for limited GPU VRAM
                model.safety_checker = None  # Disable safety checker
                stable_diffusion_cache[key] = model
    return stable_diffusion_cache[key]

def generate_image_task(task_id, prompt, num_inference_steps):
    """
    Background task to generate an image using Stable Diffusion.
    """
    try:
        model = load_stable_diffusion_model()
        start_time = time.time()
        
        # Generate the image
        image = model(prompt, num_inference_steps=num_inference_steps).images[0]
        end_time = time.time()

        # Convert the image to base64 format
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode("utf-8")

        # Save the result
        with task_results_lock:
            task_results[task_id] = {
                'status': 'completed',
                'image': image_base64,
                'execution_time': f"{end_time - start_time:.2f} seconds"
            }

        # Notify the client
        socketio.emit(f"task_update_{task_id}", {
            'status': 'completed',
            'image': image_base64,
            'execution_time': f"{end_time - start_time:.2f} seconds"
        })

    except Exception as e:
        with task_results_lock:
            task_results[task_id] = {
                'status': 'failed',
                'error': str(e)
            }
        
        # Notify the client of failure
        socketio.emit(f"task_update_{task_id}", {
            'status': 'failed',
            'error': str(e)
        })


##API ROUTES

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
            input_text = transliterate.process('Devanagari', 'ISO', input_text)
        elif indic_lang == "urdu":
            input_text = transliterate.process('Urdu', 'Latn', input_text)

    audio = None  # Initialize `audio` to ensure it has a default value
    
    try:
        audio, _ = generate_tts_audio(model, input_text, speaker)
        
        # Free GPU memory after audio generation
        model.to('cpu')  # Move model back to CPU
        torch.cuda.empty_cache()  # Clear GPU memory

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            torch.cuda.empty_cache()
            return jsonify({'status': 500, 'error': 'Out of memory. Try again with a smaller input or use CPU.'}), 500
        else:
            return jsonify({'status': 500, 'error': f'An error occurred: {str(e)}'}), 500
    
    if audio is None:
        return jsonify({'status': 500, 'error': 'Failed to generate audio due to an unexpected error.'}), 500

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

    try:
        result = analyzer.predict(text)
        
        # Free GPU memory after prediction
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear GPU memory

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            torch.cuda.empty_cache()
            return jsonify({'status': 500, 'error': 'Out of memory. Try again later or use smaller inputs.'}), 500
    
    return jsonify({
        'status': 200,
        'result': {
            'output': result.output,
            'probas': result.probas
        }
    })

@lru_cache(maxsize=500)  # Adjust cache size as needed
def detect_language_code(text):
    # Load or run your language detection model
    # Assuming you have a function `load_language_detection_model()`
    pipe = load_language_detection_model()
    result = pipe(text, top_k=1, truncation=True)
    return result[0]['label'], round(result[0]['score'], 4)

@app.route('/language-detection', methods=['POST'])
def language_detection():
    data = request.json
    texts = data.get("texts", [])

    if not texts or not isinstance(texts, list):
        return jsonify({'status': 400, 'error': 'Invalid input. Provide a list of texts.'}), 400

    response = []
    for text in texts:
        # Cache language detection calls to avoid redundant processing
        predicted_language_code, score = detect_language_code(text)
        
        # Use full language name if in our predefined mappings
        predicted_language_full_name = LANGUAGE_NAME_MAPPING.get(predicted_language_code, predicted_language_code)

        response.append({
            'text': text,
            'predicted_language': predicted_language_full_name,  # Use the full name
            'score': score
        })

    return jsonify({
        'status': 200,
        'results': response
    })


@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    article = data.get("article", "")
    max_length = int(data.get("max_length", 30))
    min_length = int(data.get("min_length", 30))

    if not article:
        return jsonify({'status': 400, 'error': 'Invalid input. Provide the text to summarize.'}), 400

    summarizer = load_summarizer_model()

    try:
        summary = summarizer(article, max_length=max_length, min_length=min_length, do_sample=False)
        
        # Free GPU memory after summarization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear GPU memory

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            torch.cuda.empty_cache()
            return jsonify({'status': 500, 'error': 'Out of memory. Try again with shorter text or use CPU.'}), 500

    return jsonify({
        'status': 200,
        'summary': summary[0]['summary_text']
    })


@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    article = data.get("article", "")
    src_lang = data.get("src_lang", "")
    tgt_lang = data.get("tgt_lang", "")

    # Validate input
    if not article or not src_lang or not tgt_lang:
        return jsonify({'status': 400, 'error': 'Invalid input. Provide article, src_lang, and tgt_lang.'}), 400

    try:
        # Translate the text
        translation = translate_text(article, src_lang, tgt_lang)
        
        # Free GPU memory after translation if CUDA is available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            torch.cuda.empty_cache()
            return jsonify({'status': 500, 'error': 'Out of memory. Try again with shorter text or use CPU.'}), 500
        else:
            return jsonify({'status': 500, 'error': f'An error occurred: {str(e)}'}), 500

    return jsonify({
        'status': 200,
        'translation': translation
    })

@app.route('/generate-keywords', methods=['POST'])
def generate_keywords():
    data = request.json
    article = data.get("article", "")
    max_length = data.get("max_length", 50)
    task_prefix = "Keywords: "

    if not article:
        return jsonify({'status': 400, 'error': 'Invalid input. Provide the text to generate keywords.'}), 400

    model, tokenizer = load_keywords_model()

    try:
        input_sequence = task_prefix + article
        input_ids = tokenizer(
            input_sequence, return_tensors="pt", truncation=True,
            max_length=tokenizer.model_max_length
        ).input_ids.to(device)

        output = model.generate(
            input_ids,
            max_length=max_length,
            no_repeat_ngram_size=3,
            num_beams=4
        )

        predicted = tokenizer.decode(output[0], skip_special_tokens=True)

        # Free GPU memory after generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            torch.cuda.empty_cache()
            return jsonify({'status': 500, 'error': 'Out of memory. Try again with shorter text or use CPU.'}), 500
        else:
            return jsonify({'status': 500, 'error': f'An error occurred: {str(e)}'}), 500

    return jsonify({
        'status': 200,
        'keywords': predicted
    })

@app.route('/generate-image', methods=['POST'])
def generate_image():
    """
    API endpoint to start an image generation task.
    """
    data = request.json
    prompt = data.get("prompt", "An artistic painting of a futuristic cityscape")
    num_inference_steps = int(data.get("num_inference_steps", 50))

    if not prompt:
        return jsonify({'status': 400, 'error': 'Invalid input. Provide a valid text prompt.'}), 400

    # Generate a unique task ID
    task_id = f"task_{int(time.time() * 1000)}"

    # Mark the task as in progress
    with task_results_lock:
        task_results[task_id] = {'status': 'in_progress'}

    # Start the background task
    threading.Thread(target=generate_image_task, args=(task_id, prompt, num_inference_steps)).start()

    return jsonify({
        'status': 200,
        'task_id': task_id,
        'message': 'Image generation task started. Use WebSocket or polling to get updates.'
    })

@app.route('/task-status/<task_id>', methods=['GET'])
def task_status(task_id):
    """
    API endpoint to check the status of an image generation task.
    """
    with task_results_lock:
        if task_id not in task_results:
            return jsonify({'status': 404, 'error': 'Task not found'}), 404
        
        return jsonify(task_results[task_id])

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
