import axios from "axios";
import { errorHandler } from '../../../utils/error.handler.js';

const getSpeech = async (req, res, next) => {
  console.time('Request Processing Time');

  const { input_text, language, speaker } = req.body;

  const languageConfig = {
    urdu: { model_id: 'v4_indic', indic_lang: 'urdu', input_lang: 'indic' },
    hindi: { model_id: 'v4_indic', indic_lang: 'hindi', input_lang: 'indic' },
    en: { model_id: 'v3_en', input_lang: 'en' },
    ru: { model_id: 'v4_ru', input_lang: 'ru' },
    es: { model_id: 'v3_es', input_lang: 'es' },
    de: { model_id: 'v3_de', input_lang: 'de' },
    fr: { model_id: 'v3_fr', input_lang: 'fr' },
  };

  const config = languageConfig[language];

  if (!input_text || !config || !speaker) {
    return next(errorHandler(400, 'Missing Text, language, or Speaker in your Request'));
  }

  try {
    const response = await axios.post('http://localhost:5000/generate-audio', {
      input_text: input_text,
      language: config.input_lang,
      model_id: config.model_id,
      speaker: speaker,
      indic_lang: config.indic_lang || ''
    });

    
    if(response.data.status !== 200){
      next(errorHandler(503, "The server is unable to handle the server at the moment."));
      return;
    }

    const audioBase64 = response.data.audio;
    res.json({ audio: audioBase64 });
  } catch (error) {
    next(error);
  }
}

export default getSpeech;