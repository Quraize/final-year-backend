import express from 'express';
import getSpeech from '../../../Controllers/Funcs/TextToAudio/TextToAudio.post.controller.js';

const getSpeechRoute = express.Router();

getSpeechRoute.post('/texttospeech',getSpeech );

export default getSpeechRoute;