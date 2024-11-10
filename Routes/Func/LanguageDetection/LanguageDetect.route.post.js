import express from "express";
import languageDetect from "../../../Controllers/Funcs/LanguageDetector/LanguageDetect.post.controller.js";

const languageDetectionRouter = express.Router();
languageDetectionRouter.post('/language-detect', languageDetect);

export default languageDetectionRouter;