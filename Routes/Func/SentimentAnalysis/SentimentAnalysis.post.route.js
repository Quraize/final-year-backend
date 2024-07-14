import express from 'express';
import getSentiAnalysis from '../../../Controllers/Funcs/SentimentAnalysis/SentimentAnalysis.post.controller.js';

const sentiAnalysisRouter = express.Router();
sentiAnalysisRouter.post('/sentiment', getSentiAnalysis);

export default sentiAnalysisRouter;