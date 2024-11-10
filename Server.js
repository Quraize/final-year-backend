import express from 'express';
import dotenv from 'dotenv';
import cors from 'cors';
import connectDB from './database/dbCnnect.js';
//pipelines

//conversion funcs imports
import getImgRoute from './Routes/Func/TextToImage/TextToImage.post.route.js';
import getSpeechRoute from './Routes/Func/TextToSpeech/TextToAudio.post.route.js';
import sentiAnalysisRouter from './Routes/Func/SentimentAnalysis/SentimentAnalysis.post.route.js';
import languageDetectionRouter from './Routes/Func/LanguageDetection/LanguageDetect.route.post.js';
import translateRouter from './Routes/Func/Translation/translation.post.route.js';
import SummarizeRouter from './Routes/Func/Summarize/Summarize.post.route.js';
import keywordsRouter from './Routes/Func/keywords/keywords.route.post.js';

//environment variable configurations "secrets"
dotenv.config();
const port = process.env.PORT || 8080;

//creating the app and setting up the rules and regulations
const app = express();
app.use(cors());
app.use(express.json({limit: '50mb'}));
connectDB();



//testing
app.get('/', async(req, res)=>{
    res.send("Hello from Conversion func");
})

//conversion funcs
app.use('/conversion', getImgRoute);
app.use('/conversion', getSpeechRoute);
app.use('/conversion', translateRouter);

//analysis funcs
app.use('/analysis', sentiAnalysisRouter);
app.use('/analysis', languageDetectionRouter);

//Evaluation
app.use('/evaluation', SummarizeRouter);
app.use('/evaluation', keywordsRouter);


//middleware
app.use((err, req, res, next) => {
    const statusCode = err.statusCode || 500;
    const message = err.message || 'Internal Server Error';
    return res.status(statusCode).json({
        success: false,
        message,
        statusCode,
    });
})

//initiating the server
app.listen(port, ()=>{
    console.log(`Server is listening on the port ${port}`);
})