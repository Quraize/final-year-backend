import express from "express";
import Summarizer from "../../../Controllers/Funcs/Summarizer/Summarizer.post.controller.js";

const SummarizeRouter = express.Router();
SummarizeRouter.post('/summarize', Summarizer);

export default SummarizeRouter;