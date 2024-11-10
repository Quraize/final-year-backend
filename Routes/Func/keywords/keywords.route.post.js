import express from "express";
import Keywords from "../../../Controllers/Funcs/Keywords/keywords.post.controller.js";

const keywordsRouter = express.Router();
keywordsRouter.post('/keywords', Keywords);

export default keywordsRouter;