import express from "express";
import translateText from "../../../Controllers/Funcs/Translation/translation.post.controller.js";

const translateRouter = express.Router();
translateRouter.post('/translate', translateText);

export default translateRouter;