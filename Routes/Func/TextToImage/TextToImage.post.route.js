import express from 'express';
import getImage from '../../../Controllers/Funcs/TextToImage/TextToImage.post.controller.js';

const getImgRoute = express.Router();

getImgRoute.post('/texttoimg', getImage);

export default getImgRoute;