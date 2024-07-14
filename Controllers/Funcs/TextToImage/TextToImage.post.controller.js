import OpenAI from 'openai';
import dotenv from 'dotenv';
import { hasAnyWord } from '../../../utils/suitable.prompt.check.js';
import { errorHandler } from '../../../utils/error.handler.js';

dotenv.config();

const openai = new OpenAI({
    apiKey:process.env.TextToImage_API_KEY,
});

const getImage = async (req, res, next) => {
    const {prompt} = req.body;

    if(!prompt){
        return next(errorHandler(400, 'Missing Prompt in your Request'));
    }
    try {
         if(hasAnyWord(prompt)){
            next(errorHandler(400, "Please provide a prompt minding the controversial criteria."))
            return;
        }
        const aiResponse = await openai.images.generate({
            prompt,
            n:1,
            size:'1024x1024',
            response_format:'b64_json',
          });
          if(aiResponse.status === 400){
            next(errorHandler(400, 'Please provide a prompt minding the controversial criteria.'))
            return;
          }
          const imageUrl = aiResponse.data[0].b64_json;
          res.status(200).json({message:'Image extracted successfully', photo:imageUrl});
    } catch (error) {
       next(error)
    }
}

export default getImage;