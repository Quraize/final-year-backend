import axios from "axios";
import { errorHandler } from "../../../utils/error.handler.js";

const getSentiAnalysis = async (req, res, next) =>{
    const { text, type} = req.body;

    const language = 'en';
    
    if(!language || !text || !type){
        next(errorHandler(400, "Missing Text, Lanuage, or Type"));
        return;
    }

    try {
        const response = await axios.post("http://localhost:5000/sentiment-analysis",{
            language: language,
            text: text,
            type: type
        })

        if(response.data.status !== 200){
            next(errorHandler(503, "The server is unable to handle the server at the moment."))
            return;
        }
        const report = response.data.result
        res.json({result: report})
    } catch (error) {
        
    }
}

export default getSentiAnalysis;