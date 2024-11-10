import axios from "axios";
import { errorHandler } from "../../../utils/error.handler.js";

const translateText = async (req, res, next) => {
    const { article, src_lang, tgt_lang } = req.body;

    // Validate input fields
    if (!article || !src_lang || !tgt_lang) {
        next(errorHandler(400, "Invalid input. Provide Text, Source Language, and Target Language."));
        return;
    }

    try {
        // Send POST request to Flask API for translation
        const response = await axios.post("http://localhost:5000/translate", {
            article: article,
            src_lang: src_lang,
            tgt_lang: tgt_lang
        });

        // Check if Flask API returned a success status
        if (response.data.status !== 200) {
            next(errorHandler(503, "The server is unable to handle the request at the moment."));
            return;
        }

        // Extract and send translation to client
        const translation = response.data.translation;
        res.status(200).json({ translation: translation });

    } catch (error) {
        // Handle possible errors from the request
        if (error.response) {
            // Handle specific status errors from Flask API response
            if (error.response.status === 500) {
                next(errorHandler(500, error.response.data.error || "Internal server error from translation service."));
            } else if (error.response.status === 400) {
                next(errorHandler(400, error.response.data.error || "Invalid input sent to translation service."));
            } else {
                next(errorHandler(error.response.status, error.response.data.error || "Unexpected error from translation service."));
            }
        } else {
            // Handle network or other unknown errors
            next(errorHandler(500, "Network error or service unavailable."));
        }
    }
};

export default translateText;
