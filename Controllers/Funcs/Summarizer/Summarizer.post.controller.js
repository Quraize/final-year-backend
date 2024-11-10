import axios from "axios";
import { errorHandler } from "../../../utils/error.handler.js";

const Summarizer = async (req, res, next) =>{
    const { article, maxLength, minLength } = req.body;

    // Validate input
    if (!article) {
        next(errorHandler(400, "Invalid input. Provide Text."));
        return;
    }

    try {
        // Send POST request to the Flask API
        const response = await axios.post('http://localhost:5000/summarize', {
            article: article,
            max_length: Number(maxLength) || 30,  // Ensure max_length is a number
            min_length: Number(minLength) || 30   // Ensure min_length is a number
        });

        // Handle response from Flask API
        if (response.data.status === 200) {
            res.status(200).json({
                status: 200,
                summary: response.data.summary
            });
        } else {
            res.status(response.data.status).json({
                status: response.data.status,
                error: response.data.error
            });
        }
    } catch (error) {
        // Handle possible errors from the request
        if (error.response) {
            // Handle specific status errors from Flask API response
            if (error.response.status === 500) {
                next(errorHandler(500, error.response.data.error || "Internal server error from Summarization service."));
            } else if (error.response.status === 400) {
                next(errorHandler(400, error.response.data.error || "Invalid input sent to Summarization service."));
            } else {
                next(errorHandler(error.response.status, error.response.data.error || "Unexpected error from Summarization service."));
            }
        } else {
            // Handle network or other unknown errors
            next(errorHandler(500, "Network error or service unavailable."));
        }
    }
}

export default Summarizer;