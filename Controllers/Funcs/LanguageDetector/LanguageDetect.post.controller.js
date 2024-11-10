import axios from "axios";
import { errorHandler } from "../../../utils/error.handler.js";

const languageDetect = async (req, res, next) => {
    const { texts } = req.body;

    // Check if texts array is provided and valid
    if (!texts || !Array.isArray(texts)) {
        next(errorHandler(400, "Invalid input. Provide a list of texts."));
        return;
    }

    try {
        // Send POST request to Flask API
        const response = await axios.post("http://localhost:5000/language-detection", {
            texts: texts
        });

        // Check if Flask API returned success
        if (response.data.status !== 200) {
            next(errorHandler(503, "The server is unable to handle the request at the moment."));
            return;
        }

        // Extract and send results to client
        const report = response.data.results;
        res.status(200).json({ result: report });

    } catch (error) {
        // Handle possible errors from the request
        if (error.response) {
            // Handle specific status errors from Flask API response
            if (error.response.status === 500) {
                next(errorHandler(500, error.response.data.error || "Internal server error from language detection service."));
            } else if (error.response.status === 400) {
                next(errorHandler(400, error.response.data.error || "Invalid input sent to language detection service."));
            } else {
                next(errorHandler(error.response.status, error.response.data.error || "Unexpected error from language detection service."));
            }
        } else {
            // Handle network or other unknown errors
            next(errorHandler(500, "Network error or service unavailable."));
        }
    }
};

export default languageDetect;