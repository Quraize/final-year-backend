import axios from "axios";
import { errorHandler } from "../../../utils/error.handler.js";

const Keywords = async (req, res, next) =>{
    const { article } = req.body;

    // Validate input
    if (!article) {
        next(errorHandler(400, "Invalid input. Provide Text."));
        return;
    }

    try {
        // Send POST request to the Flask API
        const response = await axios.post('http://localhost:5000/generate-keywords', {
            article: article,
        });

        // Handle response from Flask API
        if (response.data.status === 200) {
            res.status(200).json({
                status: 200,
                keywords: response.data.keywords
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
                next(errorHandler(500, error.response.data.error || "Internal server error from keywords service."));
            } else if (error.response.status === 400) {
                next(errorHandler(400, error.response.data.error || "Invalid input sent to keywords service."));
            } else {
                next(errorHandler(error.response.status, error.response.data.error || "Unexpected error from keywords service."));
            }
        } else {
            // Handle network or other unknown errors
            next(errorHandler(500, "Network error or service unavailable."));
        }
    }
}

export default Keywords;