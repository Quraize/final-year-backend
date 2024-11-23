import User from "../../models/user.schema.js";
import { errorHandler } from "../../utils/error.handler.js";
import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';

export const signIn = async (req, res, next) => {
    try {
      const { email, password } = req.body;
  
      // Check if user exists
      const user = await User.findOne({ email });
      if (!user) {
        return next(errorHandler(400, "User not found"));
      }
  
      // Check if password is correct
      const isPasswordValid = await bcrypt.compare(password, user.password);
      if (!isPasswordValid) {
        return next(errorHandler(400, "Invalid credentials"));
      }
  
      // Generate JWT Token
      const token = jwt.sign({ userId: user._id }, process.env.JWT_SECRET || "your_jwt_secret", { expiresIn: "1h" });
  
      // Send token as response
      res.status(200).json({
        success: true,
        message: "Login successful",
        token,
      });
    } catch (error) {
      console.error("Error during sign-in:", error);
      return next(errorHandler(500, "Server error"));
    }
  };