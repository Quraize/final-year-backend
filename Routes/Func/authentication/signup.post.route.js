import express from "express";
import { signUp } from "../../../Controllers/authentication/Signup.post.controller.js";

const signUpRouter = express.Router();

// Sign-Up route
signUpRouter.post("/signup", signUp);

export default signUpRouter;
