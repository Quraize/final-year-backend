import express from 'express';
import { signIn } from '../../../Controllers/authentication/Signin.post.controller.js';

const signInRouter = express.Router();

signInRouter.post('/signin', signIn);

export default signInRouter;