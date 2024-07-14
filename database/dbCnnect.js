import mongoose from "mongoose";

const connectDB = async () =>{
    try {
        const connection = await mongoose.connect(process.env.DB_URL);

        console.log(`Database connected: ${connection}`);
    } catch (error) {
        console.log(`Error: ${error.message}`);
        process.exit();
    }
}

export default connectDB;