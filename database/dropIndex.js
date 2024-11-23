import mongoose from "mongoose";
import User from "../models/user.schema.js";
const dropIndex = async () => {
  try {
    // Connect to the MongoDB database
    await mongoose.connect("mongodb+srv://pharmaproj:pharmaproj@cluster0.a3wbiig.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0");

    // List current indexes
    const indexes = await User.collection.indexes();
    console.log("Current indexes:", indexes);

    // Drop the unique index on the username field
    await User.collection.dropIndex("username_1"); // Adjust "username_1" if your index name differs
    console.log("Unique index on username field has been removed.");
    
    // Close the connection
    mongoose.connection.close();
  } catch (error) {
    console.error("Error removing index:", error);
  }
};

dropIndex();