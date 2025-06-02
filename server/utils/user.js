const jwt = require("jsonwebtoken");
const userRepository = require("./../repositories/userRepository");

// verify user token
async function verifyUser(req, secret){
    try{
        const authHeader = req.get("Authorization");

        if (!authHeader) {
            return {
                status: "error",
                message: "Authorization header missing"
            };
        }

        const token = authHeader.split(' ')[1];
        const decoded = jwt.verify(token, secret);

        const found = await userRepository.getByUid(decoded.uid);
        const userData = found.data;
        console.log(decoded, userData);

        if(found.success && userData.uid === decoded.uid && userData.password === decoded.password){
            return {
                status: "true",
                message: userData
            }
        }
        else{
            return {
                status: "false",
                message: "Invalid credentials"
            }
        }
    }
    catch(e){
        console.error("Profile error:", e);
        return {
            status: "error",
            message: e.message
        }
    }
}

module.exports = {verifyUser};