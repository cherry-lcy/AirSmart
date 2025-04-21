const jwt = require("jsonwebtoken");

// verify user token
function verifyUser(req, user, secret){
    try{
        const authHeader = req.get("Authorization");
        if (!authHeader) {
            return res.status(401).json({
                status: "error",
                message: "Authorization header missing"
            });
        }

        const token = authHeader.split(' ')[1];
        const decoded = jwt.verify(token, secret);

        if(decoded.uid === user.uid){
            return {
                status:"true",
                message:""
            }
        }
        else{
            return {
                status:"false",
                message:"Invalid credentials"
            }
        }
    }
    catch(e){
        console.error("Profile error:", e);
        return {
            status:"error",
            message:e.message
        }
    }
}

module.exports = { verifyUser };