const jwt = require("jsonwebtoken");
const fs = require("fs");
const path = require("path");

function readUsers(userFile){
    try{
        const userInfo = fs.readFileSync(userFile, 'utf-8');
        return JSON.parse(userInfo);
    }
    catch(e){
        console.error('Error reading users file:', e);
        return [];
    }
}

function writeUsers(userFile, data){
    try{
        const users = readUsers(userFile);
        fs.writeFileSync(userFile, JSON.stringify(data, null, 2));
        return false;
    }
    catch(e){
        console.error('Error reading users file:', e);
        return false;
    }
}

// verify user token
function verifyUser(req, userFile, secret){
    try{
        const authHeader = req.get("Authorization");
        const userData = readUsers(userFile);

        if (!authHeader) {
            return {
                status: "error",
                message: "Authorization header missing"
            };
        }

        const token = authHeader.split(' ')[1];
        const decoded = jwt.verify(token, secret);

        const found = userData.users.find(item => item.uid == decoded.uid);

        if(found){
            return {
                status: "true",
                message: found
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

module.exports = { verifyUser, readUsers, writeUsers };