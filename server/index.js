const express = require("express");
const jwt = require("jsonwebtoken");
const crypto = require("crypto");
const path = require("path");
const {addRecord, getHistory, clearAllHistory} = require("./utils/historyStorage");
const {verifyUser} = require("./utils/user");
const userRepository = require("./repositories/userRepository");

const app = express();

const secret = crypto.randomBytes(32).toString('hex');
const USER_FILE = path.join(__dirname, "user.json");

app.use((req, res, next)=>{
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
    if (req.method === 'OPTIONS') {
        return res.sendStatus(200);
    }
    next();
})
app.use(express.urlencoded({extended:true}));
app.use(express.json());

let acStatus = {
    on:false,
    temperature:22,
    mode:"cool",
    fan:"auto",
    swing:true
}

// user authorization api
app.post('/v1/authorization', async (req, res)=>{
    const {username, password} = req.body;
    console.log("post: ", req.body);

    const userData = await userRepository.getByUsername(username);
    const user = userData.data;
    
    if(username === user.username && password === user.password){
        const token = jwt.sign({ uid: user.uid, password: user.password }, secret, {expiresIn: "4h"})
        res.send({
            status:"ok",
            data:{
                token: token,
                username: user.username,
                uid: user.uid
            },
            timestamp: new Date().toISOString()
        })
    }
    else{
        res.status(403).send({
            status:"error",
            data:"Either username or password is incorrect.",
            timestamp: new Date().toISOString()
        })
    }
})

// get user profile api
app.get('/v1/user/profile', async (req, res) => {
    const {status, message} = await verifyUser(req, secret);
    
    if (status === "true") {
        return res.json({
            status: "ok",
            data: {
                userInfo: {
                    uid: message.uid,
                    username: message.username,
                    email: message.email,
                    register_time: message.register_time,
                    role: message.role
                }
            },
            timestamp: new Date().toISOString()
        });
    }
    else if(status === "false"){
        return res.status(401).json({
            status: "error",
            data: "Invalid credentials",
            timestamp: new Date().toISOString()
        });
    }
    else if(status === "error"){
        return res.status(403).json({
            status: "error",
            data: message,
            timestamp: new Date().toISOString()
        });
    }  
});

app.post('/v1/change-password', (req, res)=>{
    const {username, currPassword, newPassword} = req.body;
    const {status, message} = verifyUser(req, secret);

    if(message === "true"){
        if(currPassword === newPassword){
            return res.status(403).send({
                status:"error",
                data:"New password cannot be the same as current password.",
                timestamp: new Date().toISOString()
            })
        }

        const userData = readUsers(USER_FILE);
        const user = userData.users.find(item=>item.username===username);

        user.password = newPassword;

        if(writeUsers(USER_FILE, userData)){
            return res.send({
                status:"ok",
                data: "Password changed sucessfully.",
                timestamp: new Date().toISOString()
            })
        }
        else{
            return res.status(500).send({
                status:"error",
                data:"Fail to change password",
                timestamp: new Date().toISOString()
            })
        }
    }
    else if(status === "false"){
        return res.status(401).json({
            status: "error",
            data: "Invalid credentials",
            timestamp: new Date().toISOString()
        });
    }
    else if(status === "error"){
        return res.status(403).json({
            status: "error",
            data: message,
            timestamp: new Date().toISOString()
        });
    }  
});

// control AC api
app.post('/v1/control', (req, res)=>{
    const switchOn = req.query.on;
    const {temperature, mode, fan, swing} = req.body;
    console.log("control")

    const {status, message} = verifyUser(req, secret);

    if(status === "true"){
        acStatus.on = switchOn;
        acStatus.temperature = temperature;
        acStatus.mode = mode;
        acStatus.fan = fan;
        acStatus.swing = swing;
        acStatus.timestamp = new Date().toISOString()

        addRecord({
            ...acStatus,
            user: message.username,
            ip: req.ip
        })

        return res.send({
            status:"ok",
            data:acStatus,
            timestamp: new Date().toISOString()
        })
    }
    else if(status === "false"){
        return res.status(401).send({
            status:"error",
            data:"Invalid credentials",
            timestamp: new Date().toISOString()
        })
    }
    else if(status === "error"){
        return res.status(403).send({
            status:"error",
            data:message,
            timestamp: new Date().toISOString()
        })
    }
})

// get control history api
app.get('/v1/control/history', (req, res) => {
    const { status, message } = verifyUser(req, secret);
    console.log("get history");

    if (status === "false") {
        return res.status(401).json({
            status: "error",
            message: "Invalid credentials",
            timestamp: new Date().toISOString()
        });
    }
    else if(status === "error"){
        return res.status(403).send({
            status:"error",
            data:message,
            timestamp: new Date().toISOString()
        })
    }

    res.json({
        status: "ok",
        data: getHistory(),
        timestamp: new Date().toISOString()
    });
});

// clear all record api
app.delete('/v1/control/history', (req, res) => {
    const { status, message } = verifyUser(req, secret);
    console.log("delete history");

    if (status === "false") {
        return res.status(401).json({
            status: "error",
            message: "Invalid credentials",
            timestamp: new Date().toISOString()
        });
    }
    else if(status === "error"){
        return res.status(403).send({
            status:"error",
            data:message,
            timestamp: new Date().toISOString()
        })
    }

    clearAllHistory();
    
    res.json({
        status: "ok",
        data: getHistory(),
        timestamp: new Date().toISOString()
    });
});

app.listen(5000, ()=>{
    console.log("server start running...");
})

