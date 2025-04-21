const express = require("express");
const jwt = require("jsonwebtoken");
const crypto = require("crypto");
const {addRecord, getHistory, clearAllHistory} = require("./utils/historyStorage");
const {verifyUser} = require("./utils/user");

const app = express();

const secret = crypto.randomBytes(32).toString('hex');

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

const Admin = {
    username:"admin01",
    password:"123456",
    uid:"1"
}

let acStatus = {
    on:false,
    temperature:22,
    mode:"cool",
    fan:"auto",
    swing:true
}

app.post('/authorization', (req, res)=>{
    const {username, password} = req.body;
    console.log("post: ", req.body);
    
    if(username===Admin.username && password===Admin.password){
        const token = jwt.sign({ uid: Admin.uid }, secret, {expiresIn: "4h"})
        res.send({
            status:"ok",
            data:{
                token: token,
                username: Admin.username,
                uid:Admin.uid
            }
        })
    }
    else{
        res.status(403).send({
            status:"error",
            data:"Either username or password is incorrect."
        })
    }
})

app.get('/user/profile', (req, res) => {
    const {status, message} = verifyUser(req, Admin, secret);
    
    if (status === "true") {
        return res.json({
            status: "ok",
            data: {
                userInfo: Admin
            }
        });
    }
    else if(status === "false"){
        return res.status(403).json({
            status: "error",
            data: "Invalid credentials"
        });
    }
    else if(status === "error"){
        return res.status(403).json({
            status: "error",
            data: message
        });
    }  
});

app.post('/control', (req, res)=>{
    const switchOn = req.query.on;
    const {temperature, mode, fan, swing} = req.body;
    console.log("control")

    const {status, message} = verifyUser(req, Admin, secret);

    if(status === "true"){
        acStatus.on = switchOn;
        acStatus.temperature = temperature;
        acStatus.mode = mode;
        acStatus.fan = fan;
        acStatus.swing = swing;
        acStatus.timestamp = new Date().toISOString()

        addRecord({
            ...acStatus,
            user: Admin.username,
            ip: req.ip
        })

        return res.send({
            status:"ok",
            data:acStatus
        })
    }
    else if(status === "false"){
        return res.status(403).send({
            status:"error",
            data:"Invalid credentials"
        })
    }
    else if(status === "error"){
        return res.status(403).send({
            status:"error",
            data:message
        })
    }
})

app.get('/control/history', (req, res) => {
    const { status, message } = verifyUser(req, Admin, secret);
    console.log("get history");

    if (status === "false") {
        return res.status(403).json({
            status: "error",
            message: "Invalid credentials"
        });
    }
    else if(status === "error"){
        return res.status(403).send({
            status:"error",
            data:message
        })
    }

    res.json({
        status: "ok",
        data: getHistory()
    });
});

app.delete('/control/history', (req, res) => {
    const { status, message } = verifyUser(req, Admin, secret);
    console.log("delete history");

    if (status === "false") {
        return res.status(403).json({
            status: "error",
            message: "Invalid credentials"
        });
    }
    else if(status === "error"){
        return res.status(403).send({
            status:"error",
            data:message
        })
    }

    clearAllHistory();
    
    res.json({
        status: "ok",
        data: getHistory()
    });
});

app.listen(5000, ()=>{
    console.log("server start running...");
})

