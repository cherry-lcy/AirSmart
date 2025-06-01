const db = require("../config/db");
const User = require("../models/userModel");

class UserRepository{
    async getAll(){
        const [rows] = await db.query("SELECT * FROM users");
        return rows.map(row=>{return new User(row.uid, row.username, row.email, row.password, row.role)});
    };

    async getByUid(uid){
        const [rows] = await db.query('SELECT * FROM users WHERE uid = ?', uid);
        if(rows.length === 0){
            return null;
        }
        const row = rows[0];
        return new User(row.uid, row.username, row.email, row.password, row.role);
    };

    async getByUsername(username){
        const [rows] = await db.query('SELECT * FROM users WHERE username = ?');
        if(rows.length === 0){
            return null;
        }
        const row = rows[0];
        return new User(row.uid, row.username, row.email, row.password, row.role);
    };

    async getByEmail(email){
        const [rows] = await db.query('SELECT * FROM users WHERE email = ?', email);
        if(rows.length === 0){
            return null;
        }
        const row = rows[0];
        return new User(row.uid, row.username, row.email, row.password, row.role);
    };

    async addUser(username, email, password, role){
        const [results] = await db.query('INSERT INTO users (username, email, password, role) VALUES (?, ?, ?, ?)', username, email, password, role);
        if(results.length === 0){
            return null;
        }
        const row = results[0];
        return new User(row.uid, row.username, row.email, row.password, row.role);
    };
}

module.exports = new UserRepository();