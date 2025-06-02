const db = require("../config/db");

class UserRepository{
    async getAll(){
        try{  
            const [rows] = await db.pool.query("SELECT * FROM users");
            return {
                success: true,
                operation: 'read',
                data: rows,
                timestamp: new Date().toISOString()
            }
        }
        catch(err){
            console.error("Error in getAll:", e.message);
            return {
                success: false,
                operation: 'read',
                error: {
                    code: err.code || 'database error',
                    message: err.message
                },
                timestamp: new Date().toISOString()
            }
        }
    };

    async getByUid(uid){
        try{
            const [rows] = await db.pool.query('SELECT * FROM users WHERE uid = ?', uid);
            if(rows.length === 0){
                return {
                    success: false,
                    operation: 'read',
                    error: {
                        code: 'database error',
                        message: 'data does not exist'
                    },
                    timestamp: new Date().toISOString()
                }
            }
            const row = rows[0];
            return {
                success: true,
                operation: 'read',
                data: row,
                timestamp: new Date().toISOString()
            }
        }
        catch(err){
            console.error("Error in getByUid: ", err);
            return {
                success: false,
                operation: 'read',
                error: {
                    code: err.code || 'database error',
                    message: err.message
                },
                timestamp: new Date().toISOString()
            }
        }
    };

    async getByUsername(username){
        try{
            const [rows] = await db.pool.query('SELECT * FROM users WHERE username = ?', [username]);
            if(rows.length === 0){
                return {
                    success: false,
                    operation: 'read',
                    error: {
                        code: 'database error',
                        message: 'data does not exist'
                    },
                    timestamp: new Date().toISOString()
                }
            }
            const row = rows[0];
            return {
                success: true,
                operation: 'read',
                data: row,
                timestamp: new Date().toISOString()
            }
        }
        catch(err){
            console.error("Error in getByUsername: ", err);
            return {
                success: false,
                operation: 'read',
                error: {
                    code: err.code || 'database error',
                    message: err.message
                },
                timestamp: new Date().toISOString()
            }
        }
    };

    async getByEmail(email){
        try{
            const [rows] = await db.pool.query('SELECT * FROM users WHERE email = ?', [email]);
            if(rows.length === 0){
                return {
                    success: false,
                    operation: 'read',
                    error: {
                        code: 'database error',
                        message: 'data does not exist'
                    },
                    timestamp: new Date().toISOString()
                };
            }
            const row = rows[0];
            return {
                success: true,
                operation: 'read',
                data: row,
                timestamp: new Date().toISOString()
            }
        }
        catch(err){
            console.error("Error in getByEmail: ", err);
            return {
                success: false,
                operation: 'read',
                error: {
                    code: err.code || 'database error',
                    message: err.message
                },
                timestamp: new Date().toISOString()
            }
        }
    };

    async addUser(username, email, password, role){
        try{
            const [results] = await db.pool.query('INSERT INTO users (username, email, password, role) VALUES (?, ?, ?, ?)', [username, email, password, role]);
            if(results.length === 0){
                return {
                    success: false,
                    operation: 'read',
                    error: {
                        code: 'database error',
                        message: 'data does not exist'
                    },
                    timestamp: new Date().toISOString()
                };
            }
            const result = results[0];
            return { 
                success: true,
                operation: 'insert',
                insertedID: result.insertedID,
                affectedRows: result.affectedRows,
                data:{
                    uid: result.insertedID,
                    username: username,
                    email: email,
                    password: password,
                    role: role
                },
                timestamp: new Date().toISOString()
            };
        }
        catch(err){
            console.error("Error in addUser: ", err);
            return {
                success: false,
                operation: 'insert',
                error: {
                    code: err.code || 'database error',
                    message: err.message
                },
                timestamp: new Date().toISOString()
            }
        }
    };

    async updatePassword(uid, currPassword, newPassword){
        try{
            const [results] = await db.pool.query("UPDATE users SET password = ? WHERE uid = ? AND password = ?", [newPassword, uid, currPassword]);
            if(results.length === 0){
                return {
                    success: false,
                    operation: 'read',
                    error: {
                        code: 'database error',
                        message: 'data does not exist'
                    },
                    timestamp: new Date().toISOString()
                };
            }
            const result = results[0];
            return { 
                success: true,
                operation: 'update',
                affectedRows: result.affectedRows,
                changedRows: result.changedRows,
                oldData:{
                    password: currPassword
                },
                newData: {
                    password: newPassword
                },
                timestamp: new Date().toISOString()
            };
        }
        catch(err){
            console.error("Error in addUser: ", err);
            return {
                success: false,
                operation: 'insert',
                error: {
                    code: err.code || 'database error',
                    message: err.message
                },
                timestamp: new Date().toISOString()
            }
        }
    }
}

module.exports = new UserRepository();