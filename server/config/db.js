const mysql = require("mysql2/promise");

const pool = mysql.createPool({
    host: "localhost",
    port: "3306",
    user: "root",
    password: "Cherry5052005",
    database: "airsmart"
})

async function testConnection(){
    try{
        const connection = await pool.getConnection();
        console.log("connect sucessfully!");
        connection.release();
    }
    catch(e){
        console.error("connect failed: ", e);
    }
}

module.exports = {pool, testConnection};