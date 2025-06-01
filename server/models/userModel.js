

class User{
    #uid;
    #username;
    #email;
    #password;
    #role;
    constructor(uid, username, email, password, role){
        this.#uid = uid;
        this.#username = username;
        this.#email = email;
        this.#password = password;
        this.#role = role;
    };

    get uid(){
        return this.#uid;
    }

    get username(){
        return this.#username;
    };

    get email(){
        return this.#email;
    };

    get role(){
        return this.#role;
    };

    verify(username, password){
        if(this.#username == username && this.#password == password){
            return true;
        }
        else{
            return false;
        }
    };

    changePassword(newPassword){
        this.#password = newPassword;
    };

    getProfile(){
        return JSON.stringify({
            uid: this.#uid,
            username: this.#username,
            email: this.#email,
            role: this.#role
        });
    }
}

module.exports = User;