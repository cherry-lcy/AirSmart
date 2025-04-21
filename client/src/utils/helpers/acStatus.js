// manage AC status in local storage

const KEYLIST = ["On", "Temperature", "Mode", "Fan", "Swing"]
function setAcStatus(acStatus){
    console.log('local storage:', acStatus.on);
    localStorage.setItem(KEYLIST[0], acStatus.on);
    localStorage.setItem(KEYLIST[1], acStatus.temperature);
    localStorage.setItem(KEYLIST[2], acStatus.mode);
    localStorage.setItem(KEYLIST[3], acStatus.fan);
    localStorage.setItem(KEYLIST[4], acStatus.swing);
}

function getAcStatus(){
    try{
        const switchOn = localStorage.getItem(KEYLIST[0]);
        const temperature = localStorage.getItem(KEYLIST[1]);
        const mode = localStorage.getItem(KEYLIST[2]);
        const fan = localStorage.getItem(KEYLIST[3]);
        const swing = localStorage.getItem(KEYLIST[4]);
        return {
            on: switchOn,
            temperature: temperature,
            mode: mode,
            fan: fan,
            swing: swing
        } 
    }
    catch(e){
        console.log(e);
        return undefined;
    }
}

function removeAcStatus(){
    localStorage.removeItem(KEYLIST[0]);
    localStorage.removeItem(KEYLIST[1]);
    localStorage.removeItem(KEYLIST[2]);
    localStorage.removeItem(KEYLIST[3]);
    localStorage.removeItem(KEYLIST[4]);
}

export {setAcStatus, getAcStatus, removeAcStatus}