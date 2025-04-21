import {request} from '../utils';

// get historical record
export function getHistoryAPI(){
    return request({
        url:"/control/history",
        method:"GET"
    })
}

// delete all historical record
export function delHistoryAPI(){
    return request({
        url:"control/history",
        method:"DELETE"
    })
}