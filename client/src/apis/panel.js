import {request} from '../utils';

// control AC status
export function controlAPI(switchOn, reqData){
    return request({
        url:`/control/?on=${switchOn}`,
        method:"POST",
        data:reqData
    })
}