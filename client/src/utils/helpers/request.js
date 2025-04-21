import axios from 'axios'
import {getToken, removeToken} from '../index'
import router from '../../router'

const request = axios.create({
    baseURL:'http://localhost:5000',
    timeout: 5000
})

// request interceptor
request.interceptors.request.use((config)=>{
    const token = getToken();
    // add token in header before sending request
    if(token){
        config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
}, (error)=>{
    return Promise.reject(error)
})

// response interceptor
request.interceptors.response.use((response)=>{
    return response.data
}, (error)=>{
    // remove token if token is expired
    if(error.response.status === 401){
        removeToken();
        router.navigate('/login');
        window.location.reload();
    }
    return Promise.reject(error)
})

export {request}