// manage user information among different pages
import {createSlice} from '@reduxjs/toolkit'
import {request, setToken as _setToken, getToken, removeToken} from '../../utils'

const userStore = createSlice({
    name:"user",
    initialState:{
        token: getToken() || '',
        userInfo: {}
    },
    reducers:{
        setToken(state, action){
            state.token = action.payload;
            _setToken(action.payload);
        },
        setUserInfo(state, action){
            state.userInfo = action.payload;
        },
        clearUserInfo(state){
            state.token = '';
            state.userInfo = {};
            removeToken();
        }
    },
})

const { setToken, setUserInfo, clearUserInfo } = userStore.actions;

const userReducer = userStore.reducer;

const fetchLogin = (loginForm)=>{
    return async (dispatch)=>{
        const res = await request.post('/authorization', loginForm);
        console.log(res.data);
        dispatch(setToken(res.data.token));
        dispatch(setUserInfo());
    }
}

const fetchUserInfo = ()=>{
    return async (dispatch)=>{
        const res = await request.get('/user/profile');
        dispatch(setUserInfo(res.data.userInfo));
    }
}

export {fetchLogin, setToken, fetchUserInfo, clearUserInfo}

export default userReducer