// manage AC status among different pages
import {createSlice} from '@reduxjs/toolkit';
import {controlAPI} from '../../apis/panel';
import {setAcStatus as _setAcStatus, removeAcStatus} from '../../utils';

const acStatus = createSlice({
    name:'acStatus',
    initialState:{
        switch:false,
        status:{}
    },
    reducers:{
        setAcStatus(state, action){
            state.on = Boolean(action.payload.on);
            state.status = {
                temperature: action.payload.temperature,
                mode: action.payload.mode,
                fan: action.payload.fan,
                swing: action.payload.swing
            }
            _setAcStatus(action.payload);
        },
        clearAcStatus(state){
            state.on = false;
            state.status = {}
            removeAcStatus();
        }
    }
})

const {setAcStatus, clearAcStatus} = acStatus.actions;

const acStatusReducer = acStatus.reducer;

const fetchAcStatus = (switchOn, formData)=>{
    return async (dispatch)=>{
        const res = await controlAPI(switchOn, formData);
        dispatch(setAcStatus(res.data));
    }
}

export {fetchAcStatus, clearAcStatus}

export default acStatusReducer