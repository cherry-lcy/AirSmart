import {configureStore} from '@reduxjs/toolkit'
import userReducer from './modules/user';
import acStatusReducer from './modules/panel';
import {getAcStatus} from '../utils';

const currentStatus = getAcStatus();

const store = configureStore({
    reducer:{
        user: userReducer,
        status: acStatusReducer
    },
    currentStatus
})

export default store