import {getToken} from '../utils'
import {Navigate} from 'react-router-dom'

// control the access of routes
function AuthRoute({children}){  // the child component as args
    const token = getToken();
    if(token){
        return <>{children}</>
    }else{
        return <Navigate to="/login" replace/>
    }
}

export default AuthRoute