import {createBrowserRouter} from 'react-router-dom';
import AuthRoute from '../components/AuthRoute';
import Default from '../pages/Default';
import Layout from '../pages/Layout';
import Login from '../pages/Login';
import Home from '../pages/Home';
import Dashboard from '../pages/Dashboard';
import History from '../pages/History';
import Panel from '../pages/Panel';

const router = createBrowserRouter([
    {
        path:'/',
        element: <Default/>
    },
    {
        path:'/login',
        element:<Login/>
    },
    {
        path:'/main',
        element:<AuthRoute><Layout/></AuthRoute>,
        children:[
            {
                path:'home',
                element:<Home/>
            },
            {
                path:'panel',
                element:<Panel/>
            },
            {
                path:'dashboard',
                element:<Dashboard/>
            },
            {
                path:'history',
                element:<History/>
            }
        ]
    }
])

export default router