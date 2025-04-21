import {LoadingOutlined} from '@ant-design/icons';
import {Link, useNavigate} from 'react-router-dom';
import {useEffect} from 'react';
import './index.scss';

const Default = ()=>{
    const navigate = useNavigate();

    // direct user to the login page
    useEffect(()=>{
        const timer = setTimeout(()=>{
            navigate('/login');

            return clearTimeout(timer);
        });
    },[navigate]);

    return (<div className="default-container">
        <LoadingOutlined style={{fontSize:"64px", marginBotton:"20px", display:"block"}}/>
        <div className="default">Redirecting you to <Link to="/login">Login Page</Link></div>
    </div>)
}

export default Default