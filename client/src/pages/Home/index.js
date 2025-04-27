import {Avatar, Breadcrumb, Button, Card} from 'antd';
import { UserOutlined } from '@ant-design/icons';
import {useSelector} from 'react-redux';
import { useNavigate } from 'react-router-dom';
import "./index.scss";

const {Meta} = Card;

const Home = ()=>{
    // get user information
    const userInfo = useSelector(state => {
        return state.user.userInfo
    });

    const navigate = useNavigate();

    const changePassword = ()=>{
        navigate("/main/home/change-password");
    }

    return (<div>
        <Card title={<Breadcrumb items={[{title: 'Home'},]}/>}>
            <div className="user-info-container">
                <Card id="info-display">
                    <Meta title="User Information"/>
                    <div className="user-info-container">
                        <Avatar size={128} icon={<UserOutlined />}/>
                        <div className="user-description">
                            <p>Username: {userInfo.username}</p>
                            <p>Uid: {userInfo.uid}</p>
                        </div>
                    </div>
                </Card>
                <Card id="service-display">
                    <Meta title="User Service"></Meta>
                    <div className="service-grp">
                        <Button onClick={changePassword}>Change Password</Button>
                        <Button>Help</Button>
                        <Button>Delete Account</Button>
                    </div>
                </Card>
            </div>
        </Card>
    </div>)
}

export default Home