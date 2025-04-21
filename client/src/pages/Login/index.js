import {Card, Form, Button, Input, message} from 'antd'
import '../Login/index.scss'
import Logo from '../../assets/logo.png'
import {useDispatch} from 'react-redux'
import {fetchLogin} from '../../store/modules/user'
import {useNavigate} from 'react-router-dom'

const Login = ()=>{
    const dispatch = useDispatch();
    const navigate = useNavigate();
    // user login
    const onFinish = async (val)=>{
        await dispatch(fetchLogin(val));

        navigate('/main/home');
        message.success('Login sucessfully');
    }

    return (<div className="Login">
        <Card className="login-container" variant="borderless">
            <Form validateTrigger="onBlur" onFinish={onFinish}>
                <img className="login-logo" src={Logo} alt="/"></img>
                <Form.Item 
                    name="username" 
                    rules={[{ 
                        required: true, 
                        message: 'Please input your username!' 
                    },{
                        pattern:/\S\w|[0-9]/,
                        message: 'Please input the username in correct format'
                    }]}
                    >
                    <Input size="large" placeholder="Username"></Input>
                </Form.Item>
                <Form.Item name="password" rules={[{ required: true, message: 'Please input your password!' }]}>
                    <Input.Password size="large" placeholder="Password"></Input.Password>
                </Form.Item>
                <Form.Item>
                    <Button type="primary" htmlType="submit" size="large" block>Login</Button>
                </Form.Item>
            </Form>
        </Card>
    </div>)
}

export default Login