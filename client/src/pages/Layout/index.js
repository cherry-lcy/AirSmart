import {Layout, Menu, Popconfirm, message} from 'antd';
import {
    HomeOutlined, 
    ControlOutlined,
    DashboardOutlined, 
    HistoryOutlined,
    LogoutOutlined,
    LoadingOutlined
} from '@ant-design/icons';
import './index.scss';
import {Outlet, useNavigate, useLocation} from 'react-router-dom';
import {useEffect} from 'react';
import {fetchUserInfo, clearUserInfo} from '../../store/modules/user'
import {useDispatch, useSelector} from 'react-redux';

const {Header, Sider, Content} = Layout;

const items = [
    {
        label:"Home",
        key:"/main/home",
        icon:<HomeOutlined />
    },
    {
        label:"Panel",
        key:"/main/panel",
        icon:<ControlOutlined />
    },
    {
        label:"Dashboard",
        key:"/main/dashboard",
        icon:<DashboardOutlined />
    }, 
    {
        label:"History",
        key:"/main/history",
        icon:<HistoryOutlined />
    }
]

const SystemLayout = ()=>{
    const navigate = useNavigate();

    // update sider when user click menu
    const onMenuClick = (route) => {
        navigate(route.key);
    } 

    // post sucess message when user logout sucessfully
    const [messageApi] = message.useMessage();
    const info = () => {
      messageApi.info('Logout sucessfully!');
    };

    // get current path
    const location = useLocation();
    const selectedKey = location.pathname;

    // update userInfo
    const dispatch = useDispatch();
    useEffect(()=>{
      const fetchData = async () => {
        await dispatch(fetchUserInfo());
      };
      fetchData();
    }, [dispatch]);

    // get userInfo
    const userInfo = useSelector(state => {
      return state.user.userInfo
    });
    console.log('userInfo:', userInfo);

    // loading page if userInfo is undefined
    if(!userInfo){
        return <div className="default-container">
            <LoadingOutlined style={{fontSize:"64px", marginBotton:"20px", display:"block"}}/>
            <div className="default">Loading...</div>
        </div>;
    }

    // logout 
    const onConfirm = ()=>{
        dispatch(clearUserInfo());
        navigate('/login');
        return info;
    }

    return (
    <Layout className="main-layout">
        <Header className="header">
          <div className="header-container">
            <div className="logo-container">
              <div className="logo" />
              <div className="logo-name">AirSmart</div>
            </div>
            <div className="user-info">
              <span className="user-name">{userInfo.username||"guest"}</span>
              <span className="user-logout">
                <Popconfirm title="Are you sure that you want to logout?" okText="Continue" cancelText="Cancel" onConfirm={onConfirm}>
                  <LogoutOutlined /> Logout
                </Popconfirm>
              </span>
            </div>
          </div>
        </Header>
  
        <Layout className="content-layout">
          <Sider width={250} className="site-layout-background" style={{height:"100%"}}>
            <Menu
              mode="inline"
              items={items}
              style={{ height: '100%' }}
              onClick={onMenuClick}
              defaultSelectedKeys='/main/home'
              selectedKeys={selectedKey}
            />
          </Sider>
          <Content className="layout-content">
            <Outlet/>
          </Content>
        </Layout>
      </Layout>
    )
}

export default SystemLayout