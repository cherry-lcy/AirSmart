import {Breadcrumb, Card} from 'antd';
import { Link } from 'react-router-dom';

const ChangePassword = ()=>{
    return (<div>
        <Card title={<Breadcrumb items={[
            {title: <Link to="/main/home">Home</Link>},
                    {title: 'Dashboard'},]
                    }/>}>
            
        </Card>
    </div>)
}

export default ChangePassword