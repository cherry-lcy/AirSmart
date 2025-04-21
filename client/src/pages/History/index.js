import {
    Breadcrumb, 
    Button,
    Card, 
    Table,
    Popconfirm
} from "antd";
import {useState, useEffect} from 'react';
import {Link} from 'react-router-dom';
import {getHistoryAPI, delHistoryAPI} from "../../apis/history";
import "./index.scss"

// table head
const columns = [
    {
        title:"Time",
        dataIndex:"time",
        key:"time"
    },
    {
        title:"Status",
        dataIndex:"on",
        key:"on",
        render: text => text === true ? "On" : "Off"
    },
    {
        title:"Temperature",
        dataIndex:"temperature",
        key:"temperature"
    },
    {
        title:"Mode",
        dataIndex:"mode",
        key:"mode"
    },
    {
        title:"Fan",
        dataIndex:"fan",
        key:"fan"
    },
    {
        title:"Swing",
        dataIndex:"swing",
        key:"swing",
        render: text => text === true ? "On" : ( text === false ? "Off" : "")
    }
]

const History = ()=>{
    const [list, setList] = useState([]);
    const [count, setCount] = useState(0);

    // update history when record updates
    useEffect(()=>{
        async function getList(){
            const res = await getHistoryAPI();
            
            const formattedData = res.data.data.map(item => {
                const date = new Date(item.timestamp);
                const formattedTime = date.toLocaleString('en-GB', { timeZone: 'UTC', hour12: false });
                return {
                    time: formattedTime,
                    on: item.on === "true",
                    temperature: item.temperature,
                    mode: item.mode,
                    fan: item.fan,
                    swing: item.swing
                }
            });
            setList(formattedData);
            setCount(res.data.total_count);
        }
        getList();
    },[])

    // delete all history
    const onConfirm = async ()=>{
        await delHistoryAPI();
        window.location.reload();
    }

    return (<div className="history">
        <Card
            title={<Breadcrumb items={[
                {title: <Link to="/main/home">Home</Link>},
                {title: 'History'},
            ]}/>
        }>
            <div className="description">
                <span>{`${count} historical record found.`}</span>
                <Popconfirm
                    title="Clear All History"
                    description="Are you sure to clear all history?"
                    onConfirm={onConfirm}
                    okText="Yes"
                    cancelText="No"
                >
                    <Button type="primary">Clear all history</Button>
                </Popconfirm>
            </div>
            <Table columns={columns} dataSource={list}/>
        </Card>
    </div>)
}

export default History