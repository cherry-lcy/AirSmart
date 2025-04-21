import {
    Breadcrumb,
    Card, 
    Form,
    InputNumber,
    Switch,
    Select,
    Button,
    Space,
    message
} from 'antd';
import {useState, useEffect} from 'react';
import {Link, useNavigate} from 'react-router-dom';
import './index.scss';
import {fetchAcStatus} from '../../store/modules/panel';
import {useDispatch, useSelector} from 'react-redux';

const {Option} = Select;

const Panel = ()=>{
    const [form] = Form.useForm();
    const [formDisabled, setFormDisabled] = useState(true);
    const dispatch = useDispatch();
    // keep trach the change of the status and obtain status
    const acStatus = useSelector(state=>state.status)
    const navigate = useNavigate();

    useEffect(()=>{
        // update AC status in the form when user updates
        if(acStatus.on === true){
            form.setFieldsValue({
                temperature: acStatus.status.temperature || 22,
                mode: acStatus.status.mode || 'cool',
                fan: acStatus.status.fan || 'auto',
                swing: !!acStatus.status.swing
            });
            setFormDisabled(!acStatus.on);
        }
    }, [acStatus, form])

    // default value of the form
    const onFill = ()=>{
        form.setFieldsValue({
            temperature:22,
            mode:"cool",
            fan:"auto",
        });
    }

    // update form when the AC is on/off
    const onChange = async (checked)=>{
        setFormDisabled(!checked);
        if(!checked){
            form.resetFields();
            await dispatch(fetchAcStatus(checked, {}))

            navigate('/main/dashboard');
            message.success('Turn off sucessfully!');
            window.location.reload();
        }
        else{
            onFill();
        }
    }

    // update AC status and post it to the server when user submits form
    const onFinish = async (formValue)=>{
        const {temperature, mode, fan, swing} = formValue;
        const reqData = {
            temperature,
            mode,
            fan,
            swing
        }
        await dispatch(fetchAcStatus(!formDisabled, reqData));

        navigate('/main/dashboard');
        message.success('Status update sucessfully!');
    }

    return(
        <div>
            <Card
                title={<Breadcrumb items={[
                    {title: <Link to="/main/home">Home</Link>},
                    {title: 'Panel'},
                ]}/>
            }>
            <div className="panel-input">
                <div style={{marginBottom: 16, display: 'flex', alignItems: 'center'}}>
                    <span style={{marginRight: 8}}> On:</span>
                    <Switch
                        checked={!formDisabled}
                        onChange={onChange}
                    />
                </div>
                    
                <Form 
                    form={form} 
                    disabled={formDisabled}
                    onFinish={onFinish}
                >
                    <Form.Item
                        label="Temperature"
                        name="temperature"
                    >
                        <InputNumber 
                            min="16"
                            max="30"
                            style={{width: '100%'}} 
                        />
                    </Form.Item>
                    
                    <Form.Item
                        label="Mode"
                        name="mode"
                    >
                        <Select>
                            <Option value="cool">Cool</Option>
                            <Option value="dry">Dry</Option>
                            <Option value="fan">Fan</Option>
                        </Select>
                    </Form.Item>
                    
                    <Form.Item
                        label="Fan"
                        name="fan"
                    >
                        <Select>
                            <Option value="low">Low</Option>
                            <Option value="mid">Mid</Option>
                            <Option value="high">High</Option>
                            <Option value="auto">Auto</Option>
                        </Select>
                    </Form.Item>
                    
                    <Form.Item 
                        label="Swing" 
                        name="swing"
                        valuePropName="checked"
                    >
                        <Switch />
                    </Form.Item>
                    
                    <Form.Item>
                        <Space size={15}>
                            <Button type="primary" htmlType="submit">
                                Submit
                            </Button>
                            <Button htmlType="button" onClick={onFill}>
                                Default
                            </Button>
                            <Button htmlType="reset">Reset</Button>
                        </Space>
                    </Form.Item>
                </Form>
            </div>
        </Card>
    </div>
    )
}

export default Panel