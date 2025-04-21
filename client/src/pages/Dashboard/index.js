import {
    Breadcrumb, 
    Card
} from 'antd';
import {useEffect, useRef, useState} from 'react';
import {Link} from 'react-router-dom';
import {useSelector} from 'react-redux';
import * as echarts from 'echarts';
import fetchWeather from '../../apis/weather';
import './index.scss';

const {Meta} = Card;

// helper function: convert the first letter to upper case
function firstLetterUpperCase(str){
    if (typeof str !== 'string') {
        str = String(str);
    }
    return str.charAt(0).toUpperCase() + str.slice(1);
}

const Dashboard = ()=>{
    const chartRef = useRef();
    const [weather, setWeather] = useState({
        temperature: {
            data: [{
                "place": "King's Park",
                "value": "N/A",
                "unit": "C"
            }]
        },
        humidity: {
            data: [{
                "unit": "percent",
                "value": 77,
                "place": "Hong Kong Observatory"
            }]
        }
    });

    // get real-time weather information every 10 minutes
    useEffect(()=>{
        const fetchWeatherList = async () => {
            const response = await fetchWeather();
            setWeather({
                temperature:response.temperature.data[0],
                humidity:response.humidity.data[0]
            })
            console.log("weather:", weather);

            const timer = setInterval(async ()=>{
                const response = await fetchWeather();
                setWeather({
                    temperature:response.temperature.data[0],
                    humidity:response.humidity.data[0]
                })
            }, 600000)
            return ()=>clearInterval(timer);
        }
        fetchWeatherList();
    })

    // keep track of the weather information
    useEffect(() => {
        console.log("Updated weather:", weather);
    }, [weather]);

    // update AC status
    const acStatus = useSelector(state => {
        return state.status
    });

    // update the temperature chart when AC temperature updates
    useEffect(()=>{
        const chart = echarts.init(chartRef.current);

        const option = {
            series: [{
                type: 'gauge',
                center: ['50%', '60%'],
                startAngle: 180,
                endAngle: 0,
                min: 16,
                max: 30,
                splitNumber: 7,
                radius: '100%',
                
                axisLine: {
                    lineStyle: {
                    width: 30,
                    color: [[1,'#3770FD']]
                    }
                },
                
                splitLine: {
                    distance: -30,
                    length: 10,
                    lineStyle: {
                    color: '#fff'
                    }
                },
                
                axisLabel: {
                    distance: -20,
                    color: '#fff',
                    fontSize: 12
                },
                
                pointer: {
                    itemStyle: {
                    color: 'auto'
                    }
                },
                
                detail: {
                    valueAnimation: true,
                    formatter: `{value}°C`,
                    color: 'auto',
                    fontSize: 20,
                    offsetCenter: [0, '70%']
                },
                
                data: [{value: acStatus.status.temperature}]
            }]
        };

        chart.setOption(option);

        return () => chart.dispose();
    }, [acStatus.status.temperature])

    return (
        <div className="dashboard">
            <Card
                title={<Breadcrumb items={[
                    {title: <Link to="/main/home">Home</Link>},
                    {title: 'Dashboard'},
                ]}/>
            }>
            <div className="dashboard-container">
                <Card id="curr-status">
                    <Meta title="Current Status"></Meta>
                    <div className="status-container">
                    <div className="status-grp">
                        <p>Air Conditioner: <span>{(acStatus.on && "On") || "Off"}</span></p>
                        <p>Temperature: <span>{(acStatus.status.temperature && (acStatus.status.temperature + " °C")) || "/"}</span></p>
                        <p>Mode: <span>{(acStatus.status.mode && firstLetterUpperCase(acStatus.status.mode)) || "/"}</span></p>
                        <p>Fan: <span>{(acStatus.status.fan && firstLetterUpperCase(acStatus.status.fan)) || "/"}</span></p>
                        <p>Swing: <span>{(acStatus.status.swing && "On") || "Off"}</span></p>
                    </div>
                    <div ref={chartRef} className="temp-chart"/></div>
                </Card>
                <Card id="temp">
                    <Meta title="Temperature"></Meta>
                    <div className="temp-grp">
                        <p>Indoor: <span>22 °C</span></p>
                        <p>Outdoor: <span>{weather.temperature?.value || "N/A"} °C</span></p>
                    </div>
                </Card>
                <Card id="energy">
                    <Meta title="Energy Consumption"></Meta>
                    <div className="energy-grp">
                        <p>Today: <span>2.5 kWh</span></p>
                        <p>This month: <span>30kWh</span></p>
                    </div>
                </Card>
                <Card id="humid">
                    <Meta title="Humidity"></Meta>
                    <div className="humid-grp">
                        <p>Indoor: <span>60 %</span></p>
                        <p>Outdoor: <span>{weather.humidity?.value || "N/A"} %</span></p>
                    </div>
                </Card>
                <Card id="air-quality">
                    <Meta title="Air Quality"></Meta>
                    <div className="aq-grp">
                        <p>PM2.5: <span>12 µg/m³</span></p>
                        <p>CO2: <span>600 ppm</span></p>
                    </div>
                </Card>
            </div>
            </Card>
        </div>
    )
}

export default Dashboard;