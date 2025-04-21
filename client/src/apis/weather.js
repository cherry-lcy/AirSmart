import axios from 'axios';

const req = axios.create({
    baseURL: "", 
    timeout: 10000
});

// get real-time weather information from Hong Kong Observatory Weather API
async function getWeatherAPI() {
    try {
        const response = await req({
            url: "https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=rhrread&lang=en",
            method: "GET",
            headers: {
                "Content-Type": "application/json"
            }
        });
        return response.data;
    } catch (err) {
        console.error("Error fetching weather data:", err);
        return {
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
        };
    }
}

async function fetchWeather() {
    const weatherData = await getWeatherAPI();
    return weatherData;
}

fetchWeather();

export default fetchWeather;