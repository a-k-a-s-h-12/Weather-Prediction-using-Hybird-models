import requests

API_KEY = "ba13a38b7d189d8fa649dab6e47d3f12" 

def get_city_coordinates(city_name, state_code="", country_code="", limit=1):
    geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name},{state_code},{country_code}&limit={limit}&appid={API_KEY}"
    response = requests.get(geo_url)
    if response.status_code == 200 and response.json():
        data = response.json()[0]
        return data['lat'], data['lon']
    return None, None

def get_daily_forecasts(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code != 200:
        return None

    forecast_list = response.json()['list']
    daily_data = []
    for i in range(0, 40, 8):  
        chunk = forecast_list[i:i+8]
        precipitation = sum(period.get('rain', {}).get('3h', 0) for period in chunk)
        wind = sum(period['wind']['speed'] for period in chunk) / len(chunk)
        temp_max = max(period['main']['temp_max'] for period in chunk)
        temp_min = min(period['main']['temp_min'] for period in chunk)

        daily_data.append([round(precipitation, 2), round(temp_max, 2), round(temp_min, 2), round(wind, 2)])
    return daily_data
