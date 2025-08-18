import { motion } from "framer-motion";
import { 
  WiDaySunny, WiRain, WiCloudy, WiSnow, WiThunderstorm, 
  WiFog, WiDayCloudy, WiNightClear, WiHumidity, WiStrongWind 
} from "react-icons/wi";
import { useRef } from "react";

import jsPDF from "jspdf";
import autoTable from "jspdf-autotable";

const weatherIcons = {
  sunny: <WiDaySunny className="text-yellow-500" size="1.5em" />,
  rain: <WiRain className="text-blue-400" size="1.5em" />,
  cloudy: <WiCloudy className="text-gray-400" size="1.5em" />,
  snow: <WiSnow className="text-blue-200" size="1.5em" />,
  thunderstorm: <WiThunderstorm className="text-purple-500" size="1.5em" />,
  fog: <WiFog className="text-gray-300" size="1.5em" />,
  partly_cloudy: <WiDayCloudy className="text-gray-400" size="1.5em" />,
  clear: <WiNightClear className="text-indigo-400" size="1.5em" />,
};

const getWeatherIcon = (weather) => {
  if (!weather) return weatherIcons.clear;
  const lowerWeather = weather.toLowerCase();
  if (lowerWeather.includes("sun")) return weatherIcons.sunny;
  if (lowerWeather.includes("rain")) return weatherIcons.rain;
  if (lowerWeather.includes("cloud")) return weatherIcons.cloudy;
  if (lowerWeather.includes("snow")) return weatherIcons.snow;
  if (lowerWeather.includes("storm")) return weatherIcons.thunderstorm;
  if (lowerWeather.includes("fog")) return weatherIcons.fog;
  return weatherIcons.clear;
};

export default function WeatherResults({ data, isLoading }) {
  const reportRef = useRef();

const downloadPDF = (data) => {
  const doc = new jsPDF();

  // Title
  doc.setFontSize(18);
  doc.text(`${data.city} Weather Forecast Report`, 14, 20);

  // Subtitle
  doc.setFontSize(11);
  doc.setTextColor(100);
  doc.text(`Next ${data.predictions?.length || 0} days`, 14, 28);
  doc.text(`Generated on: ${new Date().toLocaleDateString()}`, 14, 34);

  // Table Data
  const tableColumn = ["Day", "Weather", "Precipitation", "Min Temp", "Max Temp", "Wind"];
  const tableRows = data.predictions?.map((p, idx) => [
    `Day ${idx + 1}`,
    p.weather || "Clear",
    `${p.precipitation ?? 0}%`,
    `${p.temp_min}°C`,
    `${p.temp_max}°C`,
    `${p.wind} km/h`,
  ]);

  // AutoTable call
  autoTable(doc, {
    head: [tableColumn],
    body: tableRows,
    startY: 40,
    theme: "grid",
  });

  doc.save(`${data.city}_weather_forecast.pdf`);
};

  if (isLoading) {
    return (
      <div className="flex justify-center py-12">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  if (!data) return null;

  if (data.error) {
    return (
      <motion.div 
        className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 max-w-md mx-auto my-4 rounded"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
      >
        <p>{data.error}</p>
      </motion.div>
    );
  }

  return (
    <motion.div 
      className="p-4 max-w-4xl mx-auto"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ delay: 0.2 }}
    >
      <div className="flex justify-end mb-4">
        <button
          onClick={()=> {downloadPDF(data)}}
          className="bg-green-500 hover:bg-green-600 text-white px-4 py-2 rounded-lg font-medium shadow-md transition-colors duration-300 flex items-center gap-2"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L9 10.586V3a1 1 0 112 0v7.586l1.293-1.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clipRule="evenodd" />
          </svg>
          Download Report
        </button>
      </div>

      <div ref={reportRef} className="bg-white rounded-xl shadow-lg overflow-hidden mb-6">
        <div className="bg-gradient-to-r from-blue-500 to-blue-600 p-6 text-white">
          <h2 className="text-2xl font-bold">
            {data.city} Weather Forecast Report
          </h2>
          <p className="opacity-90">Next {data.predictions?.length || 0} days</p>
          <p className="text-sm opacity-80 mt-1">
            Generated on {new Date().toLocaleDateString()}
          </p>
        </div>
        
        {/* Desktop Table */}
        <div className="hidden md:block overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Day</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Weather</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Precipitation</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Min Temp</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Max Temp</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Wind</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {data.predictions?.map((p, idx) => (
                <tr key={idx}>
                  <td className="px-6 py-4 whitespace-nowrap font-medium">Day {idx + 1}</td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center gap-2">
                      {getWeatherIcon(p.weather)}
                      <span className="capitalize">{p.weather || 'Clear'}</span>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <WiHumidity className="text-blue-400 mr-1" size="1.2em" />
                      {p.precipitation ?? 0}%
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-blue-500 font-medium">
                    {p.temp_min}°
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-red-500 font-medium">
                    {p.temp_max}°
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <WiStrongWind className="text-gray-400 mr-1" size="1.2em" />
                      {p.wind} km/h
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        
        {/* Mobile Cards - Not included in PDF for better formatting */}
        <div className="md:hidden space-y-4 p-4">
          {data.predictions?.map((p, idx) => (
            <div
              key={idx}
              className="bg-white rounded-lg shadow-md p-4"
            >
              <div className="flex justify-between items-center mb-3">
                <h3 className="font-bold text-lg">Day {idx + 1}</h3>
                <div className="flex items-center gap-2">
                  {getWeatherIcon(p.weather)}
                  <span className="capitalize">{p.weather || 'Clear'}</span>
                </div>
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div className="flex items-center">
                  <WiHumidity className="text-blue-400 mr-2" size="1.5em" />
                  <div>
                    <p className="text-xs text-gray-500">Precipitation</p>
                    <p className="font-medium">{p.precipitation ?? 0}%</p>
                  </div>
                </div>
                
                <div className="flex items-center">
                  <div>
                    <p className="text-xs text-gray-500">Temperature</p>
                    <div className="flex items-center gap-2">
                      <span className="text-blue-500 font-medium">{p.temp_min}°</span>
                      <span className="text-gray-400">/</span>
                      <span className="text-red-500 font-medium">{p.temp_max}°</span>
                    </div>
                  </div>
                </div>
                
                <div className="col-span-2 flex items-center pt-2 border-t border-gray-100">
                  <WiStrongWind className="text-gray-400 mr-2" size="1.5em" />
                  <div>
                    <p className="text-xs text-gray-500">Wind Speed</p>
                    <p className="font-medium">{p.wind} km/h</p>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </motion.div>
  );
}