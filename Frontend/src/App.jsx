import { useState } from "react";
import { motion } from "framer-motion";
import WeatherForm from "./components/WeatherForm";
import WeatherResults from "./components/WeatherResults";
import { getWeatherPrediction } from "./api";
import './App.css';

function App() {
  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFetch = async (city) => {
    setIsLoading(true);
    setError(null);
    setResults(null);
    try {
      const data = await getWeatherPrediction(city);
      setResults(data);
    } catch (err) {
      setError(err.message || "Failed to fetch weather data");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-gray-100 py-8 px-4">
      <div className="max-w-6xl mx-auto">
        <motion.div
          className="text-center mb-8"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <h1 className="text-4xl font-bold text-gray-800 mb-2">
            <span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-500 to-blue-700">
              Weather Forecast
            </span>
          </h1>
          <p className="text-gray-600 max-w-md mx-auto">
            Get accurate weather predictions for any city worldwide
          </p>
        </motion.div>

        <WeatherForm onSubmit={handleFetch} isLoading={isLoading} />
        {error && (
          <motion.div 
            className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 max-w-md mx-auto my-4 rounded"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
          >
            <p>{error}</p>
          </motion.div>
        )}
        <WeatherResults data={results} isLoading={isLoading} />
      </div>
    </div>
  );
}

export default App;