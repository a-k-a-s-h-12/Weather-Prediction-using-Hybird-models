import { useState } from "react";
import { FiSearch } from "react-icons/fi";
import { motion } from "framer-motion";

export default function WeatherForm({ onSubmit, isLoading }) {
  const [city, setCity] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    if (city.trim()) {
      onSubmit(city);
    }
  };

  return (
    <motion.form 
      onSubmit={handleSubmit}
      className="p-4 flex gap-2 max-w-md mx-auto"
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="relative flex-1">
        <input
          type="text"
          placeholder="Enter city name..."
          value={city}
          onChange={(e) => setCity(e.target.value)}
          className="border-2 border-gray-200 rounded-lg p-3 pr-10 w-full focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent shadow-sm"
          disabled={isLoading}
        />
        <FiSearch className="absolute right-3 top-3.5 text-gray-400" />
      </div>
      <button
        type="submit"
        disabled={isLoading}
        className="bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 text-white px-6 py-3 rounded-lg font-medium shadow-md hover:shadow-lg transition-all duration-300 disabled:opacity-70 disabled:cursor-not-allowed flex items-center justify-center min-w-[120px]"
      >
        {isLoading ? (
          <span className="animate-pulse">Searching...</span>
        ) : (
          "Get Forecast"
        )}
      </button>
    </motion.form>
  );
}