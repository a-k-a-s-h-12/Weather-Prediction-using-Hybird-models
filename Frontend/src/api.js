const API_BASE = "http://localhost:8000"; // backend URL

export async function getWeatherPrediction(city) {
  try {
    const response = await fetch(`${API_BASE}/weather/predict/${city}`);
    if (!response.ok) throw new Error("API error");
    return await response.json();
  } catch (error) {
    console.error("Error fetching prediction:", error);
    return { error: error.message };
  }
}