import numpy as np

def predict_weather(forecast_data, feature_extractor, xgb_model, label_encoder):
    predictions = []
    for i, day_data in enumerate(forecast_data[:5]):
        input_array = np.array(day_data).reshape(1, -1)
        input_lstm = input_array.reshape((1, input_array.shape[1], 1))

        lstm_features = feature_extractor.predict(input_lstm)
        pred_encoded = xgb_model.predict(lstm_features)
        pred_weather = label_encoder.inverse_transform(pred_encoded.astype(int))[0]

        predictions.append({
            "day": f"DAY {i+1}",
            "weather": pred_weather,
            "precipitation": day_data[0],
            "temp_min": day_data[2],
            "temp_max": day_data[1],
            "wind": day_data[3]
        })
    return predictions
