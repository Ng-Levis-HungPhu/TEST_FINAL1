from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import pickle
import os

app = Flask(__name__)
CORS(app)

MODEL_DIR = "1. SAVING MODELS"

@app.route("/")
def index():
    return "Backend is running ✅"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        mode = data.get("mode", "").strip()
        mach = float(data['mach'])
        aoa = float(data['aoa'])
        ln = float(data['ln'])
        swept = float(data['swept'])
        lln = float(data['lln'])

        warning_msg = ""

        if mode == "NASA":
            if not (-4 <= aoa <= 25):
                return jsonify({"error": "AOA phải trong [-4, 25]"}), 400
            elif aoa < -2.5:
                warning_msg += "• AOA rất thấp. "

            if mach < 1.2:
                return jsonify({"error": "Mach phải ≥ 1.2"}), 400
            elif mach <= 1.6:
                warning_msg += "• Mach trong vùng nhiều nhiễu. "
            elif mach >= 3.2:
                warning_msg += "• Mach cao, sai số tăng. "

            if not (3 <= ln <= 20.32):
                return jsonify({"error": "Ln phải trong [3, 20.32]"}), 400
            elif ln < 6:
                warning_msg += "• Ln nhỏ dễ gây sai số. "

            if not (5 <= swept <= 70):
                return jsonify({"error": "Swept phải trong [5, 70]"}), 400

            model_cl = tf.keras.models.load_model(os.path.join(MODEL_DIR, "NASA_cl.h5"))
            model_cd = tf.keras.models.load_model(os.path.join(MODEL_DIR, "NASA_cd.h5"))
            with open(os.path.join(MODEL_DIR, "NASA.pkl"), "rb") as f:
                scaler = pickle.load(f)

            input_data = np.array([[lln, ln, swept, mach, aoa]])
            input_scaled = scaler.transform(input_data)

        elif mode in ["Von-Karman Nose", "Missile Shape 1", "Missile Shape 2", "Missile Shape 3"]:
            model_cl = tf.keras.models.load_model(os.path.join(MODEL_DIR, f"{mode}_cl.h5"))
            model_cd = tf.keras.models.load_model(os.path.join(MODEL_DIR, f"{mode}_cd.h5"))
            with open(os.path.join(MODEL_DIR, f"{mode}.pkl"), "rb") as f:
                scaler = pickle.load(f)

            input_data = np.array([[mach, aoa]])
            input_scaled = scaler.transform(input_data)

        else:
            return jsonify({"error": f"Chế độ không hỗ trợ: {mode}"}), 400

        cl = float(model_cl.predict(input_scaled)[0][0])
        cd = float(model_cd.predict(input_scaled)[0][0])

        return jsonify({
            "cl": round(cl, 5),
            "cd": round(cd, 5),
            "warning": warning_msg
        })

    except Exception as e:
        print("❌ Error in /predict:", e)
        return jsonify({"error": str(e)}), 500
