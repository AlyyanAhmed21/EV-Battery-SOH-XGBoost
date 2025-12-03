from flask import Flask, render_template, request
import pandas as pd
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from batteryPredictor.pipeline.prediction import PredictionPipeline

app = Flask(__name__)

def generate_plots(df):
    """Helper to create graphs for the dashboard"""
    img_list = []
    
    # PLOT 1: Voltage Curve & Imbalance
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(df['capacity'], df['avg_voltage'], label='Avg Voltage', color='blue')
    plt.title("Voltage Discharge Curve")
    plt.xlabel("Capacity Used (Ah)")
    plt.ylabel("Voltage (V)")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.subplot(1, 2, 2)
    plt.plot(df['capacity'], df['pack_imbalance'], label='Imbalance', color='red')
    plt.title("Cell Imbalance Trend")
    plt.xlabel("Capacity Used (Ah)")
    plt.ylabel("Delta V (Max - Min)")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Convert plot to Base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    img_list.append(plot_url)
    plt.close()

    # PLOT 2: Predicted Range vs Actual
    plt.figure(figsize=(10, 5))
    plt.plot(df['capacity'], df['predicted_remaining_capacity'], color='green', linewidth=2, label="AI Predicted Range")
    plt.title("AI Range Prediction (The 'Trend of Fall')")
    plt.xlabel("Capacity Consumed (Ah)")
    plt.ylabel("Estimated Remaining Capacity (Ah)")
    plt.legend()
    plt.grid(True)
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    img_list.append(plot_url)
    plt.close()
    
    return img_list

@app.route('/', methods=['GET'])
def homePage():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload_analyze():
    try:
        if 'file' not in request.files:
            return "No file uploaded"
        
        file = request.files['file']
        if file.filename == '':
            return "No file selected"

        # Read CSV
        data = pd.read_csv(file)
        
        # Run Pipeline
        pipeline = PredictionPipeline()
        final_soh, result_df = pipeline.predict_bulk(data)
        
        # Generate Graphs
        plots = generate_plots(result_df)
        
        return render_template("dashboard.html", 
                               soh=f"{final_soh:.2f}%", 
                               filename=file.filename,
                               plots=plots)

    except Exception as e:
        return f"Error analyzing file: {e}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)