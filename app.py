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

def generate_overlay_plot(history_df, projection_df):
    img_list = []
    
    plt.figure(figsize=(12, 7))
    
    # 1. Plot User History
    plt.plot(history_df['capacity'], history_df['avg_voltage'], 
             label='Actual Data (History)', color='blue', linewidth=2)
    
    # 2. Plot AI Projection
    if not projection_df.empty:
        plt.plot(projection_df['capacity'], projection_df['avg_voltage'], 
                 label='AI Predicted Path', color='orange', linestyle='-', linewidth=2, alpha=0.9)
        
        plt.fill_between(projection_df['capacity'], 
                         projection_df['lower_bound'], 
                         projection_df['upper_bound'], 
                         color='orange', alpha=0.2, 
                         label='Predicted Variability Zone')

    # Styling
    plt.title("Live Diagnosis: Discharge Path & Variability Prediction", fontsize=14)
    plt.xlabel("Capacity Removed (Ah)", fontsize=12)
    plt.ylabel("Voltage (V)", fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # --- AUTO ZOOM ---
    # Focus Y-axis on relevant area (e.g. 4.2V down to 3.0V)
    plt.ylim(3.0, 4.3)
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
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
        final_soh, result_df, proj_df, total_cap = pipeline.predict_bulk(data)
        
        # Generate Plot
        plots = generate_overlay_plot(result_df, proj_df)
        
        # Calculate Percentage
        current_cap = result_df['capacity'].iloc[-1]
        percent_used = (current_cap / total_cap) * 100
        
        return render_template("dashboard.html", 
                               soh=f"{final_soh:.2f}%", 
                               total_cap=f"{total_cap:.2f} Ah",
                               status=f"{percent_used:.1f}% Discharged",
                               filename=file.filename,
                               plots=plots)

    except Exception as e:
        return f"Error analyzing file: {e}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)