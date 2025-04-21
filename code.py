import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import scipy.cluster.hierarchy as sch
from PIL import Image, ImageTk

# ================== Frontend Code ==================
root = tk.Tk()
root.title("House Price Prediction Project")
root.geometry("1400x1000")

# Main parent panels
main_frame = tk.Frame(root)
main_frame.pack(expand=True, fill="both")

# Configure main columns
main_frame.columnconfigure(0, weight=3)  # Parent Panel 1 (wider)
main_frame.columnconfigure(1, weight=1)  # Parent Panel 2 (narrower)
main_frame.rowconfigure(0, weight=1)

# ========== PARENT PANEL 1 ==========
parent_panel1 = tk.Frame(main_frame, bg='#f0f0f0', relief="groove")
parent_panel1.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

# Configure Parent 1 rows
parent_panel1.rowconfigure(0, weight=1)  # Sub-panel 1 (Image + Info)
parent_panel1.rowconfigure(1, weight=3)  # Sub-panel 2 (Input Fields)
parent_panel1.rowconfigure(2, weight=2)  # Sub-panel 3 (Similar Houses)
parent_panel1.columnconfigure(0, weight=1)

# Sub-panel 1 (Image + Project Info)
subpanel1 = tk.Frame(parent_panel1, bg='white', relief="solid")
subpanel1.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
subpanel1.columnconfigure(0, weight=1)
subpanel1.columnconfigure(1, weight=1)
subpanel1.rowconfigure(0, weight=1)

# 1a - Image Panel
panel1a = tk.Frame(subpanel1, bg='white', relief="solid")
panel1a.grid(row=0, column=0, sticky="nsew")
try:
    img = Image.open("house.png")
    img = img.resize((300, 200), Image.LANCZOS)
    photo_img = ImageTk.PhotoImage(img)
    image_label = tk.Label(panel1a, image=photo_img, bg='white')
    image_label.image = photo_img
    image_label.pack(expand=True)
except FileNotFoundError:
    tk.Label(panel1a, text="Image not found", bg='white', fg='red').pack(expand=True)

# 1b - Project Info
panel1b = tk.Frame(subpanel1, bg='white', relief="solid")
panel1b.grid(row=0, column=1, sticky="nsew")
info_text = """House Price Prediction & Clustering Project

Team Members:
- Omar
- Yamen
- Naffa

Machine Learning 2"""
tk.Label(panel1b, text=info_text, bg='white', 
        font=("Arial", 12), justify='left').pack(expand=True, padx=10)

# Sub-panel 2 (Input Fields)
input_panel = tk.Frame(parent_panel1, bg='white', relief="solid")
input_panel.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)
input_panel.columnconfigure(0, weight=3)
input_panel.columnconfigure(1, weight=1)
input_panel.rowconfigure(0, weight=1)

# Input Fields Grid
input_grid = tk.Frame(input_panel, bg='white')
input_grid.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

for i in range(4):
    input_grid.columnconfigure(i, weight=1)
for i in range(8):
    input_grid.rowconfigure(i, weight=1)

features = [
    "Overall Qual", "Gr Liv Area", "Total Bsmt SF", "Year Built",
    "Garage Cars", "Full Bath", "TotRms AbvGrd", "Lot Area",
    "1st Flr SF", "2nd Flr SF", "Exter Qual", "Kitchen Qual",
    "Bsmt Qual", "Year Remod/Add", "Neighborhood", "MS Zoning"
]

for idx, feature in enumerate(features):
    row = idx % 8
    col = (idx // 8) * 2
    tk.Label(input_grid, text=feature, bg='white', anchor='w').grid(
        row=row, column=col, sticky='nsew', padx=2, pady=2)
    tk.Entry(input_grid, width=15).grid(
        row=row, column=col+1, sticky='nsew', padx=2, pady=2)

# Prediction Controls
control_panel = tk.Frame(input_panel, bg='white', relief="groove")
control_panel.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

control_panel.columnconfigure(0, weight=1)
control_panel.rowconfigure(0, weight=2)
control_panel.rowconfigure(1, weight=1)

btn_predict = tk.Button(control_panel, text="PREDICT", 
                      font=("Arial", 14, 'bold'),
                      bg='#4CAF50', fg='white', relief="raised")
btn_predict.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

prediction_frame = tk.Frame(control_panel, bg='#f8f9fa', relief="sunken")
prediction_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
tk.Label(prediction_frame, text="Predicted Value:", 
       font=("Arial", 12), bg='#f8f9fa').pack(side='top', pady=5)
lbl_prediction = tk.Label(prediction_frame, text="$ ---", 
                        font=("Arial", 14, 'bold'), bg='#f8f9fa')
lbl_prediction.pack(expand=True)

# Sub-panel 3 (Similar Houses)
subpanel3 = tk.Frame(parent_panel1, bg='white', relief="solid")
subpanel3.grid(row=2, column=0, sticky="nsew", padx=2, pady=2)

canvas = tk.Canvas(subpanel3, bg='white', highlightthickness=0)
scrollbar = tk.Scrollbar(subpanel3, orient="vertical", command=canvas.yview)
content_frame = tk.Frame(canvas, bg='white')

canvas.configure(yscrollcommand=scrollbar.set)
canvas.create_window((0, 0), window=content_frame, anchor="nw")
canvas.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

for i in range(1, 11):
    tk.Label(content_frame, text=f"Similar House (example) {i}: $ {250000 + i*5000}", 
            bg='white', font=("Arial", 12)).pack(pady=3, anchor='w')

# ========== PARENT PANEL 2 ==========
parent_panel2 = tk.Frame(main_frame, bg='#f0f0f0', relief="groove")
parent_panel2.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

parent_panel2.rowconfigure(0, weight=1)
parent_panel2.rowconfigure(1, weight=1)
parent_panel2.rowconfigure(2, weight=1)
parent_panel2.rowconfigure(3, weight=1)
parent_panel2.columnconfigure(0, weight=1)

vis_panels = []
for i in range(4):
    frame = tk.Frame(parent_panel2, bg='white', relief="solid")
    frame.grid(row=i, column=0, sticky="nsew", padx=2, pady=2)
    vis_panels.append(frame)

# ================== Backend Code ==================
class HousingData:
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

    def preprocess(self):
        data = self.df.copy()
        threshold = 0.65
        
        missing_percentages = data.isnull().sum() / len(data)
        columns_to_drop = missing_percentages[missing_percentages > threshold].index
        data = data.drop(columns=columns_to_drop)

        numerical_cols = data.select_dtypes(include=np.number).columns
        categorical_cols = data.select_dtypes(exclude=np.number).columns

        for col in numerical_cols:
            if data[col].isnull().any():
                data[col] = data[col].fillna(data[col].median())

        for col in categorical_cols:
            if data[col].isnull().any():
                data[col] = data[col].fillna('Missing')

        data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
        data = data.astype(np.float32)

        target = 'SalePrice'
        numerical_features = numerical_cols.drop(target)

        data[numerical_features] = self.feature_scaler.fit_transform(data[numerical_features])
        data[target] = self.target_scaler.fit_transform(data[[target]])

        X = data.drop(target, axis=1)
        y = data[target]
        return train_test_split(X, y, test_size=0.2, random_state=42)

class TabularAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

class HousePriceRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

def train_autoencoder(model, data, epochs=50, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()
    return model

def train_regressor(model, X_train, y_train, epochs=100, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_history = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs.squeeze(), y_train)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
    
    fig = plt.figure(figsize=(4, 3))
    plt.plot(loss_history)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.tight_layout()
    plt.close(fig)
    return model, fig

def plot_dendrogram(features):
    linkage_matrix = sch.linkage(features, method='ward')
    fig = plt.figure(figsize=(4, 3))
    sch.dendrogram(linkage_matrix, truncate_mode='level', p=5)
    plt.title("Feature Clustering Dendrogram")
    plt.xlabel("Features")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.close(fig)
    return linkage_matrix, fig

def create_plots(y_test, preds, y_test_original, preds_original):
    reg_fig = plt.figure(figsize=(4, 3))
    plt.scatter(y_test_original, preds_original, alpha=0.5)
    plt.plot([y_test_original.min(), y_test_original.max()], 
             [y_test_original.min(), y_test_original.max()], 'r--')
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual vs Predicted Prices")
    plt.tight_layout()
    plt.close(reg_fig)

    residuals = y_test_original - preds_original
    res_fig = plt.figure(figsize=(4, 3))
    plt.scatter(preds_original, residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted Prices")
    plt.ylabel("Residuals")
    plt.title("Prediction Residuals")
    plt.tight_layout()
    plt.close(res_fig)
    
    return reg_fig, res_fig

def embed_plot(fig, panel):
    canvas = FigureCanvasTkAgg(fig, master=panel)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    return canvas

def show_metrics_window(baseline_mse, baseline_mae, baseline_rmse,
                        test_mse, test_mae, test_rmse):
    metrics_window = tk.Toplevel(root)
    metrics_window.title("Evaluation Metrics")
    metrics_window.geometry("400x220")  # Slightly larger window
    
    content = tk.Frame(metrics_window, padx=15, pady=15)
    content.pack(expand=True, fill='both')
    
    # Configure columns with spacing
    content.columnconfigure(0, weight=1)
    content.columnconfigure(1, weight=0)  # Spacer column
    content.columnconfigure(2, weight=1)
    
    # Baseline Metrics
    tk.Label(content, text="Baseline Metrics", font=('Arial', 10, 'bold')).grid(
        row=0, column=0, sticky='w', pady=5)
    
    tk.Label(content, text=f"MSE: {baseline_mse:.2f}").grid(row=1, column=0, sticky='w')
    tk.Label(content, text=f"MAE: {baseline_mae:.2f}").grid(row=2, column=0, sticky='w')
    tk.Label(content, text=f"RMSE: {baseline_rmse:.2f}").grid(row=3, column=0, sticky='w')
    
    # Test Metrics
    tk.Label(content, text="Test Metrics", font=('Arial', 10, 'bold')).grid(
        row=0, column=2, sticky='w', pady=5)
    
    tk.Label(content, text=f"MSE: {test_mse:.2f}").grid(row=1, column=2, sticky='w')
    tk.Label(content, text=f"MAE: {test_mae:.2f}").grid(row=2, column=2, sticky='w')
    tk.Label(content, text=f"RMSE: {test_rmse:.2f}").grid(row=3, column=2, sticky='w')
    
    # Add visual separator
    separator = tk.Frame(content, bg='#cccccc', width=2)
    separator.grid(row=0, column=1, rowspan=4, sticky='ns', padx=15)
    
    # OK button
    tk.Button(content, text="OK", command=metrics_window.destroy, width=10,
            bg='#4CAF50', fg='white').grid(row=4, column=0, columnspan=3, pady=15)
    
    metrics_window.grab_set()
    metrics_window.wait_window()

def load_data_and_train():
    data = HousingData("AmesHousing.csv")
    X_train, X_test, y_train, y_test = data.preprocess()
    
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    
    autoencoder = TabularAutoencoder(X_train_tensor.shape[1])
    autoencoder = train_autoencoder(autoencoder, X_train_tensor)
    
    encoded_features = autoencoder.encoder(X_train_tensor).detach().numpy()
    _, dendro_fig = plot_dendrogram(encoded_features)
    
    regressor = HousePriceRegressor(X_train_tensor.shape[1])
    regressor, loss_fig = train_regressor(regressor, X_train_tensor, y_train_tensor)
    
    with torch.no_grad():
        preds = regressor(X_test_tensor).numpy()
    y_test_original = data.target_scaler.inverse_transform(y_test.values.reshape(-1, 1))
    preds_original = data.target_scaler.inverse_transform(preds.reshape(-1, 1))
    
    reg_fig, res_fig = create_plots(y_test, preds, y_test_original, preds_original)
    
    # Calculate metrics
    baseline_pred = np.full_like(y_test_original, 
                               data.target_scaler.inverse_transform(
                                   y_train.values.reshape(-1, 1)).mean())
    baseline_mse = mean_squared_error(y_test_original, baseline_pred)
    baseline_mae = mean_absolute_error(y_test_original, baseline_pred)
    baseline_rmse = np.sqrt(baseline_mse)
    
    test_mse = mean_squared_error(y_test_original, preds_original)
    test_mae = mean_absolute_error(y_test_original, preds_original)
    test_rmse = np.sqrt(test_mse)
    
    # Show metrics window
    show_metrics_window(baseline_mse, baseline_mae, baseline_rmse,
                       test_mse, test_mae, test_rmse)
    
    # Embed plots
    embed_plot(loss_fig, vis_panels[0])
    embed_plot(dendro_fig, vis_panels[1])
    embed_plot(reg_fig, vis_panels[2])
    embed_plot(res_fig, vis_panels[3])

load_data_and_train()
root.mainloop()