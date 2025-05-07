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
from sklearn.metrics import mean_squared_error, mean_absolute_error, silhouette_score, davies_bouldin_score
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
from PIL import Image, ImageTk

# ================== Frontend Code ==================
root = tk.Tk()
root.title("House Price Prediction & Clustering Project")
root.geometry("1400x1000")

main_frame = tk.Frame(root)
main_frame.pack(expand=True, fill="both")
main_frame.columnconfigure(0, weight=3)
main_frame.columnconfigure(1, weight=1)
main_frame.rowconfigure(0, weight=1)

# ────────── PANEL 1 ──────────
parent_panel1 = tk.Frame(main_frame, bg='#f0f0f0', relief="groove")
parent_panel1.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
parent_panel1.rowconfigure(0, weight=1)
parent_panel1.rowconfigure(1, weight=3)
parent_panel1.rowconfigure(2, weight=2)
parent_panel1.columnconfigure(0, weight=1)

# Image + Info
subpanel1 = tk.Frame(parent_panel1, bg='white', relief="solid")
subpanel1.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
subpanel1.columnconfigure(0, weight=1)
subpanel1.columnconfigure(1, weight=1)
panel1a = tk.Frame(subpanel1, bg='white', relief="solid")
panel1a.grid(row=0, column=0, sticky="nsew")
try:
    img = Image.open("house.png").resize((300,200), Image.LANCZOS)
    photo = ImageTk.PhotoImage(img)
    tk.Label(panel1a, image=photo, bg='white').pack(expand=True)
except:
    tk.Label(panel1a, text="Image not found", bg='white', fg='red').pack(expand=True)
panel1b = tk.Frame(subpanel1, bg='white', relief="solid")
panel1b.grid(row=0, column=1, sticky="nsew")
info = """House Price Prediction & Clustering Project

Team Members:
- Omar Obaid
- Yamen Attar

Machine Learning Project"""
tk.Label(panel1b, text=info, bg='white', font=("Arial",12), justify='left').pack(expand=True, padx=10)

# Input fields
input_panel = tk.Frame(parent_panel1, bg='white', relief="solid")
input_panel.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)
input_panel.columnconfigure(0, weight=3)
input_panel.columnconfigure(1, weight=1)
input_panel.rowconfigure(0, weight=1)

input_grid = tk.Frame(input_panel, bg='white')
input_grid.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
for i in range(4): input_grid.columnconfigure(i, weight=1)
for i in range(8): input_grid.rowconfigure(i, weight=1)

features = [
    "Overall Qual","Gr Liv Area","Total Bsmt SF","Year Built",
    "Garage Cars","Full Bath","TotRms AbvGrd","Lot Area",
    "1st Flr SF","2nd Flr SF","Exter Qual","Kitchen Qual",
    "Bsmt Qual","Year Remod/Add","Neighborhood","MS Zoning"
]

default_values = {
    "Overall Qual": "15",
    "Gr Liv Area": "1656",
    "Total Bsmt SF": "1080",
    "Year Built": "1960",
    "Garage Cars": "2",
    "Full Bath": "1",
    "TotRms AbvGrd": "7",
    "Lot Area": "31770",
    "1st Flr SF": "1656",
    "2nd Flr SF": "0",
    "Exter Qual": "TA",
    "Kitchen Qual": "TA",
    "Bsmt Qual": "TA",
    "Year Remod/Add": "1960",
    "Neighborhood": "NAmes",
    "MS Zoning": "RL"
}

entries = {}
for idx, feat in enumerate(features):
    r, c = idx%8, (idx//8)*2
    tk.Label(input_grid, text=feat, bg='white', anchor='w')\
      .grid(row=r, column=c, sticky='nsew', padx=2, pady=2)
    e = tk.Entry(input_grid, width=15)
    e.grid(row=r, column=c+1, sticky='nsew', padx=2, pady=2)
    e.insert(0, default_values.get(feat, ""))
    entries[feat] = e

# Predict controls
control = tk.Frame(input_panel, bg='white', relief="groove")
control.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
control.columnconfigure(0, weight=1)
control.rowconfigure(0, weight=2)
control.rowconfigure(1, weight=1)

btn_predict = tk.Button(control, text="PREDICT", font=("Arial",14,"bold"),
                        bg='#4CAF50', fg='white')
btn_predict.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
prediction_frame = tk.Frame(control, bg='#f8f9fa', relief="sunken")
prediction_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
tk.Label(prediction_frame, text="Predicted Value:",
         font=("Arial",12), bg='#f8f9fa').pack(side='top', pady=5)
lbl_prediction = tk.Label(prediction_frame, text="$ ---",
                          font=("Arial",14,"bold"), bg='#f8f9fa')
lbl_prediction.pack(expand=True)

# Similar Houses panel
subpanel3 = tk.Frame(parent_panel1, bg='white', relief="solid")
subpanel3.grid(row=2, column=0, sticky="nsew", padx=2, pady=2)
canvas = tk.Canvas(subpanel3, bg='white', highlightthickness=0)
scroll = tk.Scrollbar(subpanel3, orient="vertical", command=canvas.yview)
content_frame = tk.Frame(canvas, bg='white')
canvas.configure(yscrollcommand=scroll.set)
canvas.create_window((0,0), window=content_frame, anchor="nw")
canvas.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
canvas.pack(side="left", fill="both", expand=True)
scroll.pack(side="right", fill="y")

# ────────── PANEL 2 ──────────
parent_panel2 = tk.Frame(main_frame, bg='#f0f0f0', relief="groove")
parent_panel2.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
for i in range(4): parent_panel2.rowconfigure(i, weight=1)
parent_panel2.columnconfigure(0, weight=1)
vis_panels = [tk.Frame(parent_panel2, bg='white', relief="solid") for _ in range(4)]
for i,frame in enumerate(vis_panels):
    frame.grid(row=i, column=0, sticky="nsew", padx=2, pady=2)

# ────────── Backend ──────────
class HousingData:
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.kept_columns = None
        self.numerical_cols = None
        self.categorical_cols = None
        self.numerical_medians = None
        self.processed_columns = None
        self.numerical_features = None

    def preprocess(self):
        data = self.df.copy()
        thresh = 0.65
        miss = data.isnull().sum()/len(data)
        drop = miss[miss>thresh].index
        data.drop(columns=drop, inplace=True)
        self.kept_columns = data.columns.tolist()

        num = data.select_dtypes(include=np.number).columns
        cat = data.select_dtypes(exclude=np.number).columns
        self.numerical_cols = num.tolist()
        self.categorical_cols = cat.tolist()
        self.numerical_medians = data[num].median()

        for c in num:
            if data[c].isnull().any():
                data[c] = data[c].fillna(self.numerical_medians[c])
        for c in cat:
            if data[c].isnull().any():
                data[c] = data[c].fillna('Missing')

        data = pd.get_dummies(data, columns=cat, drop_first=True).astype(np.float32)
        target = 'SalePrice'
        feats = num.drop(target)
        self.numerical_features = feats.tolist()
        data[feats] = self.feature_scaler.fit_transform(data[feats])
        data[target] = self.target_scaler.fit_transform(data[[target]])
        self.processed_columns = data.drop(target,axis=1).columns.tolist()

        X = data.drop(target,axis=1)
        y = data[target]
        return train_test_split(X, y, test_size=0.2, random_state=42)

class TabularAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim,128), nn.ReLU(),
            nn.Linear(128,encoding_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim,128), nn.ReLU(),
            nn.Linear(128,input_dim)
        )
    def forward(self,x):
        return self.decoder(self.encoder(x))

class HousePriceRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim,128), nn.ReLU(),
            nn.Linear(128,64), nn.ReLU(),
            nn.Linear(64,1)
        )
    def forward(self,x):
        return self.model(x)

def train_autoencoder(model, data, epochs=50, lr=0.001):
    crit = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=lr)
    model.train()
    history = []
    for _ in range(epochs):
        opt.zero_grad()
        out = model(data)
        loss = crit(out, data)
        loss.backward()
        opt.step()
        history.append(loss.item())
    
    fig = plt.figure(figsize=(4,3))
    plt.plot(history)
    plt.title("Encoder Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.tight_layout()
    plt.close(fig)
    return model, fig

def train_regressor(model, X_tr, y_tr, epochs=100, lr=0.001):
    crit = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=lr)
    history = []
    for _ in range(epochs):
        opt.zero_grad()
        preds = model(X_tr)
        loss = crit(preds.squeeze(), y_tr)
        loss.backward()
        opt.step()
        history.append(loss.item())
    fig = plt.figure(figsize=(4,3))
    plt.plot(history)
    plt.title("Regressor Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.tight_layout()
    plt.close(fig)
    return model, fig

def create_plots(y_test_orig, preds_orig):
    fig = plt.figure(figsize=(4,3))
    plt.scatter(y_test_orig, preds_orig, alpha=0.5)
    plt.plot([y_test_orig.min(),y_test_orig.max()],
             [y_test_orig.min(),y_test_orig.max()],'r--')
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual vs Predicted Prices")
    plt.tight_layout()
    plt.close(fig)
    return fig

def plot_gmm_clusters(X, gmm):
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X)
    labels = gmm.predict(X)
    
    metrics = {
        'Silhouette': silhouette_score(X, labels),
        'BIC': gmm.bic(X),
        'AIC': gmm.aic(X),
        'Davies-Bouldin': davies_bouldin_score(X, labels)
    }
    
    fig = plt.figure(figsize=(4,3))
    ax = fig.add_subplot(111)
    ax.scatter(X2[:,0], X2[:,1], c=labels, cmap='tab10', s=20, alpha=0.6)
    for i in range(gmm.n_components):
        mean_pca = pca.transform(gmm.means_[i].reshape(1,-1))[0]
        cov_pca = pca.components_ @ gmm.covariances_[i] @ pca.components_.T
        vals, vecs = np.linalg.eigh(cov_pca)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:,order]
        w,h = 2*np.sqrt(vals)
        ang = np.degrees(np.arctan2(vecs[1,0], vecs[0,0]))
        ellipse = Ellipse(mean_pca, w, h, angle=ang, alpha=0.3, edgecolor='black', facecolor='none')
        ax.add_patch(ellipse)
    ax.set_title("GMM Clusters (PCA projection)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    plt.tight_layout(); plt.close(fig)
    return fig, metrics

def embed_plot(fig, panel):
    c = FigureCanvasTkAgg(fig, master=panel)
    c.draw()
    c.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    return c

def show_metrics_window(b_mse,b_mae,b_rmse,t_mse,t_mae,t_rmse):
    w = tk.Toplevel(root); w.title("Evaluation Metrics"); w.geometry("400x220")
    cnt = tk.Frame(w, padx=15, pady=15); cnt.pack(expand=True, fill='both')
    cnt.columnconfigure(0, weight=1); cnt.columnconfigure(1, weight=0); cnt.columnconfigure(2, weight=1)
    tk.Label(cnt, text="Baseline Metrics", font=('Arial',10,'bold')).grid(row=0,column=0,sticky='w', pady=5)
    tk.Label(cnt, text=f"MSE: {b_mse:.2f}").grid(row=1,column=0,sticky='w')
    tk.Label(cnt, text=f"MAE: {b_mae:.2f}").grid(row=2,column=0,sticky='w')
    tk.Label(cnt, text=f"RMSE: {b_rmse:.2f}").grid(row=3,column=0,sticky='w')
    tk.Label(cnt, text="Test Metrics", font=('Arial',10,'bold')).grid(row=0,column=2,sticky='w', pady=5)
    tk.Label(cnt, text=f"MSE: {t_mse:.2f}").grid(row=1,column=2,sticky='w')
    tk.Label(cnt, text=f"MAE: {t_mae:.2f}").grid(row=2,column=2,sticky='w')
    tk.Label(cnt, text=f"RMSE: {t_rmse:.2f}").grid(row=3,column=2,sticky='w')
    sep = tk.Frame(cnt, bg='#cccccc', width=2); sep.grid(row=0,column=1, rowspan=4, sticky='ns', padx=15)
    tk.Button(cnt, text="OK", command=w.destroy, width=10, bg='#4CAF50', fg='white')\
      .grid(row=4, column=0, columnspan=3, pady=15)
    w.grab_set(); w.wait_window()

def show_cluster_metrics_window(metrics):
    w = tk.Toplevel(root)
    w.title("Clustering Quality Metrics")
    w.geometry("300x220")
    cnt = tk.Frame(w, padx=15, pady=15)
    cnt.pack(expand=True, fill='both')
    
    tk.Label(cnt, text="Clustering Metrics", font=('Arial',12,'bold')).pack(pady=5)
    
    for name, value in metrics.items():
        row = tk.Frame(cnt)
        row.pack(fill='x', pady=2)
        tk.Label(row, text=f"{name}:", width=15, anchor='w').pack(side='left')
        tk.Label(row, text=f"{value:.3f}").pack(side='right')
    
    tk.Button(w, text="OK", command=w.destroy, width=10, bg='#4CAF50', fg='white')\
      .pack(pady=10)
    w.grab_set()
    w.wait_window()

def create_new_sample(hd, ui):
    new = pd.DataFrame(columns=hd.kept_columns)
    for c in hd.kept_columns:
        if c in ui:
            new[c] = [ui[c]]
        else:
            new[c] = [hd.numerical_medians[c] if c in hd.numerical_cols else 'Missing']
    new = pd.get_dummies(new, columns=hd.categorical_cols, drop_first=True)
    new = new.reindex(columns=hd.processed_columns, fill_value=0)
    new[hd.numerical_features] = hd.feature_scaler.transform(new[hd.numerical_features])
    return new

# Globals for clustering
X_train_np = None
y_train_orig_np = None

def predict_price():
    user_inputs = {}
    num_feats = [
        "Overall Qual","Gr Liv Area","Total Bsmt SF","Year Built",
        "Garage Cars","Full Bath","TotRms AbvGrd","Lot Area",
        "1st Flr SF","2nd Flr SF","Year Remod/Add"
    ]
    for f in features:
        v = entries[f].get()
        if f in num_feats:
            try: user_inputs[f] = float(v) if v else housing_data.numerical_medians[f]
            except: user_inputs[f] = housing_data.numerical_medians[f]
        else:
            user_inputs[f] = v if v else 'Missing'

    try:
        ns = create_new_sample(housing_data, user_inputs)
        tns = torch.tensor(ns.values, dtype=torch.float32)
        with torch.no_grad():
            p = regressor(tns)
        price = housing_data.target_scaler.inverse_transform(p.numpy())[0][0]
        lbl_prediction.config(text=f"$ {price:,.2f}")
    except Exception as e:
        lbl_prediction.config(text="Error"); print(e)
        return

    global X_train_np, y_train_orig_np
    combined = np.vstack([X_train_np, ns.values])
    gmm = GaussianMixture(n_components=5, random_state=42)
    labels = gmm.fit_predict(combined)
    new_lbl = labels[-1]

    for w in content_frame.winfo_children(): w.destroy()
    idxs = np.where(labels[:-1]==new_lbl)[0]
    for i in range(min(len(idxs),12)):  
        tk.Label(content_frame,
                text=f"Similar House {i+1}: $ {y_train_orig_np[idxs[i]]:,.2f}",
                bg='white', font=("Arial",12)).pack(pady=3, anchor='w')

btn_predict.config(command=predict_price)

housing_data = None
regressor = None
autoencoder = None

def load_data_and_train():
    global housing_data, regressor, autoencoder, X_train_np, y_train_orig_np
    housing_data = HousingData("AmesHousing.csv")
    X_train, X_test, y_train, y_test = housing_data.preprocess()

    X_train_np = X_train.values
    y_train_orig_np = housing_data.target_scaler.inverse_transform(y_train.values.reshape(-1,1)).flatten()

    X_tr = torch.tensor(X_train_np, dtype=torch.float32)
    y_tr = torch.tensor(y_train.values, dtype=torch.float32)
    X_te = torch.tensor(X_test.values, dtype=torch.float32)

    autoencoder, ae_loss_fig = train_autoencoder(TabularAutoencoder(X_tr.shape[1]), X_tr)

    encoded = autoencoder.encoder(X_tr).detach().numpy()
    gmm = GaussianMixture(n_components=5, random_state=42).fit(encoded)
    gmm_fig, cluster_metrics = plot_gmm_clusters(encoded, gmm)
    show_cluster_metrics_window(cluster_metrics)

    regressor, reg_loss_fig = train_regressor(HousePriceRegressor(X_tr.shape[1]), X_tr, y_tr)

    with torch.no_grad():
        preds = regressor(X_te).numpy()
    y_test_orig = housing_data.target_scaler.inverse_transform(y_test.values.reshape(-1,1))
    preds_orig = housing_data.target_scaler.inverse_transform(preds.reshape(-1,1))

    baseline = np.full_like(y_test_orig, y_train_orig_np.mean())
    b_mse = mean_squared_error(y_test_orig, baseline)
    b_mae = mean_absolute_error(y_test_orig, baseline)
    t_mse = mean_squared_error(y_test_orig, preds_orig)
    t_mae = mean_absolute_error(y_test_orig, preds_orig)

    show_metrics_window(
        b_mse, b_mae, np.sqrt(b_mse),
        t_mse, t_mae, np.sqrt(t_mse)
    )
    
    actual_vs_pred_fig = create_plots(y_test_orig, preds_orig)

    embed_plot(reg_loss_fig, vis_panels[0])
    embed_plot(ae_loss_fig, vis_panels[1])
    embed_plot(gmm_fig, vis_panels[2])
    embed_plot(actual_vs_pred_fig, vis_panels[3])

load_data_and_train()
root.mainloop()