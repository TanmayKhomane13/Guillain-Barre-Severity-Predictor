import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import os

def computeGradient(x, y, w, b):
    m, n = x.shape
    dj_dw = np.zeros((n, ))
    dj_db = 0.

    for i in range(m):
        f_wb_i = np.dot(x[i], w) + b
        err_i = f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] += err_i.item() * x[i, j].item()
        dj_db += err_i
    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db

def gradientDescent(x, y, w_in, b_in, alpha, num_iters):
    w = copy.deepcopy(w_in)
    b = b_in
    for i in range(num_iters):
        dj_dw, dj_db = computeGradient(x, y, w, b)

        # update the parameters
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
    
    return w, b


# load dataset
df = pd.read_csv("GBS_Synthetic_Data.csv")

# training data
x_train = df[["Muscle Weakness Level", "Reflex Loss Percentage", "CSF Protein Level (mg/dL)"]].values
y_train = df[["Severity Score"]].values
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)  # Normalize features

# initial settings
w_init = np.zeros_like(x_train[0])
b_init = 0.
alph = 0.001
iters = 10000

w_out, b_out = gradientDescent(x_train, y_train, w_init, b_init, alph, iters)

# user input
os.system('clear')
mwl = float(input("Enter Muscle Weakness Level (eg. scale 0-10): "))
rlp = float(input("Enter Reflex Loss Percentage (eg. 0-100%): "))
csf = float(input("Enter CSF Protein Level (mg/dL): "))
user = np.array([mwl, rlp, csf])
user_scaled = scaler.transform(user.reshape(1, -1))

# predictions
y_user = np.dot(user_scaled, w_out) + b_out
print(f"Severity Score: {y_user[0]}")

if y_user[0] >= 0 and y_user[0] <= 40:
    print("Risk: Mild\nMedical Advice: manageable with supportive care")
elif y_user[0] >= 41 and y_user[0] <= 70:
    print("Risk: Moderate\nMedical Advice: hospitalization needed!")
else:
    print("Risk: Severe\nMedical Advice: ICU care")

# --------------------------------- plotting -----------------------
# 3D Plotting
fig = plt.figure(figsize = (12, 9), label = "Guillain Barre Syndrome Severity")
ax = fig.add_subplot(111, projection = '3d')

# extracting feature data
x = df["Muscle Weakness Level"]
y = df["Reflex Loss Percentage"]
z = df["CSF Protein Level (mg/dL)"]

# scatter plot of data
c = y_train # severity score as color

scatter = ax.scatter(x, y, z, c = c, cmap = "viridis")

# add labels and title
ax.set_xlabel("Muscle Weakness Level")
ax.set_ylabel("Reflex Loss Percentage")
ax.set_zlabel("CSF Protein Level (mg/dL)")
ax.set_title("3D scatter plot of Features vs Severity Score")

fig.colorbar(scatter, ax = ax, label = "Severity Score")

plt.show()
