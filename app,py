import streamlit as st
from scipy.optimize import linprog
import math
import sympy as sp
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(layout="wide")
st.title("Aplikasi Model Matematika Industri")

# Tabs
menu = st.tabs([
    "Optimasi Produksi", 
    "Model Persediaan (EOQ)", 
    "Model Antrian (M/M/1)", 
    "Model Integral Energi Listrik"
])

# 1. Optimasi Produksi
with menu[0]:
    st.header("Optimasi Produksi (Linear Programming)")
    st.write("Contoh: Produksi Blender dan Pemanggang Roti")

    # Input
    st.subheader("Input")
    c1 = st.number_input("Keuntungan per unit Produk A (Rp)", value=40)
    c2 = st.number_input("Keuntungan per unit Produk B (Rp)", value=60)
    a1 = st.number_input("Waktu mesin per unit Produk A (jam)", value=2)
    a2 = st.number_input("Waktu mesin per unit Produk B (jam)", value=3)
    total_time = st.number_input("Total waktu mesin tersedia (jam)", value=100)

    # Hitung
    if st.button("Hitung Optimasi"):
        res = linprog(c=[-c1, -c2], A_ub=[[a1, a2]], b_ub=[total_time], bounds=[(0, None), (0, None)])
        if res.success:
            st.success("Solusi Ditemukan")
            xA = res.x[0]
            xB = res.x[1]
            st.write(f"Produksi optimal Produk A: {xA:.2f} unit")
            st.write(f"Produksi optimal Produk B: {xB:.2f} unit")
            st.write(f"Keuntungan maksimal: Rp{(-res.fun):,.0f}")

            # Grafik visualisasi area feasible
            A_vals = np.linspace(0, total_time/a1, 100)
            B_vals = (total_time - a1*A_vals) / a2

            fig, ax = plt.subplots()
            ax.plot(A_vals, B_vals, label="Batasan: 2A + 3B = 100")
            ax.fill_between(A_vals, 0, B_vals, alpha=0.3, label="Area Feasible")
            ax.plot(xA, xB, 'ro', label="Solusi Optimal")
            ax.set_xlabel("Produk A")
            ax.set_ylabel("Produk B")
            ax.legend()
            st.pyplot(fig)
        else:
            st.error("Solusi tidak ditemukan")

# 2. Model Persediaan (EOQ)
with menu[1]:
    st.header("Model Persediaan EOQ")
    st.subheader("Input")
    D = st.number_input("Permintaan tahunan (D)", value=10000)
    S = st.number_input("Biaya pemesanan per pesanan (S)", value=50000)
    H = st.number_input("Biaya penyimpanan per unit per tahun (H)", value=2000)

    if st.button("Hitung EOQ"):
        eoq = math.sqrt((2 * D * S) / H)
        st.success(f"Jumlah pemesanan optimal (EOQ): {eoq:.2f} unit")

        # Grafik Biaya Total vs Q
        Q = np.linspace(1, 2*eoq, 200)
        TC = (D/Q)*S + (Q/2)*H

        fig, ax = plt.subplots()
        ax.plot(Q, TC, label="Total Cost")
        ax.axvline(eoq, color='r', linestyle='--', label=f'EOQ ≈ {eoq:.2f}')
        ax.set_xlabel("Jumlah Pemesanan (Q)")
        ax.set_ylabel("Biaya Total")
        ax.set_title("Kurva Biaya Total EOQ")
        ax.legend()
        st.pyplot(fig)

# 3. Model Antrian (M/M/1)
with menu[2]:
    st.header("Model Antrian M/M/1")
    st.subheader("Input")
    lam = st.number_input("Rata-rata kedatangan pelanggan per jam (λ)", value=10.0)
    mu = st.number_input("Rata-rata pelayanan pelanggan per jam (μ)", value=12.0)

    if st.button("Hitung Antrian"):
        if lam < mu:
            rho = lam / mu
            W = 1 / (mu - lam)
            Wq = lam / (mu * (mu - lam))
            L = lam * W
            st.write(f"Tingkat Utilisasi (ρ): {rho:.2f}")
            st.write(f"Rata-rata waktu dalam sistem (W): {W:.2f} jam")
            st.write(f"Rata-rata waktu tunggu (Wq): {Wq:.2f} jam")
            st.write(f"Rata-rata jumlah pelanggan dalam sistem (L): {L:.2f}")

            # Visualisasi Wq terhadap λ
            lam_vals = np.linspace(0.1, mu - 0.01, 200)
            Wq_vals = lam_vals / (mu * (mu - lam_vals))

            fig, ax = plt.subplots()
            ax.plot(lam_vals, Wq_vals, label="Waktu Tunggu (Wq)")
            ax.axvline(lam, color='r', linestyle='--', label=f'λ Sekarang = {lam}')
            ax.set_xlabel("λ (Pelanggan/jam)")
            ax.set_ylabel("Waktu Tunggu Rata-rata (jam)")
            ax.set_title("Wq terhadap Variasi λ")
            ax.legend()
            st.pyplot(fig)
        else:
            st.error("λ harus lebih kecil dari μ agar sistem stabil")

# 4. Model Integral (Energi Listrik)
with menu[3]:
    st.header("Model Integral Energi Listrik")
    st.subheader("Fungsi Laju Konsumsi Energi")
    st.latex(r"""P(t) = 20 + 5t - 0.5t^2""")

    t = sp.Symbol('t')
    P = 20 + 5*t - 0.5*t**2
    integral = sp.integrate(P, (t, 0, 10))

    st.write(f"Total energi listrik yang digunakan (kWh): {integral.evalf():.2f}")

    # Visualisasi Grafik Konsumsi
    st.subheader("Grafik Konsumsi Energi Harian")
    T = np.linspace(0, 10, 100)
    P_func = sp.lambdify(t, P, 'numpy')
    Y = P_func(T)

    fig, ax = plt.subplots()
    ax.plot(T, Y, label="P(t) = 20 + 5t - 0.5t²")
    ax.fill_between(T, Y, alpha=0.3)
    ax.set_xlabel("Waktu (jam)")
    ax.set_ylabel("Konsumsi Energi (kW)")
    ax.legend()
    st.pyplot(fig)
