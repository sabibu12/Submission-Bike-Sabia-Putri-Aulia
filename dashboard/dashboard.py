import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from scipy.stats import pearsonr

sns.set(style='darkgrid')

hari_df = pd.read_csv("dashboard/hari_all_data.csv")
jam_df = pd.read_csv("dashboard/jam_all_data.csv")

hari_df['dateday'] = pd.to_datetime(hari_df['dateday'])
jam_df['dateday'] = pd.to_datetime(jam_df['dateday'])

st.sidebar.header("Filter Data")

selected_year = st.sidebar.multiselect(
    "Pilih Tahun",
    options=hari_df['year'].unique(),
    default=hari_df['year'].unique()
)

selected_season = st.sidebar.multiselect(
    "Pilih Musim",
    options=hari_df['season'].unique(),
    default=hari_df['season'].unique()
)

filtered_hari = hari_df[
    (hari_df['year'].isin(selected_year)) &
    (hari_df['season'].isin(selected_season))
]

st.title("Dashboard Analisis Penyewaan Sepeda")
st.write("Analisis berbasis waktu, musim, cuaca, dan faktor lingkungan")

total_penyewaan = filtered_hari['count'].sum()
average_penyewaan = filtered_hari['count'].mean()

col1, col2 = st.columns(2)
col1.metric("Total Penyewaan", f"{total_penyewaan:,.0f}")
col2.metric("Rata-rata Penyewaan", f"{average_penyewaan:.2f}")

st.subheader("Penyewaan Sepeda Per Hari")

per_hari = filtered_hari.groupby('weekday')['count'].sum()

urutan_hari = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']
per_hari = per_hari.reindex(urutan_hari)

fig, ax = plt.subplots()
ax.plot(per_hari.index, per_hari.values, marker='o')
ax.set_title("Jumlah Penyewaan Per Hari")
ax.set_xlabel("Hari")
ax.set_ylabel("Jumlah")

st.pyplot(fig)

top_day = per_hari.idxmax()
top_value = per_hari.max()

st.success(f"Hari dengan penyewaan tertinggi adalah: {top_day} ({top_value:,.0f})")

st.subheader("Perbandingan Pengguna Sepeda Casual vs Registered Per Musim")

per_musim = filtered_hari.groupby('season')[['casual', 'registered']].mean()

fig, ax = plt.subplots()
per_musim.plot(kind='bar', ax=ax)
ax.set_title("Rata-rata Penyewaan Per Musim")
ax.set_ylabel("Rata-rata")
plt.xticks(rotation=0)

st.pyplot(fig)

st.info("Pengguna casual lebih sensitif terhadap musim dibandingkan registered.")

st.subheader("Distribusi Penyewaan Per Jam")

per_jam = jam_df.groupby('hour')['count'].mean()

fig, ax = plt.subplots()
ax.plot(per_jam.index, per_jam.values, marker='o')
ax.set_title("Rata-rata Penyewaan Per Jam")
ax.set_xlabel("Jam")
ax.set_ylabel("Rata-rata")

st.pyplot(fig)

peak_hour = per_jam.idxmax()
peak_value = per_jam.max()

st.success(f"Jam puncak penyewaan adalah: {peak_hour}:00 ({peak_value:.2f})")

st.subheader("Pengaruh Cuaca")

cuaca = filtered_hari.groupby('weather')[['casual', 'registered', 'count']].agg({
    'casual': 'mean',
    'registered': 'mean',
    'count': 'sum'
})

fig, ax = plt.subplots()

x = range(len(cuaca.index))
width = 0.35

ax.bar([i - width/2 for i in x], cuaca['casual'], width, label='Casual')
ax.bar([i + width/2 for i in x], cuaca['registered'], width, label='Registered')

ax.set_xticks(x)
ax.set_xticklabels(cuaca.index, rotation=30)
ax.set_title("Rata-rata Penyewaan Berdasarkan Cuaca")
ax.legend()

st.pyplot(fig)

best_weather = cuaca['count'].idxmax()
st.success(f"Cuaca dengan penyewaan tertinggi adalah {best_weather}")

st.info("Pengguna casual lebih terpengaruh kondisi cuaca dibandingkan registered.")

st.subheader("Analisis Korelasi Windspeed")

corr, p_value = pearsonr(filtered_hari['windspeed'], filtered_hari['count'])

st.write(f"Korelasi: {round(corr,3)}")
st.write(f"P-value: {p_value:.5f}")

fig, ax = plt.subplots()

sns.regplot(
    x='windspeed',
    y='count',
    data=filtered_hari,
    ax=ax
)

ax.set_title("Windspeed vs Penyewaan")

st.pyplot(fig)

if abs(corr) < 0.3:
    st.warning("Hubungan antara windspeed dan penyewaan lemah.")
else:
    st.success("Terdapat hubungan yang cukup kuat.")

if p_value < 0.05:
    st.success("Hubungan signifikan secara statistik.")
else:
    st.warning("Hubungan tidak signifikan secara statistik.")
