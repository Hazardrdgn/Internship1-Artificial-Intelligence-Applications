import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ============================================================
# 1) VERİ YÜKLEME
# ============================================================
print("Veri Yükleniyor...")
df = pd.read_csv("secondary_sales.csv")
dist = pd.read_csv("district_prices_monthly.csv")  # ilçe/ay bazlı piyasa endeksi

# ============================================================
# 2) TARİH ÖZELLİKLERİ (asıl eksik olan kısım buydu!)
# ------------------------------------------------------------
# ESKİ KODDA 'date_listed' veri sızıntısı sanılıp direkt siliniyordu.
# Oysa bu bir sızıntı DEĞİL: ilan tarihi, satış anında zaten bilinen
# bir bilgi. Ve veri setinde 2020->2026 arası ortalama USD fiyatı
# ~83.000$ -> ~290.000$'a çıkmış (TL enflasyonu / kur etkisiyle).
# Bu dev trendi modele hiç göstermeyince model yıllar arası fiyat
# farkını "gürültü" gibi algılıyor ve tahminler sapıyordu.
# ============================================================
df['date_listed'] = pd.to_datetime(df['date_listed'], format='%d/%m/%Y')
df['list_year'] = df['date_listed'].dt.year
df['list_month'] = df['date_listed'].dt.month
df['year_month'] = df['date_listed'].dt.to_period('M').astype(str)
# 2020-01'den itibaren geçen ay sayısı -> modelin doğrusal/monoton
# zaman trendini (piyasa yükselişini) yakalamasını kolaylaştırır
df['months_since_start'] = (df['list_year'] - 2020) * 12 + df['list_month']
# binanın ilan anındaki yaşı, ham 'year_built' yerine çok daha anlamlı
df['property_age'] = df['list_year'] - df['year_built']

# ============================================================
# 3) İLÇE BAZLI PİYASA ENDEKSİ (bir ay GECİKMELİ / lag-1)
# ------------------------------------------------------------
# district_prices_monthly.csv, o ilçe-ay'daki ORTALAMA m2 fiyatını
# içeriyor. Aynı ayın ortalamasını doğrudan kullanmak hafif bir
# sızıntı riski taşır (o ay içindeki satış zaten ortalamaya dahil).
# Bu yüzden bir önceki ayın endeksini kullanıyoruz: "bu ilçede bir
# ay önce piyasa neredeydi?" -> tamamen geçmişe ait, sızıntısız,
# ama yine de güçlü bir "karşılaştırmalı emsal" sinyali veriyor.
# ============================================================
dist['ym_period'] = pd.PeriodIndex(dist['year_month'], freq='M')
dist_lag = dist.copy()
dist_lag['ym_period'] = dist_lag['ym_period'] + 1
dist_lag['year_month'] = dist_lag['ym_period'].astype(str)
dist_lag = dist_lag[['district', 'year_month', 'secondary_price_per_m2_usd', 'rental_per_m2_monthly_try']]
dist_lag = dist_lag.rename(columns={
    'secondary_price_per_m2_usd': 'district_price_m2_usd_lag1',
    'rental_per_m2_monthly_try': 'district_rent_m2_try_lag1'
})

df = df.merge(dist_lag, on=['district', 'year_month'], how='left')

# veri setinin ilk ayı (2020-01) için "bir önceki ay" yok; bu satırlarda
# geriye düşüp aynı ayın endeksiyle dolduruyoruz (marjinal, ~%1 satır)
same_month = dist[['district', 'year_month', 'secondary_price_per_m2_usd']].rename(
    columns={'secondary_price_per_m2_usd': 'fallback'})
df = df.merge(same_month, on=['district', 'year_month'], how='left')
df['district_price_m2_usd_lag1'] = df['district_price_m2_usd_lag1'].fillna(df['fallback'])
df = df.drop(columns=['fallback'])

# ============================================================
# 4) GERÇEK VERİ SIZINTISI YAPAN SÜTUNLARI ÇIKARMA
# ------------------------------------------------------------
# Bunlar price_usd'den ARİTMETİK olarak türetildiği için gerçek sızıntı:
#   price_try            = price_usd * kur
#   price_per_m2_try/usd = price_usd / alan
# 'transit_station' ise 85 farklı istasyon adı taşıyan, neredeyse
# id gibi davranan bir sütun; zaten 'to_levent_km', 'to_sultanahmet_km'
# ve 'transit_distance_min' konumu yeterince temsil ediyor, o yüzden
# gereksiz gürültü/aşırı öğrenme riskine karşı çıkarıyoruz.
# ============================================================
cols_to_drop = [
    'id', 'date_listed', 'year_month',
    'price_try', 'price_per_m2_try', 'price_per_m2_usd',
    'transit_station',
]
df = df.drop(columns=cols_to_drop)

# ============================================================
# 5) EKSİK VERİ TEMİZLİĞİ
# ============================================================
df = df.dropna(subset=['price_usd', 'gross_area_m2', 'bedrooms'])

for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna('Unknown')

# ============================================================
# 6) KATEGORİK VERİLERİ SAYISALLAŞTIRMA (One-Hot Encoding)
# ============================================================
df = pd.get_dummies(df, drop_first=True)

# ============================================================
# 7) X / y AYRIMI VE TRAIN-TEST BÖLME
# ============================================================
X = df.drop('price_usd', axis=1)
y = df['price_usd']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# hedefi logaritmik dönüştürüyoruz (uçuk fiyatlı yalılar modeli bozmasın diye)
y_train_log = np.log1p(y_train)

# ============================================================
# 8) MODEL KURULUMU VE EĞİTİMİ
# ============================================================
print("Model eğitiliyor...")
model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)
model.fit(X_train, y_train_log)

# ============================================================
# 9) DEĞERLENDİRME
# ============================================================
predictions_log = model.predict(X_test)
predictions = np.expm1(predictions_log)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("-" * 30)
print(f"Ortalama Mutlak Hata (MAE): ${mae:,.2f}")
print(f"Başarı Oranı (R2 Score): %{r2 * 100:.2f}")
print("-" * 30)

print("Gerçek Fiyat: $", f"{y_test.iloc[0]:,.2f}")
print("Modelin Tahmini: $", f"{predictions[0]:,.2f}")

# hangi özellikler en çok işe yaradı?
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nEn önemli 15 özellik:")
print(importances.head(15))