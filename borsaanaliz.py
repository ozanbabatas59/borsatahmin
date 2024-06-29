import yfinance as yf
import datetime
from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

def veri_indir(hisse, baslangic_tarihi, bitis_tarihi):
    df = yf.download(hisse, start=baslangic_tarihi, end=bitis_tarihi, progress=False)
    return df

def model_calistir(model, gun_sayisi):
    veri = veri_seti[['Close']]
    veri['tahmin'] = veri.Close.shift(-gun_sayisi)
    
    x = veri.drop(['tahmin'], axis=1).values
    x = olcekleyici.fit_transform(x)
    x = x[:-gun_sayisi]
    x_tahmin = x[-gun_sayisi:]
    y = veri.tahmin.values
    y = y[:-gun_sayisi]
    x_egitim, x_test, y_egitim, y_test = train_test_split(x, y, test_size=0.2, random_state=7)
    model.fit(x_egitim, y_egitim)
    tahminler = model.predict(x_test)
    print(f'Hata Payı: {r2_score(y_test, tahminler)}')
    
    gelecekteki_tahmin = model.predict(x_tahmin)
    gun = 1
    for i in gelecekteki_tahmin:
        print(f'{gun}. Günün Tahmini: {i}')
        gun += 1

# Parametreler
hisse = "HISSEADI.IS"
bugun = datetime.date.today()
sure = 3000
once = bugun - datetime.timedelta(days=sure)
baslangic_tarihi = once
bitis_tarihi = bugun

# Veriyi indir
veri_seti = veri_indir(hisse, baslangic_tarihi, bitis_tarihi)
print(veri_seti)

# Model parametreleri
gun_sayisi = 10
olcekleyici = StandardScaler()
motor = LinearRegression()

# Modeli çalıştır
model_calistir(motor, gun_sayisi)
