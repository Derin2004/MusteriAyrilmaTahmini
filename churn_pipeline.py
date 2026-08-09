"""
Musteri Ayrilma (Churn) Tahmini - Temel Makine Ogrenmesi Akisi
================================================================

Amac
----
Bu script, bir musteri veri seti uzerinde uctan uca temel bir makine
ogrenmesi akisini uygular:
    1) Veri olusturma / okuma
    2) Kesif amacli veri inceleme (EDA)
    3) Eksik deger kontrolu ve temizleme
    4) Kategorik degiskenlerin One-Hot Encoding ile sayisallastirilmasi
    5) Sayisal degiskenlerin olceklenmesi (StandardScaler)
    6) Basit oznitelik (feature) muhendisligi
    7) Train / Validation / Test bolme (stratify ile)
    8) En az iki modelin (Logistic Regression, KNN) egitimi
       + bonus olarak Decision Tree
    9) Validation performansina gore model karsilastirmasi
    10) Secilen modelin test seti uzerinde degerlendirilmesi
        (confusion matrix, accuracy, precision, recall, f1-score)
    11) Kisa bir sonuc yorumu

Kullanilan Kutuphaneler
------------------------
- pandas      : veri okuma / DataFrame islemleri
- numpy       : rastgele veri uretimi, sayisal islemler
- scikit-learn: on isleme, model egitimi, degerlendirme metrikleri

Calistirma Adimlari
--------------------
1) Gerekli kutuphaneleri kurun:
       pip install -r requirements.txt
2) Scripti calistirin:
       python churn_pipeline.py
3) Script calistiginda:
   - Eger 'musteri_verisi.csv' dosyasi klasorde varsa onu okur.
   - Yoksa, 300 satirlik ornek bir musteri veri seti uretir ve
     'musteri_verisi.csv' olarak kaydeder.
   - Tum adimlarin ciktilarini konsola yazdirir.
"""

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Sonuclarin tekrar uretilebilir (reproducible) olmasi icin sabit seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_PATH = "musteri_verisi.csv"


def veri_yukle_veya_uret(path: str = DATA_PATH, n: int = 300) -> pd.DataFrame:
    """
    Eger belirtilen path'te bir CSV dosyasi varsa onu okur.
    Yoksa, n satirlik sentetik bir musteri churn veri seti uretir
    ve ayni path'e CSV olarak kaydeder.

    Sutunlar:
        yas                  : musteri yasi (int)
        gelir                : aylik gelir, TL (float)
        abonelik_suresi_ay   : musterinin abone oldugu sure, ay (int)
        destek_talebi_sayisi : musterinin actigi destek talebi sayisi (int)
        sehir                : musterinin yasadigi sehir (kategorik)
        uyelik_tipi          : abonelik paketi (kategorik)
        churn                : hedef degisken, 0 = kalir, 1 = ayrilir
    """
    if os.path.exists(path):
        print(f"'{path}' bulundu, veri buradan okunuyor.\n")
        return pd.read_csv(path)

    print(f"'{path}' bulunamadi, {n} satirlik sentetik veri seti uretiliyor.\n")

    sehirler = ["Istanbul", "Ankara", "Izmir", "Bursa", "Antalya"]
    uyelik_tipleri = ["Standart", "Premium", "Gold"]

    yas = np.random.randint(18, 70, size=n)
    gelir = np.round(np.random.normal(loc=15000, scale=5000, size=n).clip(3000, 40000), 2)
    abonelik_suresi_ay = np.random.randint(1, 60, size=n)
    destek_talebi_sayisi = np.random.poisson(lam=1.5, size=n)
    sehir = np.random.choice(sehirler, size=n, p=[0.30, 0.20, 0.20, 0.15, 0.15])
    uyelik_tipi = np.random.choice(uyelik_tipleri, size=n, p=[0.5, 0.3, 0.2])

    # Churn olasiligini bazi degiskenlere bagli, gercekci bir sekilde kuruyoruz:
    # - abonelik suresi kisa olanlar daha cok ayrilir
    # - destek talebi cok olanlar daha cok ayrilir
    # - geliri dusuk olanlar biraz daha cok ayrilir
    churn_skoru = (
        -0.05 * abonelik_suresi_ay
        + 0.9 * destek_talebi_sayisi
        - 0.0002 * gelir
        + np.where(uyelik_tipi == "Standart", 0.8, 0.0)
        + np.random.normal(0, 1.5, size=n)
    )
    churn_prob = 1 / (1 + np.exp(-(churn_skoru - churn_skoru.mean()) / churn_skoru.std()))
    churn = (churn_prob > np.random.uniform(0.4, 0.6, size=n)).astype(int)

    df = pd.DataFrame({
        "yas": yas,
        "gelir": gelir,
        "abonelik_suresi_ay": abonelik_suresi_ay,
        "destek_talebi_sayisi": destek_talebi_sayisi,
        "sehir": sehir,
        "uyelik_tipi": uyelik_tipi,
        "churn": churn,
    })

    # Gercekci olmasi icin veri setine kasitli olarak birkac eksik deger ekliyoruz
    eksik_index_gelir = np.random.choice(df.index, size=max(1, n // 40), replace=False)
    df.loc[eksik_index_gelir, "gelir"] = np.nan

    eksik_index_sehir = np.random.choice(df.index, size=max(1, n // 50), replace=False)
    df.loc[eksik_index_sehir, "sehir"] = np.nan

    df.to_csv(path, index=False)
    return df


def veriyi_incele(df: pd.DataFrame) -> None:
    """Veri setinin genel yapisini konsola yazdirir."""
    print("=" * 60)
    print("1) VERI SETI GENEL BAKIS")
    print("=" * 60)
    print("\nIlk 5 satir:")
    print(df.head())

    print(f"\nSatir sayisi: {df.shape[0]}, Sutun sayisi: {df.shape[1]}")

    print("\nSutun veri tipleri:")
    print(df.dtypes)

    print("\nHedef degisken (churn) dagilimi:")
    print(df["churn"].value_counts())
    print("\nHedef degisken (churn) oran dagilimi (%):")
    print((df["churn"].value_counts(normalize=True) * 100).round(2))


def eksik_deger_kontrolu(df: pd.DataFrame) -> pd.DataFrame:
    """
    Eksik degerleri kontrol eder ve doldurur:
    - Sayisal sutunlarda medyan ile doldurma
    - Kategorik sutunlarda mod (en sik gorulen deger) ile doldurma
    """
    print("\n" + "=" * 60)
    print("2) EKSIK DEGER KONTROLU")
    print("=" * 60)

    print("\nSutun basina eksik deger sayisi:")
    print(df.isnull().sum())

    df = df.copy()
    sayisal_sutunlar = df.select_dtypes(include=[np.number]).columns.tolist()
    if "churn" in sayisal_sutunlar:
        sayisal_sutunlar.remove("churn")
    kategorik_sutunlar = [c for c in df.columns if c not in sayisal_sutunlar and c != "churn"]

    for col in sayisal_sutunlar:
        if df[col].isnull().sum() > 0:
            medyan = df[col].median()
            df[col] = df[col].fillna(medyan)
            print(f"-> '{col}' sutunundaki eksik degerler medyan ({medyan:.2f}) ile dolduruldu.")

    for col in kategorik_sutunlar:
        if df[col].isnull().sum() > 0:
            mod_deger = df[col].mode()[0]
            df[col] = df[col].fillna(mod_deger)
            print(f"-> '{col}' sutunundaki eksik degerler mod ('{mod_deger}') ile dolduruldu.")

    print("\nDoldurma sonrasi toplam eksik deger sayisi:", df.isnull().sum().sum())
    return df


def oznitelik_uret(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basit, anlamli oznitelikler (feature) uretir:
    - gelir_grubu           : gelire gore dusuk/orta/yuksek kategorisi
    - destek_talebi_var_mi  : hic destek talebi acmis mi (0/1)
    - abonelik_yili         : abonelik suresini yil cinsinden ifade eder
    """
    print("\n" + "=" * 60)
    print("3) OZNITELIK (FEATURE) URETIMI")
    print("=" * 60)

    df = df.copy()

    df["gelir_grubu"] = pd.cut(
        df["gelir"],
        bins=[0, 10000, 20000, np.inf],
        labels=["dusuk", "orta", "yuksek"],
    )

    df["destek_talebi_var_mi"] = (df["destek_talebi_sayisi"] > 0).astype(int)

    df["abonelik_yili"] = (df["abonelik_suresi_ay"] / 12).round(2)

    print("Eklenen yeni oznitelikler: gelir_grubu, destek_talebi_var_mi, abonelik_yili")
    print("\nYeni oznitelikleri iceren ilk 5 satir:")
    print(df[["gelir", "gelir_grubu", "destek_talebi_sayisi",
              "destek_talebi_var_mi", "abonelik_suresi_ay", "abonelik_yili"]].head())

    return df


def on_isleme(df: pd.DataFrame):
    """
    - Kategorik degiskenleri One-Hot Encoding ile sayisallastirir
    - Sayisal degiskenleri StandardScaler ile olcekler
    Geriye: X (oznitelikler), y (hedef) dondurur.
    """
    print("\n" + "=" * 60)
    print("4) KATEGORIK DONUSUM VE OLCEKLEME")
    print("=" * 60)

    df = df.copy()
    y = df["churn"]
    X = df.drop(columns=["churn"])

    sayisal_sutunlar = X.select_dtypes(include=[np.number]).columns.tolist()
    kategorik_sutunlar = [c for c in X.columns if c not in sayisal_sutunlar]

    print(f"\nKategorik sutunlar (One-Hot Encoding uygulanacak): {kategorik_sutunlar}")
    print(f"Sayisal sutunlar (StandardScaler uygulanacak): {sayisal_sutunlar}")

    # One-Hot Encoding
    X = pd.get_dummies(X, columns=kategorik_sutunlar, drop_first=True)

    # Olcekleme (sadece asil sayisal sutunlara uygulaniyor)
    scaler = StandardScaler()
    X[sayisal_sutunlar] = scaler.fit_transform(X[sayisal_sutunlar])

    print(f"\nOn isleme sonrasi oznitelik sayisi: {X.shape[1]}")
    print("On isleme sonrasi ilk 5 satir:")
    print(X.head())

    return X, y


def veriyi_bol(X: pd.DataFrame, y: pd.Series):
    """
    Veriyi once train+val / test, sonra train / val olacak sekilde iki asamada boler.
    Sonuc oranlari yaklasik: %60 train, %20 validation, %20 test
    Sinif dengesizligini korumak icin stratify kullanilir.
    """
    print("\n" + "=" * 60)
    print("5) TRAIN / VALIDATION / TEST BOLME")
    print("=" * 60)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=RANDOM_STATE, stratify=y_train_val
    )
    # 0.25 * 0.80 = 0.20 -> toplamda test %20, val %20, train %60

    print(f"\nTrain seti boyutu     : {X_train.shape[0]} satir")
    print(f"Validation seti boyutu: {X_val.shape[0]} satir")
    print(f"Test seti boyutu      : {X_test.shape[0]} satir")

    return X_train, X_val, X_test, y_train, y_val, y_test


def metrikleri_hesapla(y_true, y_pred) -> dict:
    """Verilen gercek ve tahmin degerleri icin temel siniflandirma metriklerini hesaplar."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def modelleri_egit_ve_karsilastir(X_train, X_val, y_train, y_val):
    """
    En az 2 model egitir (Logistic Regression, KNN) ve bonus olarak
    Decision Tree ekler. Validation seti uzerindeki performanslarini
    karsilastirir ve en iyi modeli (f1-score'a gore) secer.
    """
    print("\n" + "=" * 60)
    print("6) MODEL EGITIMI VE VALIDATION KARSILASTIRMASI")
    print("=" * 60)

    modeller = {
        "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree (bonus)": DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=5),
    }

    sonuclar = {}
    for isim, model in modeller.items():
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        metrikler = metrikleri_hesapla(y_val, y_val_pred)
        sonuclar[isim] = {"model": model, "metrikler": metrikler}

        print(f"\n--- {isim} (Validation Sonuclari) ---")
        for m_isim, m_deger in metrikler.items():
            print(f"{m_isim:<10}: {m_deger:.4f}")

    # F1-score'a gore en iyi modeli sec (dengesiz siniflarda accuracy'den daha guvenilir)
    en_iyi_isim = max(sonuclar, key=lambda k: sonuclar[k]["metrikler"]["f1"])
    print(f"\n>>> Validation F1-score'a gore secilen en iyi model: {en_iyi_isim}")

    return sonuclar, en_iyi_isim


def test_uzerinde_degerlendir(model, model_isim: str, X_test, y_test):
    """Secilen modeli test seti uzerinde degerlendirir ve metrikleri yazdirir."""
    print("\n" + "=" * 60)
    print(f"7) TEST SETI DEGERLENDIRMESI -> Secilen Model: {model_isim}")
    print("=" * 60)

    y_test_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_test_pred)
    print("\nConfusion Matrix:")
    print("                 Tahmin: Kalir(0)   Tahmin: Ayrilir(1)")
    print(f"Gercek: Kalir(0)      {cm[0][0]:<10}        {cm[0][1]:<10}")
    print(f"Gercek: Ayrilir(1)    {cm[1][0]:<10}        {cm[1][1]:<10}")

    metrikler = metrikleri_hesapla(y_test, y_test_pred)
    print("\nTest Metrikleri:")
    for m_isim, m_deger in metrikler.items():
        print(f"{m_isim:<10}: {m_deger:.4f}")

    print("\nDetayli siniflandirma raporu:")
    print(classification_report(y_test, y_test_pred, target_names=["Kalir (0)", "Ayrilir (1)"], zero_division=0))

    return metrikler


def sonuc_yorumu(sonuclar: dict, en_iyi_isim: str, test_metrikleri: dict) -> None:
    """Validation ve test sonuclarina dayanarak kisa bir yorum uretir."""
    print("\n" + "=" * 60)
    print("8) SONUC YORUMU")
    print("=" * 60)

    val_f1 = sonuclar[en_iyi_isim]["metrikler"]["f1"]
    test_f1 = test_metrikleri["f1"]

    diger_modeller = [isim for isim in sonuclar if isim != en_iyi_isim]
    karsilastirma_satirlari = [
        f"  - {isim}: F1={sonuclar[isim]['metrikler']['f1']:.4f}, "
        f"Accuracy={sonuclar[isim]['metrikler']['accuracy']:.4f}"
        for isim in diger_modeller
    ]

    print(f"""
Validation asamasinda en iyi F1-score'u '{en_iyi_isim}' modeli elde etti
(F1 = {val_f1:.4f}). Bu modelin test setindeki F1-score'u {test_f1:.4f},
Accuracy degeri {test_metrikleri['accuracy']:.4f} olarak olculdu.

Karsilastirilan diger modellerin validation performanslari:
{chr(10).join(karsilastirma_satirlari)}

Genel degerlendirme:
- '{en_iyi_isim}' modelinin daha iyi sonuc vermesinin olasi nedenleri:
  veri setindeki iliskilerin (ozellikle abonelik suresi ve destek talebi
  sayisi ile churn arasindaki iliskinin) bu modelin varsayimlarina
  (ornegin dogrusal ayrilabilirlik ya da komsuluk tabanli benzerlik)
  diger modellere gore daha uygun olmasi olabilir.
- Veri seti kucuk ve sentetik oldugu icin sonuclar veri uretimindeki
  rastgeleligine bagli olarak degisebilir; gercek dunya verisinde
  farkli bir model daha iyi performans gosterebilir.
- Sonraki adim olarak, cross-validation, hiperparametre optimizasyonu
  veya daha fazla/gercek veri ile modelin tekrar degerlendirilmesi
  onerilir.
""")


def main():
    df = veri_yukle_veya_uret()
    veriyi_incele(df)
    df = eksik_deger_kontrolu(df)
    df = oznitelik_uret(df)
    X, y = on_isleme(df)
    X_train, X_val, X_test, y_train, y_val, y_test = veriyi_bol(X, y)
    sonuclar, en_iyi_isim = modelleri_egit_ve_karsilastir(X_train, X_val, y_train, y_val)

    en_iyi_model = sonuclar[en_iyi_isim]["model"]
    test_metrikleri = test_uzerinde_degerlendir(en_iyi_model, en_iyi_isim, X_test, y_test)

    sonuc_yorumu(sonuclar, en_iyi_isim, test_metrikleri)


if __name__ == "__main__":
    main()
