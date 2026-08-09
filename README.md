# Müşteri Ayrılma (Churn) Tahmini — Temel Makine Öğrenmesi Akışı

**Türkiye Yapay Zeka Akademisi — Makine Öğrenmesi Ara Ödev**

## Projenin Amacı

Bu proje, derste işlenen temel makine öğrenmesi akışını küçük ve anlaşılır bir
sınıflandırma problemi üzerinde uygulamak için hazırlanmıştır. Senaryo, bir
müşterinin abonelikten ayrılıp ayrılmayacağını (**churn**) tahmin etmektir.

Uygulanan adımlar:

1. Veri okuma / veri oluşturma
2. Temel veri inceleme (satır-sütun sayısı, hedef değişken dağılımı)
3. Eksik değer kontrolü ve doldurma
4. Kategorik değişkenlerin One-Hot Encoding ile sayısallaştırılması
5. Sayısal değişkenlerin StandardScaler ile ölçeklenmesi
6. Basit öznitelik (feature) üretimi (`gelir_grubu`, `destek_talebi_var_mi`,
   `abonelik_yili`)
7. Train / Validation / Test bölme (stratify ile)
8. En az iki modelin eğitimi: **Logistic Regression** ve **KNN**
   (bonus olarak **Decision Tree** de eklendi)
9. Validation sonuçlarına göre model karşılaştırması
10. Seçilen modelin test verisi üzerinde değerlendirilmesi
    (confusion matrix, accuracy, precision, recall, F1-score)

## Proje Yapısı

```
.
├── churn_pipeline.py     # Tüm ML akışının bulunduğu ana Python dosyası
├── musteri_verisi.csv    # Kullanılan veri seti (script ilk çalıştırıldığında otomatik oluşturulur)
├── requirements.txt      # Gerekli kütüphaneler
└── README.md
```

## Veri Seti

Ders kapsamında paylaşılan hazır bir müşteri veri seti bulunmadığı için,
script çalıştırıldığında **300 satırlık sentetik bir müşteri veri seti**
Python içinde otomatik olarak üretilir ve `musteri_verisi.csv` olarak
kaydedilir. Eğer bu dosya klasörde zaten mevcutsa, script yeniden üretim
yapmaz, doğrudan mevcut dosyayı okur.

Sütunlar:

| Sütun                   | Açıklama                                              |
|--------------------------|--------------------------------------------------------|
| `yas`                    | Müşteri yaşı                                            |
| `gelir`                  | Aylık gelir (TL)                                        |
| `abonelik_suresi_ay`     | Müşterinin abone olduğu süre (ay)                       |
| `destek_talebi_sayisi`   | Açılan destek talebi sayısı                              |
| `sehir`                  | Müşterinin yaşadığı şehir (kategorik)                    |
| `uyelik_tipi`            | Abonelik paketi: Standart / Premium / Gold (kategorik)   |
| `churn`                  | Hedef değişken — 0 = kalır, 1 = ayrılır                  |

Veri setine, gerçekçi olması açısından `gelir` ve `sehir` sütunlarında
kasıtlı olarak birkaç eksik değer eklenmiştir; script bunları otomatik
olarak tespit edip doldurur.

## Nasıl Çalıştırılır

1. Gerekli kütüphaneleri kurun:

   ```bash
   pip install -r requirements.txt
   ```

2. Script'i çalıştırın:

   ```bash
   python churn_pipeline.py
   ```

3. Konsolda sırasıyla şu çıktıları göreceksiniz:
   - Veri setinin ilk satırları, boyutu ve hedef değişken dağılımı
   - Eksik değer kontrolü ve doldurma adımları
   - Üretilen yeni öznitelikler
   - One-Hot Encoding ve ölçekleme sonrası veri
   - Train / Validation / Test seti boyutları
   - Logistic Regression, KNN ve Decision Tree modellerinin validation
     performansları
   - Seçilen modelin test seti üzerindeki confusion matrix ve metrikleri
   - Kısa bir sonuç yorumu

## Sonuç ve Kısa Yorum

Sentetik veri seti üzerinde yapılan bir örnek koşuda validation sonuçları
şu şekilde gözlemlenmiştir:

| Model                  | Val. Accuracy | Val. F1-score |
|--------------------------|:---:|:---:|
| Logistic Regression       | 0.70 | 0.69 |
| **KNN**                   | **0.72** | **0.72** |
| Decision Tree (bonus)     | 0.62 | 0.55 |

Validation setinde **F1-score**'a göre en iyi performansı **KNN** modeli
göstermiştir (dengesiz sınıf dağılımlarına karşı F1-score, accuracy'den
daha güvenilir bir metrik olduğu için model seçiminde temel kriter olarak
F1-score kullanılmıştır). Bu model test seti üzerinde de tutarlı bir
performans sergilemiş, **Accuracy ≈ 0.78, F1-score ≈ 0.79** değerlerine
ulaşmıştır.

KNN modelinin daha başarılı olmasının olası nedeni, veri setindeki
`abonelik_suresi_ay` ve `destek_talebi_sayisi` gibi değişkenler ile churn
arasındaki ilişkinin doğrusal olmaktan çok, komşuluk/benzerlik temelli bir
yapıya (benzer profildeki müşterilerin benzer davranış göstermesi) daha
yakın olmasıdır. Logistic Regression bu ilişkiyi doğrusal bir sınır ile
ayırmaya çalıştığı için biraz daha düşük kalmış, Decision Tree ise küçük
veri setinde overfit'e yatkın olduğundan validation setinde en zayıf
performansı vermiştir.

> Not: Veri seti her script çalıştığında (dosya silinip yeniden
> üretildiğinde) rastgele oluşturulduğu için sonuçlar küçük farklılıklar
> gösterebilir; genel eğilim ve yorum yaklaşımı yukarıdaki gibi kalır.

## Kullanılan Kütüphaneler

- **pandas** — veri okuma ve DataFrame işlemleri
- **numpy** — sayısal işlemler ve sentetik veri üretimi
- **scikit-learn** — ön işleme (encoding, scaling), model eğitimi ve
  değerlendirme metrikleri
