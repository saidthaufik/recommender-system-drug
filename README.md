# Proyek Machine Learning - Model Recommender Content-Based System Berbasis Obat

# Data Diri
- Nama : Said Thaufik Rizaldi
- Jenis Kelamin : Laki-Laki
- Email : saidthaufik24@gmail.com
- Kota Domisili : Bogor
- Jalur : Educators 2024

# Project Overview
Keterbatasan obat dan manajemen stok sering menjadi tantangan bagi apotek dan fasilitas kesehatan [[1](10.22487/j24428744.2024.v10.i1.16479)]. Ketika obat utama yang diresepkan tidak tersedia, pasien harus menunggu atau mencari obat di tempat lain, yang dapat menyebabkan penundaan pengobatan dan ketidakpuasan [[2](10.20473/jaki.v7i1.2019.1-8)]. Untuk mengatasi masalah ini, sistem rekomendasi obat dapat menjadi solusi yang efektif. Sistem ini bertujuan untuk memberikan rekomendasi alternatif obat yang relevan berdasarkan kesamaan karakteristik obat dengan kondisi medis pasien [[3](10.3390/ijerph20010309)].

Sistem rekomendasi obat berbasis _Content-Based Filtering_ merupakan salah satu pendekatan yang memanfaatkan data deskriptif dari obat untuk memberikan rekomendasi yang dipersonalisasi [[4](10.1016/j.compbiomed.2024.108117)]. Dengan menganalisis atribut seperti kondisi medis yang diobati, sistem ini dapat secara otomatis mengidentifikasi obat lain yang memiliki kesamaan dengan obat yang Diinputkan [[5](10.22146/ijitee.45538)]. Keunggulan ini membuat _Content-Based Filtering_ sangat berguna dalam meningkatkan efisiensi proses pengambilan keputusan oleh apotek dan profesional kesehatan [[6](10.11591/ijece.v13i1.pp884-893)], serta dalam meningkatkan pengalaman pasien [[7](10.1111/exsy.12519)].

Pendekatan permodelan yang pernah dilakukan dalam konteks serupa mencakup penggabungan _Term Frequency-Inverse Document Frequency_ (TF-IDF) dengan algoritma _cosine similarity_ untuk mengevaluasi tingkat kesamaan antar entitas menunjukkan bahwa metode ini dapat digunakan untuk merekomendasikan pengobatan alternatif yang relevan berdasarkan analisis data historis pasien [[5](10.22146/ijitee.45538)] dan data review [[8](10.1109/ACCESS.2021.3088838)].

# Business Understanding
Manajemen stok obat yang tidak optimal sering kali menyebabkan ketidaktersediaan obat utama yang dibutuhkan oleh pasien, sehingga mengganggu proses pengobatan dan menurunkan kepuasan pasien. Untuk mengatasi masalah ini, sistem rekomendasi berbasis _Content-Based Filtering_ dapat digunakan untuk memberikan alternatif obat yang relevan berdasarkan kesamaan karakteristik obat dengan kondisi medis pasien. Dengan menganalisis atribut seperti kondisi medis yang diobati, sistem ini membantu apotek dan profesional kesehatan memastikan pasien mendapatkan pengobatan yang optimal, bahkan ketika stok obat utama tidak tersedia. Hal ini tidak hanya meningkatkan efisiensi pelayanan tetapi juga memberikan pengalaman yang lebih baik bagi pasien.

## Problem Statements
1. Bagaimana cara mengetahui dan memahami karakteristik dataset dalam pembuatan model _Recommender System_ berbasis _Content-Based Filtering_ untuk obat?
2. Bagaimana cara merancang dan membangun model _Recommender System_ dengan pendekatan _Content-Based Filtering_?
3. Bagaimana metode yang tepat untuk mengukur performa (evaluasi) model Recommender System berbasis _Content-Based Filtering_ yang telah dibangun?

## Goals
1. Melakukan proses _Exploratory Data Analysis_ (EDA) dan Visualisasi Data untuk memahami karakteristik dataset yang digunakan.
2. Merancang dan membangun model _Recommender System_ dengan pendekatan _Content-Based Filtering_ untuk obat.
3. Melakukan evaluasi performa model _Recommender System_ berbasis _Content-Based Filtering_ secara sistematis.

## Solution Approach
1. Melakukan EDA untuk menganalisa karakteristik dataset hingga fiturnya dengan menggunakan fungsi `info()`,`describe()`,`nunique()`, dan `duplicate()`. Kemudian melakukan visualisasi data dengan menggunakan Histogram untuk mendapatkan pemahaman yang lebih komprehensif
2. Membangun _Recommender System_ dengan `Content-Based Filtering` yang memberikan rekomendasi kepada pengguna berdasarkan kesamaan pada item yang ada. Data yang digunakan adalah data Obat dengan kondisi medisnya. Dataset tersebut juga melewati tahap _Data Preparation_ agar dataset dapat digunakan untuk proses pembangunan model seperti Handling Data Duplikat, _Handling Missing Value_/ Nilai `NaN`, Filter Frekuensi Data, dan Filter Fitur Penting. Kemudian, data yang sudah siap, diproses ke tahap _modelling_ yang memanfaatkan `Tfidvectorizer`, `cosine similarity`, dan fungsi buatan yang mengembalikan rekomendasi berdasarkan kesamaan pada item. Pendekatan ini berfokus pada karakteristik atau konten dari item yang direkomendasikan
3. Melakukan perhitungan skor **Precision** untuk mengukur performa dari model _Recommender System_ Obat dengan _Content-Based Fearning_. 

# Data Understanding 
Dataset yang digunakan untuk pembuatan model sistem rekomendasi ini adalah dataset "Drug Review Dataset (Drugs.com)" yang tersedia di situs [UCI Machine Learning Repository](https://doi.org/10.24432/C55G6J). Dataset ini berisi ulasan pasien terkait obat-obatan tertentu beserta kondisi medis yang relevan.

Terdapat dua file utama yang digunakan:
1. **`drugLibTrain_raw.tsv`** atau variabel `df_a` – Berisi 3107 baris dan 9 kolom
2. **`drugLibTest_raw.tsv`** atau variabel `df_b` – Berisi 1036 baris dan 9 kolom.

Dataset ini digunakan untuk membangun sistem rekomendasi berbasis **Content-Based Filtering**

Berikut adalah informasi mengenai atribut-atribut yang terdapat pada dataset:

**Atribut pada Dataset:**
- **`Unnamed:0`** / **ReviewID** : Data Unik Review dari pengguna
- **`urlDrugName`**: Nama obat yang diulas.
- **`rating`**: Skor penilaian yang diberikan pengguna (skala 1–10).
- **`effectiveness`**: Efektivitas obat berdasarkan pengalaman pengguna.
- **`sideEffects`**: Efek samping yang dirasakan pengguna.
- **`condition`**: Kondisi medis yang diatasi oleh obat.
- **`benefitsReview`**: Ulasan mengenai manfaat obat dari pengguna.
- **`sideEffectsReview`**: Ulasan mengenai efek samping obat dari pengguna.
- **`commentsReview`**: Komentar tambahan mengenai pengalaman pengguna dengan obat.

**Dataset tersebut dapat diunduh di [https://doi.org/10.24432/C55G6J](https://doi.org/10.24432/C55G6J)**

- **Exploratory Data Analysis**
  
  **Exploratory Data Analysis (EDA)** adalah proses eksplorasi dan analisis awal terhadap data untuk memahami struktur, pola, dan karakteristiknya sebelum dilakukan pemodelan lebih lanjut. EDA bertujuan untuk mengidentifikasi distribusi data, mendeteksi nilai yang hilang atau pencilan, serta menganalisis hubungan antar variabel.
  
  - **Dataset `df_a`**

    - Menampilan jumlah baris dan kolom yang ada pada dataset
      ``` python
      df_a.shape
      ```
      Dengan _output_:
      ```
      (3107, 9)
      ```
      Berdasarkan _output_ diatas, `df_a` memiliki:
      - 3107 baris data
      - 9 kolom data
    
    - Menampilkan kolom, tipe data, cek data kosong dari setiap kolom yang ada
      ``` python
      df_a.info()
      ```
      Dengan _output_:
      ```
      <class 'pandas.core.frame.DataFrame'>
      Int64Index: 3107 entries, 0 to 3106
      Data columns (total 9 columns):
       #   Column             Non-Null Count  Dtype 
      ---  ------             --------------  ----- 
       0   Unnamed: 0         3107 non-null   int64 
       1   urlDrugName        3107 non-null   object
       2   rating             3107 non-null   int64 
       3   effectiveness      3107 non-null   object
       4   sideEffects        3107 non-null   object
       5   condition          3106 non-null   object
       6   benefitsReview     3107 non-null   object
       7   sideEffectsReview  3105 non-null   object
       8   commentsReview     3099 non-null   object
      dtypes: int64(2), object(7)
      memory usage: 323.7+ KB
      ```
      Berdasarkan _output_ diatas, `df_a` memiliki 9 kolom pada dataset ini diantaranya memiliki tipe datanya masing-masing, yaitu:
      - `Unnamed:0` : `int64`
      - `urlDrugName` : `object`
      - `rating` : `int64`
      - `effectiveness` : `object`
      - `sideEffects` : `object`
      - `condition` : `object`
      - `benefitsReview` : `object`
      - `sideEffectsReview` : `object`
      - `commentsReview` : `object`
    
    - Menampilkan deskripsi statistik kolom `rating`
      ``` python
      df_a['rating'].describe()
      ```
      Dengan _output_
      ```
      count    3107.000000
      mean        7.006115
      std         2.937582
      min         1.000000
      25%         5.000000
      50%         8.000000
      75%         9.000000
      max        10.000000
      Name: rating, dtype: float64
      ```
      Fungsi di atas menyediakan informasi statistik deskriptif untuk kolom `review` pada variabel `df_a`, meliputi:
    
      - **`count`**: Jumlah total data dalam kolom.
      - **`mean`**: Nilai rata-rata dari data dalam kolom.
      - **`std`**: Standar deviasi dari data dalam kolom.
      - **`min`**: Nilai terkecil dalam kolom.
      - **`25%`**: Kuartil pertama (Q1), yaitu nilai yang memisahkan 25% data terendah.
      - **`50%`**: Kuartil kedua (Q2) atau median, yaitu nilai tengah dari data.
      - **`75%`**: Kuartil ketiga (Q3), yaitu nilai yang memisahkan 25% data tertinggi.
      - **`max`**: Nilai terbesar dalam kolom.
    
    - Menampilkann total unique value di kolom `Unnamed:0` / ReviewID
      ``` python
      print(df_a['Unnamed: 0'].nunique())
      ```
      Dengan _output_
      ```
      3107
      ```
      Berdasarkan output diatas nilai fungsi `nunique()` dari kolom `Unnamed:0` atau `ReviewID` adalah sebanyak **3107 data unik**
    
    - Menampilkann total unique value di kolom `urlDrugName`
      ``` python
      print(df_a['urlDrugName'].nunique())
      ```
      Dengan _output_
      ```
      502
      ```
      Berdasarkan _output_ diatas nilai `nunique()` dari kolom `urlDrugName` adalah sebanyak **541 data unik** terkait dengan obat.
      
    - Menampilkann total unique value di kolom `effectiveness`
      ``` python
      print(df_a['effectiveness'].nunique())
      ```
      Dengan _output_
      ```
      5
      ```
      Berdasarkan _output_ diatas nilai `nunique()` dari kolom `effectiveness` adalah sebanyak **5 data unik**. Ini sudah merepresentasikan `effectiveness` yang digunakan pada _dataset_ ini dengan skala dengan 5 level.
    
    - Menampilkann total unique value di kolom `sideEffects`
      ``` python
      print(df_a['sideEffects'].nunique())
      ```
      Dengan _output_
      ```
      5
      ```
      Berdasarkan output diatas nilai `nunique()` dari kolom `sideEffects` adalah sebanyak **5 data unik**. Ini sudah merepresentasikan `sideEffects` yang digunakan pada dataset ini dengan skala dengan 5 level.
      
    - Menampilkann total unique value di kolom `condition`
      ``` python
      print(df_a['condition'].nunique())
      ```
      Dengan _output_
      ```
      1426
      ```
      Berdasarkan _output_ diatas nilai `nunique()` dari kolom `condition` adalah sebanyak **1426 data unik**.
    
    - Menampilkann total unique value di kolom `benefitsReview`
      ``` python
      print(df_a['benefitsReview'].nunique())
      ```
      Dengan _output_
      ```
      3032
      ```
      Berdasarkan _output_ diatas nilai `nunique()` dari kolom `benefitsReview` adalah sebanyak **3032 data unik** atau hampir keseluruhan data merupakan data unik.
    
      ```python
      df_a['benefitsReview']
      ```
      Dengan _output_
      ```
      0       slowed the progression of left ventricular dys...
      1       Although this type of birth control has more c...
      2       I was used to having cramps so badly that they...
      3       The acid reflux went away for a few months aft...
      4       I think that the Lyrica was starting to help w...
                                    ...                        
      3102    Increased focus, attention, productivity. Bett...
      3103      Emotions were somewhat blunted. Less moodiness.
      3104                                                  ---
      3105                 Controlled complex partial seizures.
      3106    The drug Micardis did seem to alleviate my hig...
      Name: benefitsReview, Length: 3107, dtype: object
      ```
      Berdasarkan _output_ diatas bahwa karakteristik data pada kolom `benefitsReview` adalah berbentuk _Review_ atau _Natural Language_
    
    - Menampilkann total unique value di kolom `sideEffectsReview`
      ```python
      print(df_a['sideEffectsReview'].nunique())
      ```
      Dengan _output_
      ```
      2814
      ```
      Berdasarkan output diatas nilai `nunique()` dari kolom `sideEffectsReview` adalah sebanyak **2814 data unik** atau hampir keseluruhan data merupakan data unik.
    
      ```python
      df_a['sideEffectsReview']
      ```
      Dengan _output_
      ```
      0       cough, hypotension , proteinuria, impotence , ...
      1       Heavy Cycle, Cramps, Hot Flashes, Fatigue, Lon...
      2              Heavier bleeding and clotting than normal.
      3       Constipation, dry mouth and some mild dizzines...
      4       I felt extremely drugged and dopey.  Could not...
                                    ...                        
      3102    Restless legs at night, insomnia, headache (so...
      3103    Weight gain, extreme tiredness during the day,...
      3104    Constant issues with the patch not staying on....
      3105                           Dizziness, fatigue, nausea
      3106    I find when I am taking Micardis that I tend t...
      Name: sideEffectsReview, Length: 3107, dtype: object
      ```
      Berdasarkan _output_ diatas bahwa karakteristik data pada kolom `sideEffectsReview` adalah berbentuk _Review_ atau _Natural Language_
    
    - Menampilkann total unique value di kolom `commentsReview`
      ``` python
      print(df_a['commentsReview'].nunique())
      ```
      Dengan _output_
      ```
      3047
      ```
      Berdasarkan output diatas nilai `nunique()` dari kolom `commentsReview` adalah sebanyak **3047 data unik** atau hampir keseluruhan data merupakan data unik.
    
      ```python
      df_a['commentsReview']
      ```
      Dengan _output_
      ```
      0       monitor blood pressure , weight and asses for ...
      1       I Hate This Birth Control, I Would Not Suggest...
      2       I took 2 pills at the onset of my menstrual cr...
      3       I was given Prilosec prescription at a dose of...
      4                                               See above
                                    ...                        
      3102    I took adderall once as a child, and it made m...
      3103    I was on Zoloft for about 2 years total. I am ...
      3104                                                  ---
      3105    Started at 2 doses of 300 mg a day and worked ...
      3106             I take Micardis in pill form once daily.
      Name: commentsReview, Length: 3107, dtype: object
      ```
    
      Berdasarkan _output_ diatas bahwa karakteristik data pada kolom `commentsReview` adalah berbentuk _Review_ atau _Natural Language_ 
    
    - Pengecekan data duplikat pada dataframe `df_a`
      ```python
      df_a.duplicated().sum()
      ```
      Dengan _output_
      ```
      0
      ```
      Berdasarkan _output_ diatas bahwa tidak terdapat data yang duplikat terhadap dataset `df_a`

    - **Data Visualization df_a**
    
      **Data Visualization** adalah proses penyajian data dalam bentuk grafik atau diagram untuk mempermudah interpretasi dan analisis informasi. Visualisasi data memungkinkan kita untuk mengidentifikasi pola, tren, dan hubungan antar variabel dengan lebih intuitif dibandingkan hanya melihat tabel angka mentah. 

      - Melihat Distribusi 30 `urlDrugName` Teratas
        <div align="center">
        <img src="https://github.com/user-attachments/assets/1c1e3fed-0ecf-42da-8a2d-1b689604b680" alt="Distribusi 30 Obat Teratas df_a" width="500"/>
        <br>
        <b>Gambar 1. Distribusi 30 Obat Teratas Dataset df_a</b>
        </div>
      
        Berdasarkan grafik diatas bahwa persebaran data pada `urlDrugName` cukup merata sehingga tetap dipertahankan untuk analisis berikutnya
      
      - Melihat Distribusi `rating`
        <div align="center">
        <img src="https://github.com/user-attachments/assets/b7c0fbe4-c38e-49df-9c82-35e122f2e6d3" alt="Distribusi Rating df_a" width="500"/>
        <br>
        <b>Gambar 2. Distribusi Rating Dataset df_a</b>
        </div>
      
        Berdasarkan grafik diatas bahwa distribusi `rating` yang **terbesar adalah 10 dengan data 742**, sedangkan `rating` yang **terkecil adalah 2 dengan data 103**
      
      - Melihat Distribusi `effectiveness`
        <div align="center">
        <img src="https://github.com/user-attachments/assets/c3e9e644-d9ea-4cd4-8754-7472e4f1d9be" alt="Distribusi Efektifitas Obat" width="500"/>
        <br>
        <b>Gambar 3. Distribusi Efektifitas Obat Dataset df_a</b>
        </div>
      
        Berdasarkan grafik diatas bahwa distribusi `effectiveness` yang **terbesar adalah Highly Effective dengan data 1330**, sedangkan `effectiveness` yang **terkecil adalah Marginally Effective dengan 187 data**
      
      - Melihat Distribusi `sideEffects`
        <div align="center">
        <img src="https://github.com/user-attachments/assets/11143a43-b4ea-4178-ae9a-d13c2e12e0d5" alt="Distribusi Efek Samping Obat" width="500"/>
        <br>
        <b>Gambar 4. Distribusi Efek Samping Obat Dataset df_a</b>
        </div>
      
        Berdasarkan grafik diatas bahwa distribusi `sideEffects` yang **terbesar adalah Mid Side Effects dengan data 1019**, sedangkan `sideEffects` yang **terkecil adalah Extremely Severe Side Effects dengan 175 data**
      
      - Melihat Distribusi `condition`
        <div align="center">
        <img src="https://github.com/user-attachments/assets/68e34c37-e706-4dfb-b7ef-04f9aaa5ee41" alt="Distribusi Kondisi Medis" width="500"/>
        <br>
        <b>Gambar 5. Distribusi Kondisi Medis Dataset df_a</b>
        </div>
      
        Berikut adalah grafik 10 teratas distribusi `condition` terlihat bahwa **despression** dan **acne** menjadi yang paling mencolok dari distribusi lainnya

  - **Dataset `df_b`**

    - Menampilan jumlah baris dan kolom yang ada pada dataset
      ``` python
      df_b.shape
      ```
      Dengan _output_:
      ```
      (1036, 9)
      ```
      Berdasarkan _output_ diatas, `df_b` memiliki:
      - 1036 baris data
      - 9 kolom data
    
    - Menampilkan kolom, tipe data, cek data kosong dari setiap kolom yang ada
      ``` python
      df_b.info()
      ```
      Dengan _output_:
      ```
      <class 'pandas.core.frame.DataFrame'>
      RangeIndex: 1036 entries, 0 to 1035
      Data columns (total 9 columns):
       #   Column             Non-Null Count  Dtype 
      ---  ------             --------------  ----- 
       0   Unnamed: 0         1036 non-null   int64 
       1   urlDrugName        1036 non-null   object
       2   rating             1036 non-null   int64 
       3   effectiveness      1036 non-null   object
       4   sideEffects        1036 non-null   object
       5   condition          1036 non-null   object
       6   benefitsReview     1036 non-null   object
       7   sideEffectsReview  1036 non-null   object
       8   commentsReview     1036 non-null   object
      dtypes: int64(2), object(7)
      memory usage: 73.0+ KB
      ```
      Berdasarkan _output_ diatas, `df_b` memiliki 9 kolom pada dataset ini diantaranya memiliki tipe datanya masing-masing, yaitu:
      - `Unnamed:0` : `int64`
      - `urlDrugName` : `object`
      - `rating` : `int64`
      - `effectiveness` : `object`
      - `sideEffects` : `object`
      - `condition` : `object`
      - `benefitsReview` : `object`
      - `sideEffectsReview` : `object`
      - `commentsReview` : `object`
    
    - Menampilkan deskripsi statistik kolom `rating`
      ``` python
      df_b['rating'].describe()
      ```
      Dengan _output_
      ```
      count    1036.000000
      mean        6.767375
      std         2.976711
      min         1.000000
      25%         5.000000
      50%         8.000000
      75%         9.000000
      max        10.000000
      Name: rating, dtype: float64
      ```
      Fungsi di atas menyediakan informasi statistik deskriptif untuk kolom `review` pada variabel `df_b`, meliputi:
    
      - **`count`**: Jumlah total data dalam kolom.
      - **`mean`**: Nilai rata-rata dari data dalam kolom.
      - **`std`**: Standar deviasi dari data dalam kolom.
      - **`min`**: Nilai terkecil dalam kolom.
      - **`25%`**: Kuartil pertama (Q1), yaitu nilai yang memisahkan 25% data terendah.
      - **`50%`**: Kuartil kedua (Q2) atau median, yaitu nilai tengah dari data.
      - **`75%`**: Kuartil ketiga (Q3), yaitu nilai yang memisahkan 25% data tertinggi.
      - **`max`**: Nilai terbesar dalam kolom.
    
    - Menampilkann total unique value di kolom `Unnamed:0` / ReviewID
      ``` python
      print(df_b['Unnamed: 0'].nunique())
      ```
      Dengan _output_
      ```
      1036
      ```
      Berdasarkan output diatas nilai fungsi `nunique()` dari kolom `Unnamed:0` atau `ReviewID` adalah sebanyak **1036 data unik**
    
    - Menampilkann total unique value di kolom `urlDrugName`
      ``` python
      print(df_b['urlDrugName'].nunique())
      ```
      Dengan _output_
      ```
      314
      ```
      Berdasarkan _output_ diatas nilai `nunique()` dari kolom `urlDrugName` adalah sebanyak **314 data unik** terkait dengan obat.
      
    - Menampilkann total unique value di kolom `effectiveness`
      ``` python
      print(df_b['effectiveness'].nunique())
      ```
      Dengan _output_
      ```
      5
      ```
      Berdasarkan _output_ diatas nilai `nunique()` dari kolom `effectiveness` adalah sebanyak **5 data unik**. Ini sudah merepresentasikan `effectiveness` yang digunakan pada _dataset_ ini dengan skala dengan 5 level.
    
    - Menampilkann total unique value di kolom `sideEffects`
      ``` python
      print(df_b['sideEffects'].nunique())
      ```
      Dengan _output_
      ```
      5
      ```
      Berdasarkan output diatas nilai `nunique()` dari kolom `sideEffects` adalah sebanyak **5 data unik**. Ini sudah merepresentasikan `sideEffects` yang digunakan pada dataset ini dengan skala dengan 5 level.
      
    - Menampilkann total unique value di kolom `condition`
      ``` python
      print(df_b['condition'].nunique())
      ```
      Dengan _output_
      ```
      566
      ```
      Berdasarkan _output_ diatas nilai `nunique()` dari kolom `condition` adalah sebanyak **566 data unik**.
    
    - Menampilkann total unique value di kolom `benefitsReview`
      ``` python
      print(df_b['benefitsReview'].nunique())
      ```
      Dengan _output_
      ```
      1023
      ```
      Berdasarkan _output_ diatas nilai `nunique()` dari kolom `benefitsReview` adalah sebanyak **1023 data unik** atau hampir keseluruhan data merupakan data unik.
    
      ```python
      df_b['benefitsReview']
      ```
      Dengan _output_
      ```
      0       The antibiotic may have destroyed bacteria cau...
      1       Lamictal stabilized my serious mood swings. On...
      2       Initial benefits were comparable to the brand ...
      3       It controlls my mood swings. It helps me think...
      4       Within one week of treatment superficial acne ...
                                    ...                        
      1031    Detoxing effect by pushing out the system thro...
      1032    The albuterol relieved the constriction, irrit...
      1033                      Serve Acne has turned to middle
      1034    My overall mood, sense of well being, energy l...
      1035    Up until 2 years ago, it worked really well on...
      Name: benefitsReview, Length: 1036, dtype: object
      ```
      Berdasarkan _output_ diatas bahwa karakteristik data pada kolom `benefitsReview` adalah berbentuk _Review_ atau _Natural Language_
    
    - Menampilkann total unique value di kolom `sideEffectsReview`
      ```python
      print(df_b['sideEffectsReview'].nunique())
      ```
      Dengan _output_
      ```
      968
      ```
      Berdasarkan output diatas nilai `nunique()` dari kolom `sideEffectsReview` adalah sebanyak **968 data unik** atau hampir keseluruhan data merupakan data unik.
    
      ```python
      df_b['sideEffectsReview']
      ```
      Dengan _output_
      ```
      0                           Some back pain, some nauseau.
      1       Drowsiness, a bit of mental numbness. If you t...
      2       Depakene has a very thin coating, which caused...
      3                 I didnt really notice any side effects.
      4       Side effects included moderate to severe dry s...
                                    ...                        
      1031    Hairloss, extreme dry skin, itchiness, raises ...
      1032                  I have experienced no side effects.
      1033      Painfull muscles, problems with seeing at night
      1034    No side effects of any kind were noted or appa...
      1035    Have stopped using it and have also learned th...
      Name: sideEffectsReview, Length: 1036, dtype: object
      ```
      Berdasarkan _output_ diatas bahwa karakteristik data pada kolom `sideEffectsReview` adalah berbentuk _Review_ atau _Natural Language_
    
    - Menampilkann total unique value di kolom `commentsReview`
      ``` python
      print(df_b['commentsReview'].nunique())
      ```
      Dengan _output_
      ```
      1030
      ```
      Berdasarkan output diatas nilai `nunique()` dari kolom `commentsReview` adalah sebanyak **3047 data unik** atau hampir keseluruhan data merupakan data unik.
    
      ```python
      df_b['commentsReview']
      ```
      Dengan _output_
      ```
      0       Took the antibiotics for 14 days. Sinus infect...
      1       Severe mood swings between hypomania and depre...
      2       Depakote was prescribed to me by a Kaiser psyc...
      3       This drug may not be for everyone but its wond...
      4       Drug was taken in gelatin tablet at 0.5 mg per...
                                    ...                        
      1031    Treatment period is 3 months/12 weeks. Dosage ...
      1032    I use the albuterol as needed because of aller...
      1033    This drug is highly teratogenic ,females must ...
      1034    Divigel is a topically applied Bio-Identical H...
      1035                 Stopped using it for the time being.
      Name: commentsReview, Length: 1036, dtype: object
      ```
    
      Berdasarkan _output_ diatas bahwa karakteristik data pada kolom `commentsReview` adalah berbentuk _Review_ atau _Natural Language_ 
    
    - Pengecekan data duplikat pada dataframe `df_b`
      ```python
      df_b.duplicated().sum()
      ```
      Dengan _output_
      ```
      0
      ```
      Berdasarkan _output_ diatas bahwa tidak terdapat data yang duplikat terhadap dataset `df_b`

    - **Data Visualization df_b**
    
      **Data Visualization** adalah proses penyajian data dalam bentuk grafik atau diagram untuk mempermudah interpretasi dan analisis informasi. Visualisasi data memungkinkan kita untuk mengidentifikasi pola, tren, dan hubungan antar variabel dengan lebih intuitif dibandingkan hanya melihat tabel angka mentah. 

      - Melihat Distribusi 30 `urlDrugName` Teratas
        <div align="center">
        <img src="https://github.com/user-attachments/assets/59bc3c76-59b7-4c6e-aea0-9ea689c80069" alt="Distribusi 30 Obat Teratas df_a" width="500"/>
        <br>
        <b>Gambar 6. Distribusi 30 Obat Teratas Dataset df_b</b>
        </div>
      
        Berdasarkan grafik diatas bahwa persebaran data pada `urlDrugName` cukup merata sehingga tetap dipertahankan untuk analisis berikutnya
      
      - Melihat Distribusi `rating`
        <div align="center">
        <img src="https://github.com/user-attachments/assets/007f90d4-8670-418a-8c49-4fd020576f09" alt="Distribusi Rating df_a" width="500"/>
        <br>
        <b>Gambar 7. Distribusi Rating Dataset df_b</b>
        </div>
      
        Berdasarkan grafik diatas bahwa distribusi `rating` yang **terbesar adalah 10 dengan data 226**, sedangkan `rating` yang **terkecil adalah 2 dengan data 33**
      
      - Melihat Distribusi `effectiveness`
        <div align="center">
        <img src="https://github.com/user-attachments/assets/d09a6183-4b5c-4198-8152-e98ff48d2dc6" alt="Distribusi Efektifitas Obat" width="500"/>
        <br>
        <b>Gambar 8. Distribusi Efektifitas Obat Dataset df_b</b>
        </div>
      
        Berdasarkan grafik diatas bahwa distribusi `effectiveness` yang **terbesar adalah Highly Effective dengan data 411**, sedangkan `effectiveness` yang **terkecil adalah Marginally Effective dengan 76 data**
      
      - Melihat Distribusi `sideEffects`
        <div align="center">
        <img src="https://github.com/user-attachments/assets/49178a77-d901-4139-8408-b49d2d8678bc" alt="Distribusi Efek Samping Obat" width="500"/>
        <br>
        <b>Gambar 9. Distribusi Efek Samping Obat Dataset df_b</b>
        </div>
      
        Berdasarkan grafik diatas bahwa distribusi `sideEffects` yang **terbesar adalah Mid Side Effects dengan data 330**, sedangkan `sideEffects` yang **terkecil adalah Extremely Severe Side Effects dengan 80 data**
              
      - Melihat Distribusi `condition`
        <div align="center">
        <img src="https://github.com/user-attachments/assets/00c92145-0420-4f9b-b8c6-65f1616d8181" alt="Distribusi Kondisi Medis" width="500"/>
        <br>
        <b>Gambar 10. Distribusi Kondisi Medis Dataset df_b</b>
        </div>
      
        Berikut adalah grafik 10 teratas distribusi `condition` terlihat bahwa **despression** dan **acne** menjadi yang paling mencolok dari distribusi lainnya


# Data Preparation

## Penggabungan Dataset

Pada proses ini dilakukan penggabungan dua dataset utama untuk memperkaya informasi dan memudahkan proses analisis data lebih lanjut. Adapun data yang digabungkan adalah `drugLibTrain_raw.tsv` atau variabel `df_a` dengan 3107 baris dan 9 kolom dan `drugLibTest_raw.tsv`atau variabel `df_b` dengan 1036 baris dan 9 kolom

**Alasan:**
Dataset ini dilakukan penggabungan karena memiliki karakteristik yang sama dan kemudahan dalam pengolahan. Selanjutnya, data akan digabungkan menggunakan fungsi `concat()` menjadi variabel `df` agar lebih kaya untuk analisis lanjutan.

```python
df = pd.concat([df_a, df_b])
```
Kemudian dilakukan pengecekan `shape` dari varibel `df`
```python
df.shape
```

Dengan _output_
```
(4143, 9)
```
Berhasil menggabungkan data `df_a` dan `df_b` menjadi variabel `df` dengan menggunakan fungsi `concat()` dengan jumlah baris adalah 4143 dan jumlah kolom adalah 9 

## Handling Data Duplikat
Pada proses ini, langkah yang dilakukan adalah memeriksa dan menghapus data duplikat dari dataset untuk memastikan kualitas data yang baik sebelum masuk ke tahap pemodelan. Proses ini penting untuk menghindari data yang terulang yang dapat mempengaruhi hasil analisis dan performa model rekomendasi.

**Alasan:**  
Data duplikat dapat menyebabkan bias dalam analisis dan model karena beberapa data akan memiliki bobot yang lebih besar dari yang seharusnya. Selain itu, keberadaan data yang berulang dapat membuat model memberikan rekomendasi yang tidak akurat atau menyesatkan. Menghilangkan duplikat membantu menjaga keakuratan hasil rekomendasi dengan memastikan setiap data unik dan representatif.

```python
print("Jumlah data duplikat", df.duplicated().sum())
```
Dengan _output_
```
Jumlah data duplikat 0
```
Sehingga pada operasi ini tidak ada penghapusan data duplikat, karena tidak terdapat data yang duplikat

## Missing Value/ Nilai NaN
Pada proses ini, langkah yang dilakukan adalah memeriksa keberadaan **missing value** atau **nilai NaN** pada dataset dan mengatasinya agar tidak memengaruhi hasil analisis dan pemodelan. Data yang hilang sering kali muncul akibat kesalahan pencatatan atau ketidaklengkapan informasi yang diberikan oleh pengguna.

**Alasan:**  
**Missing value** dapat menyebabkan masalah dalam proses analisis dan pemodelan tidak dapat menangani data yang kosong. Kehadiran nilai yang hilang juga dapat menurunkan kualitas model. Oleh karena itu, langkah penanganan yang tepat seperti **menghapus data yang hilang** untuk menjaga integritas dataset.

```python
df.isnull().sum()
```
Dengan _output_
```
Unnamed: 0           0
urlDrugName          0
rating               0
effectiveness        0
sideEffects          0
condition            1
benefitsReview       0
sideEffectsReview    2
commentsReview       8
dtype: int64
```
Sehingga dilakukan penghapusan nilai `Nan` dengan fungsi `dropna()` pada kolom yang memilki nilai kosong
```python
df = df.dropna(subset=['condition', 'sideEffectsReview', 'commentsReview'])
```
Sehingga dilakukan pengecekan kembali 
```
Unnamed: 0           0
urlDrugName          0
rating               0
effectiveness        0
sideEffects          0
condition            0
benefitsReview       0
sideEffectsReview    0
commentsReview       0
dtype: int64
```
Berdasarkan _output_ diatas, tidak terdapat data pada fitur/kolom `condition` yang terdapat nilai NaN, sehingga tidak ada tindakan penghapusan data

## Filter Frekuensi Data
Pada proses ini, dilakukan filter terhadap kolom `condition` untuk memastikan hanya data dengan frekuensi kemunculan yang cukup tinggi yang digunakan dalam analisis. Filter ini dilakukan dengan menetapkan ambang batas frekuensi lebih besar dari 10 (>10). 

**Alasan**: Kondisi dengan frekuensi kemunculan kurang dari atau sama dengan 30 dianggap terlalu jarang muncul dan cenderung tidak memberikan kontribusi yang signifikan terhadap hasil analisis. Selain itu, data dengan frekuensi rendah sering kali merupakan _outlier_ atau tidak relevan dalam konteks permasalahan yang sedang dianalisis. 

```python
condition_counts = df['condition'].value_counts()
filtered_conditions = condition_counts[condition_counts > 30]
df_filtered = df[df['condition'].isin(filtered_conditions.index)]
```
Berhasil menerapkan filter terhadap kolom `condition`  lebih besar dari 30
```python
print(f"Jumlah kondisi unik setelah filter (>10): {len(filtered_conditions)}")
print(f"Jumlah total baris setelah filter: {len(df_filtered)}")
```
Dengan _output_
```
Jumlah kondisi unik setelah filter (>30): 16
Jumlah total baris setelah filter: 1198
```
Berdasarkan output diatas **data yang digunakan adalah sebanyak 1.198 data** dengan **kondisi unik sebanyak 16 pada kolom `condition`**

## Filter Fitur Penting
Pada proses ini, dataset difokuskan hanya pada fitur-fitur yang relevan untuk membangun sistem rekomendasi obat.

**Alasan pemilihan fitur**:
- Untuk kolom `benefitsReview`,`sideEffectsReview`, dan`commentsReview` tidak akan digunakan pada proyek ini, karena proyek ini **terbatas** dalam pengelolaan kolom dengan kajian atau penanganan data **Natural Language Processing (NLP)**
- Karena proyek ini hanya menerapkan Content-Based Filtering saja dengan pertimbangan `condition` sehingga kolom `rating`, `effectiveness` dan `sideEffects` tidak digunakan

```python
df = df_filtered[['urlDrugName','condition']]
```
Dengan _output_

| urlDrugName  | condition      |
|--------------|----------------|
| prilosec     | acid reflux    |
| propecia     | hair loss      |
| vyvanse      | add            |
| elavil       | depression     |
| claritin     | allergies      |
| ...          | ...            |
| ambien       | insomnia       |
| nexium       | acid reflux    |
| retin-a      | acne           |
| vyvanse      | adhd           |
| proair-hfa   | asthma         |

_1198 rows × 2 columns_

Sekarang variabel `df` hanya berisikan fitur :
- `urlDrugName`
- `condition`

Dengan jumlah data **1198 baris dengan 2 kolom**

# Modelling dan Result
Pada proses ini akan dilakukan permodelan sistem rekomendasi dan menerapkan model tersebut. Adapun pendekatan Sistem Rekomendasi yang digunakan adalah**Content-Based Filtering** adalah pendekatan dalam sistem rekomendasi yang menggunakan atribut-atribut atau fitur-fitur dari item untuk menentukan kesamaan antara item yang ada. Pendekatan ini berfokus pada karakteristik item yang telah dinilai pengguna untuk merekomendasikan item serupa.

**Kelebihan:**
- Tidak memerlukan data dari pengguna lain, sehingga cocok untuk personalisasi.
- Efektif dalam menangani _cold start_ untuk pengguna baru.
- Rekomendasi dapat dijelaskan dengan mudah karena berbasis pada fitur yang relevan.

***Kekurangan:***
- Cenderung menghasilkan rekomendasi yang terlalu spesifik dan tidak bervariasi (_overspecialization_).
- Bergantung pada ketersediaan dan kualitas fitur/atribut item.
- Tidak dapat merekomendasikan item yang berbeda dari sejarah preferensi pengguna (tidak ada eksplorasi).

Pendekatan ini menggunakan atribut-atribut atau fitur-fitur item untuk menentukan kesamaan antara item yang ada. Dalam konteks proyek ini, content-based filtering akan memberikan rekomendasi obat (_drug_) berdasarkan **condition** dari dataset yang tersedia.

- **Modelling**

  - Penggabungan Fitur dan Membuat matriks TF-IDF
    ```python
    tf_id = TfidfVectorizer(stop_words='english')
    tf_id.fit(df['condition'])
    tf_id.get_feature_names_out()
    ```
    Dengan _output_
    ```
    array(['acid', 'acne', 'add', 'adhd', 'allergies', 'anxiety', 'asthma',
         'birth', 'blood', 'cholesterol', 'control', 'depression', 'hair',
         'high', 'hypothyroidism', 'insomnia', 'loss', 'migraine',
         'migraines', 'pressure', 'reflux'], dtype=object)
    ```
    Berdasarkan _output_ diatas menghasilkan _array_ yang berisi nilai-nilai yang ada pada kolom `condition`
  
  - Melihat ukuran matrix tfidf
    ```python
    tfidf_matrix = tf_id.fit_transform(df['condition'])
    tfidf_matrix.shape
    ```
    Dengan _output_
    ```
    (1198, 21)
    ```
    Berdasarkan _output_ diatas menampilkan **ukuran data 1198 baris x 21 kolom**, sehingga akan masuk tahap pembuatan matrix agar bisa dianalisa lebih lanjut
  
  - Mengubah vektor tf-idf yang berbentuk matriks menggunakan fungsi `todense()`
    ```python
    tfidf_matrix.todense()
    ```
    Dengan _output_
    ```
    matrix([[0.70710678, 0.        , 0.        , ..., 0.        , 0.        ,
           0.70710678],
          [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
           0.        ],
          [0.        , 0.        , 1.        , ..., 0.        , 0.        ,
           0.        ],
          ...,
          [0.        , 1.        , 0.        , ..., 0.        , 0.        ,
           0.        ],
          [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
           0.        ],
          [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
           0.        ]])
    ```
    Berdasarkan _output_ diatas berhasil menerapkan fungsi `todense()` untuk membentuk matrix
  
  - Membuat DataFrame untuk melihat TF-IDF matrix
    ```python
    tfidf_df = pd.DataFrame(
      tfidf_matrix.todense(),                    
      columns=tf_id.get_feature_names_out(),  
      index=df['urlDrugName'])
    ```
    Berhasil membuat Dataframe `tfidf_df`
  
  - Proses perhitungan `cosine_similarity()`
    ```python
    cosine_sim = cosine_similarity(tfidf_matrix)
    cosine_sim
    ```
    Dengan _output_
    ```
    array([[1., 0., 0., ..., 0., 0., 0.],
         [0., 1., 0., ..., 0., 0., 0.],
         [0., 0., 1., ..., 0., 0., 0.],
         ...,
         [0., 0., 0., ..., 1., 0., 0.],
         [0., 0., 0., ..., 0., 1., 0.],
         [0., 0., 0., ..., 0., 0., 1.]])
    ```
    Berdasarkan _output_ diatas, proses perhitungan `cosine_similarity` telah berhasil dilakukan.
  
  - Membuat DataFrame dari matriks kesamaan
    ```python
    cosine_sim_df = pd.DataFrame(
      cosine_sim,                      
      index=df['urlDrugName'],         
      columns=df['urlDrugName'])
  
    print('Ukuran Dataframe : ', cosine_sim_df.shape)
    ```
    Dengan _output_
    ```
    Ukuran Dataframe :  (1198, 1198)
    ```
    Berdasarkan output diatas, proses pembuatan dataframe berhasil dengan nama `cosine_sim_df` dilakukan dan dataframe memiliki ukuran 1198 x 1198
  
  - Membangun Fungsi Utama rekomendasi obat berbasis Content-Based Filtering
    ```python
    def drug_recommendations(drug_name, n=5):
      global cosine_sim_df
      cosine_sim_df = cosine_sim_df[~cosine_sim_df.index.duplicated(keep='first')]
      cosine_sim_df = cosine_sim_df.loc[:, ~cosine_sim_df.columns.duplicated(keep='first')]
      
      if drug_name not in cosine_sim_df.index:
          raise ValueError(f"Drug name '{drug_name}' not found in similarity matrix.")
          
      sim_scores = cosine_sim_df.loc[drug_name].sort_values(ascending=False)
      sim_scores = sim_scores.drop(drug_name, errors='ignore')
      top_recommendations = sim_scores.head(n * 2).index.tolist()
      
      recommendations_df = (
          df[df['urlDrugName'].isin(top_recommendations)][['urlDrugName', 'condition']]
          .drop_duplicates(subset=['urlDrugName'])
          .head(n)
      )
      return recommendations_df
    ```
    Berikut diatas merupakan _function_ yang diguunakan untuk dilakukannya rekomendasi obat berbasis Content-Based Filtering

- **Result**
  
  Pada proses hasil ini akan dicoba rekomendasi obat yang lain `lexapro` yang merupakan obat antidepresan:

  ```python
   df[df['urlDrugName'].eq('lexapro')].head(1)
  ```
  Dengan _output_
  
  | urlDrugName | condition   |
  |-------------|-------------|
  | lexapro     | depression  |

  Kemudian memanggil fungsi `drug_recommendations()` untuk mendapatkan hasil `Top-N Recommendation`
  
  ```python
  recommendations_result = drug_recommendations('lexapro')
  recommendations_result
  ```
  Dengan _output_
  
  | urlDrugName     | condition      |
  |------------------|----------------|
  | wellbutrin-xl   | depression     |
  | wellbutrin-sr   | depression     |
  | zoloft          | depression     |
  | paxil           | depression     |
  | citalopram      | depression     |
  
  Berikut ini adalah hasil `Top-N Recommendation` yang merupakan rekomendasi obat menggunakan Content-Based Filtering. 
  
  Berdasarkan percobaan diatas, jika input yang diberikan adalah obat dengan nama `'lexapro'`, model menghasilkan rekomendasi obat-obat lain seperti `wellbutrin-xl`, `wellbutrin-sr`, `zoloft`, `paxil`, dan `citalopram`, yang semuanya memiliki kondisi `depression`. 
  
  Hasil ini menunjukkan bahwa metode ini dapat membantu menemukan alternatif obat yang sesuai untuk kondisi yang sama, sehingga mempermudah pengguna dalam memilih obat yang relevan.

# Evaluation

Untuk mengukur bagaimana performa dari model yang telah dibuat, diperlukan metrik evaluasi untuk mengevaluasi model _Recommender Content-Based System_ berbasis obat. Adapun metrik evaluasi yang digunakan adalah **Precision**, dengan penjelasan formula, konteks, dan cara kerjanya terhadap model sebagai berikut:

- **Penjelasan Formula dan Konteks Terhadap Proyek**

  **Precision** adalah metrik yang digunakan untuk mengevaluasi relevansi hasil rekomendasi model. **Precision** mengukur proporsi rekomendasi yang relevan (benar) dibandingkan dengan jumlah total rekomendasi yang diberikan. Metrik ini sangat penting untuk memastikan bahwa model memberikan rekomendasi yang tepat sasaran dan berguna bagi pengguna.
  
  Formula untuk menghitung **Precision** adalah sebagai berikut:

  $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
  
  **Keterangan:**
  - **True Positive (TP):** Jumlah rekomendasi yang relevan dan benar 
  - **False Positive (FP):** Jumlah rekomendasi yang tidak relevan atau salah 
  
  Dalam konteks _Recommender System_, formula tersebut dapat disesuaikan menjadi:

  $$\text{Precision} = \frac{\text{Jumlah Rekomendasi Relevan}}{\text{Jumlah Total Rekomendasi}}$$
  
  **Keterangan:**
  - **Jumlah Rekomendasi Relevan:** Rekomendasi obat yang sesuai dengan kondisi medis (_condition_).
  - **Jumlah Total Rekomendasi:** Seluruh obat yang direkomendasikan oleh model berdasarkan _input_ obat awal.

- **Cara Kerja**

  Formula tersebut mengukur **Precision** dalam konteks sistem rekomendasi obat. **Precision** dihitung dengan membagi jumlah rekomendasi yang relevan dengan jumlah total item yang direkomendasikan.
  
  Jika model merekomendasikan 10 obat dan 7 di antaranya relevan dengan kondisi medis, maka nilai **Precision** adalah:

  $$\text{Precision} = \frac{7}{10} = 0.7 \, \text{atau} \, 70\%.$$
  
  Ini menunjukkan bahwa 70% dari rekomendasi yang diberikan oleh model relevan dan bermanfaat bagi pasien.

- **Implementasi**
  - Fungsi untuk menghitung Precision
    ```python
    def calculate_precision(drug_name, condition, top_n=5):
    recommended_drugs = drug_recommendations(drug_name, n=top_n)
    return (
        len(recommended_drugs[recommended_drugs['condition'] == condition]) / len(recommended_drugs)
        if len(recommended_drugs) > 0 else 0 )
    ```
    Berikut diatas adalah fungsi `calculate_precision` dari **Precision** untuk kasus ini

  - Menghitung Precision untuk setiap kombinasi `urlDrugName` dan `condition`
    ```python
    precision_results_df = (df.groupby('condition').first()[['urlDrugName']].reset_index())

    precision_results_df['precision'] = precision_results_df.apply(
    lambda row: calculate_precision(row['urlDrugName'], row['condition'], top_n=5), axis=1)

    precision_results_df = precision_results_df.sort_values(by='precision', ascending=False)
    overall_precision = precision_results_df['precision'].mean()
    ```
    Berikut diatas adalah proses menghitung precision untuk setiap kombinasi `drug_name` dan `condition` sehingga mengghasilkan variabel `precision_result_df` sebagai hasil dan variabel `overall_precision` sebagai rerata precision

  - Mencetak hasil Precision dan _overall_ nya
    ```python
    print(precision_results_df)
    print(f"Overall Precision: {overall_precision:}")
    ```
    Dengan _output_
    ```
                  condition       urlDrugName  precision
    0           acid reflux          prilosec        1.0
    1                  acne            sotret        1.0
    2                   add           vyvanse        1.0
    4             allergies          claritin        1.0
    5               anxiety        effexor-xr        1.0
    6                asthma         singulair        1.0
    7         birth control  ortho-tri-cyclen        1.0
    8            depression            elavil        1.0
    10  high blood pressure            lotrel        1.0
    13             insomnia            ambien        1.0
    15            migraines           prempro        1.0
    12       hypothyroidism         synthroid        0.8
    9             hair loss          propecia        0.6
    11     high cholesterol           crestor        0.6
    14             migraine             zomig        0.6
    3                  adhd           vyvanse        0.0
    Overall Precision: 0.85
    ```
    
Berdasarkan hasil _output_ diatas, berikut adalah hasil **precision** yang dihasil model untuk setiap `condition` adalah sebagai berikut:

1. **Precision Tertinggi**: Sebanyak 11 kondisi, seperti `acid reflux`, `acne`, `add`, dan lainnya, memiliki **precision** sebesar **1.0**, menunjukkan bahwa semua rekomendasi obat untuk kondisi tersebut sangat relevan.
2. **Precision Menengah**: Kondisi seperti `hypothyroidism`, `hair loss`, `high cholesterol`, dan `migraine` memiliki **precision** **0.6 hingga 0.8**, menunjukkan rekomendasi cukup relevan, tetapi tidak sepenuhnya akurat.
3. **Precision Terendah**: Kondisi `adhd` memiliki **precision** **0.0**, menunjukkan bahwa rekomendasi obat untuk kondisi ini tidak relevan sama sekali.

Rata-rata **precision** untuk seluruh kondisi adalah **0.85**

# Referensi
[1]	R. D. Zaafira and Y. Yardi, “Analysis of the Effectiveness of Drug Management Systems in Tangerang Selatan General Hospital in 2021,” J. Farm. Galen. (Galenika J. Pharmacy), vol. 10, no. 1, pp. 62–72, Mar. 2024, doi: 10.22487/j24428744.2024.v10.i1.16479.

[2]	A. S. Pertiwi and T. N. Rochmah, “Implementation of Theory of Constraint on Waiting Time of Prescription Service,” J. Adm. Kesehat. Indones., vol. 7, no. 1, p. 1, Jun. 2019, doi: 10.20473/jaki.v7i1.2019.1-8.

[3]	A. Sae-Ang, S. Chairat, N. Tansuebchueasai, O. Fumaneeshoat, T. Ingviya, and S. Chaichulee, “Drug Recommendation from Diagnosis Codes: Classification vs. Collaborative Filtering Approaches,” Int. J. Environ. Res. Public Health, vol. 20, no. 1, p. 309, Dec. 2022, doi: 10.3390/ijerph20010309.

[4]	R. F. T. Ceskoutsé, A. B. Bomgni, D. R. Gnimpieba Zanfack, D. D. M. Agany, T. Bouetou Bouetou, and E. Gnimpieba Zohim, “Sub-clustering based recommendation system for stroke patient: Identification of a specific drug class for a given patient,” Comput. Biol. Med., vol. 171, p. 108117, Mar. 2024, doi: 10.1016/j.compbiomed.2024.108117.

[5]	C. Fiarni and H. Maharani, “Product Recommendation System Design Using Cosine Similarity and Content-based Filtering Methods,” IJITEE (International J. Inf. Technol. Electr. Eng., vol. 3, no. 2, p. 42, Sep. 2019, doi: 10.22146/ijitee.45538.

[6]	Q. Y. Shambour, M. M. Al-Zyoud, A. H. Hussein, and Q. M. Kharma, “A doctor recommender system based on collaborative and content filtering,” Int. J. Electr. Comput. Eng., vol. 13, no. 1, p. 884, Feb. 2023, doi: 10.11591/ijece.v13i1.pp884-893.

[7]	D. Çelik Ertuğrul and A. Elçi, “A survey on semanticized and personalized health recommender systems,” Expert Syst., vol. 37, no. 4, Aug. 2020, doi: 10.1111/exsy.12519.

[8]	E. Saad et al., “Determining the Efficiency of Drugs Under Special Conditions From Users’ Reviews on Healthcare Web Forums,” IEEE Access, vol. 9, pp. 85721–85737, 2021, doi: 10.1109/ACCESS.2021.3088838.






