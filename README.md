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
1. Bagaimana mengetahui dan memahami karakteristik _dataset_  dalam pembuatan model _Recommender Content-Based System_ Berbasis Obat
2. Bagaimana membuat model _Recommender System_ dengan pendekatan _content-based filtering_?
3. Bagaimana cara mengukur performa (evaluasi) model _Recommender Content-Based System_ Berbasis Obat yang sudah dibangun?

## Goals
1. Melakukan proses _Exploratory Data Analysis_ (EDA) dan Data Visualization untuk memahami karakteristik data
2. Merancang dan Membangun model _Recommender System_ dengan pendekatan _content-based filtering_
3. Melakukan evaluasi terhadap model _Recommender System_ _content-based filtering_

# Data Understanding 
Dataset yang digunakan untuk pembuatan model sistem rekomendasi ini adalah dataset "Drug Review Dataset (Drugs.com)" yang tersedia di situs [UCI Machine Learning Repository](https://doi.org/10.24432/C55G6J). Dataset ini berisi ulasan pasien terkait obat-obatan tertentu beserta kondisi medis yang relevan.

Terdapat dua file utama yang digunakan:
1. **`drugLibTrain_raw.tsv`** – Berisi 3107 baris dan 5 kolom.
2. **`drugLibTest_raw.tsv`** – Berisi 1036 baris dan 5 kolom.

Dataset ini digunakan untuk membangun sistem rekomendasi berbasis **Content-Based Filtering**

Dataset tersebut dapat diunduh di [sini](https://doi.org/10.24432/C55G6J).

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

## Exploratory Data Analysis
**Exploratory Data Analysis (EDA)** adalah proses eksplorasi dan analisis awal terhadap data untuk memahami struktur, pola, dan karakteristiknya sebelum dilakukan pemodelan lebih lanjut. EDA bertujuan untuk mengidentifikasi distribusi data, mendeteksi nilai yang hilang atau pencilan, serta menganalisis hubungan antar variabel.

- Menampilan jumlah baris dan kolom yang ada pada dataset
  ``` python
  df.shape
  ```
  Dengan _output_:
  ```
  (4143, 9)
  ```
  Berdasarkan _output_ diatas, `df` memiliki:
  - 4143 baris data
  - 9 kolom data
  
  Untuk selanjutnya kita akan melihat lebih jauh setiap kolom, cek data kosong dan tipe datanya

- Menampilkan kolom, tipe data, cek data kosong dari setiap kolom yang ada
  ``` python
  df.info()
  ```
  Dengan _output_:
  ```
  <class 'pandas.core.frame.DataFrame'>
  Int64Index: 4143 entries, 0 to 1035
  Data columns (total 9 columns):
   #   Column             Non-Null Count  Dtype 
  ---  ------             --------------  ----- 
   0   Unnamed: 0         4143 non-null   int64 
   1   urlDrugName        4143 non-null   object
   2   rating             4143 non-null   int64 
   3   effectiveness      4143 non-null   object
   4   sideEffects        4143 non-null   object
   5   condition          4142 non-null   object
   6   benefitsReview     4143 non-null   object
   7   sideEffectsReview  4141 non-null   object
   8   commentsReview     4135 non-null   object
  dtypes: int64(2), object(7)
  memory usage: 323.7+ KB
  ```
  Berdasarkan _output_ diatas, `df` memiliki 9 kolom pada dataset ini diantaranya memiliki tipe datanya masing-masing, yaitu:
  - `Unnamed:0` : `int64`
  - `urlDrugName` : `object`
  - `rating` : `int64`
  - `effectiveness` : `object`
  - `sideEffects` : `object`
  - `condition` : `object`
  - `benefitsReview` : `object`
  - `sideEffectsReview` : `object`
  - `commentsReview` : `object`

  _Insight_:
  - Masih terdapat data yang **`Non-Null Count`nya tidak merata** alias terdapat data yang `NaN`, hal ini akan ditangani pada Proses **Data Preparation**
  - Untuk kolom `rating` yang memiliki tipe data `int64` akan dilakukan analisis lebih lanjut yakni dengan fungsi `describe()`, untuk kolom `Unnamed:0` / `reviewID` diabaikan karena hanya bersifat `index` saja
  - Untuk kolom lain seperti `effectiveness`, `sideEffects`, `condition`, `benefitsReview`, `sideEffectsReview`, dan `commentsReview` yang memiliki tipe data `object` akan dilakukan analisis lebih lanjut yakni fungsi `nunique()`

- Menampilkan deskripsi statistik kolom `rating`
  ``` python
  df['rating'].describe()
  ```
  Dengan _output_
  ```
  count    4143.000000
  mean        6.946416
  std         2.948868
  min         1.000000
  25%         5.000000
  50%         8.000000
  75%         9.000000
  max        10.000000
  Name: rating, dtype: float64
  ```
  Fungsi di atas menyediakan informasi statistik deskriptif untuk kolom `review`, meliputi:

  - **`count`**: Jumlah total data dalam kolom.
  - **`mean`**: Nilai rata-rata dari data dalam kolom.
  - **`std`**: Standar deviasi dari data dalam kolom.
  - **`min`**: Nilai terkecil dalam kolom.
  - **`25%`**: Kuartil pertama (Q1), yaitu nilai yang memisahkan 25% data terendah.
  - **`50%`**: Kuartil kedua (Q2) atau median, yaitu nilai tengah dari data.
  - **`75%`**: Kuartil ketiga (Q3), yaitu nilai yang memisahkan 25% data tertinggi.
  - **`max`**: Nilai terbesar dalam kolom.
  
  Untuk analisis lebih jauh, akan dilakukan **Data Visualization** untuk melihat persebaran data secara visual

- Menampilkann total unique value di kolom `Unnamed:0` / ReviewID
  ``` python
  print(df['Unnamed: 0'].nunique())
  ```
  Dengan _output_
  ```
  4143
  ```
  Berdasarkan output diatas nilai fungsi `nunique()` dari kolom `Unnamed:0` atau `ReviewID` adalah sebanyak **4143 data unik**, sehingga pada proses **Data Preparation**, kolom ini akan dihapus karena hanya bersifat `index` saja

- Menampilkann total unique value di kolom `urlDrugName`
  ``` python
  print(df['urlDrugName'].nunique())
  ```
  Dengan _output_
  ```
  541
  ```
  Berdasarkan _output_ diatas nilai `nunique()` dari kolom `urlDrugName` adalah sebanyak **541 data unik** terkait dengan obat. Untuk analisis lebih jauh, akan dilakukan **Data Visualization** untuk melihat persebaran data

- Menampilkann total unique value di kolom `effectiveness`
  ``` python
  print(df['effectiveness'].nunique())
  ```
  Dengan _output_
  ```
  5
  ```
  Berdasarkan _output_ diatas nilai `nunique()` dari kolom `effectiveness` adalah sebanyak **5 data unik**. Ini sudah merepresentasikan `effectiveness` yang digunakan pada _dataset_ ini dengan skala dengan 5 level.

- Menampilkann total unique value di kolom `sideEffects`
  ``` python
  print(df['sideEffects'].nunique())
  ```
  Dengan _output_
  ```
  5
  ```
  Berdasarkan output diatas nilai `nunique()` dari kolom `sideEffects` adalah sebanyak **5 data unik**. Ini sudah merepresentasikan `sideEffects` yang digunakan pada dataset ini dengan skala dengan 5 level. Untuk analisis lebih jauh, akan dilakukan **Data Visualization** untuk melihat persebaran data

- Menampilkann total unique value di kolom `condition`
  ``` python
  print(df['condition'].nunique())
  ```
  Dengan _output_
  ```
  1807
  ```
  Berdasarkan _output_ diatas nilai `nunique()` dari kolom `condition` adalah sebanyak **1807 data unik**. Untuk analisis lebih jauh, akan dilakukan **Data Visualization** untuk melihat persebaran data.

- Menampilkann total unique value di kolom `benefitsReview`
  ``` python
  print(df['benefitsReview'].nunique())
  ```
  Dengan _output_
  ```
  4028
  ```
  Berdasarkan _output_ diatas nilai `nunique()` dari kolom `benefitsReview` adalah sebanyak **4028 data unik** atau hampir keseluruhan data merupakan data unik.

  ```python
  df['benefitsReview']
  ```
  Dengan _output_
  ```
  0       slowed the progression of left ventricular dys...
  1       Although this type of birth control has more c...
  2       I was used to having cramps so badly that they...
  3       The acid reflux went away for a few months aft...
  4       I think that the Lyrica was starting to help w...
                                ...                        
  1031    Detoxing effect by pushing out the system thro...
  1032    The albuterol relieved the constriction, irrit...
  1033                      Serve Acne has turned to middle
  1034    My overall mood, sense of well being, energy l...
  1035    Up until 2 years ago, it worked really well on...
  Name: benefitsReview, Length: 4143, dtype: object
  ```
  Berdasarkan _output_ diatas bahwa karakteristik data pada kolom `benefitsReview` adalah berbentuk _Review_ atau _Natural Language_ , sehingga pada proses **Data Preparation** kolom ini tidak digunakan

- Menampilkann total unique value di kolom `sideEffectsReview`
  ```python
  print(df['sideEffectsReview'].nunique())
  ```
  Dengan _output_
  ```
  3746
  ```
  Berdasarkan output diatas nilai `nunique()` dari kolom `sideEffectsReview` adalah sebanyak **3746 data unik** atau hampir keseluruhan data merupakan data unik.

  ```python
  df['sideEffectsReview']
  ```
  Dengan _output_
  ```
  0       cough, hypotension , proteinuria, impotence , ...
  1       Heavy Cycle, Cramps, Hot Flashes, Fatigue, Lon...
  2              Heavier bleeding and clotting than normal.
  3       Constipation, dry mouth and some mild dizzines...
  4       I felt extremely drugged and dopey.  Could not...
                                ...                        
  1031    Hairloss, extreme dry skin, itchiness, raises ...
  1032                  I have experienced no side effects.
  1033      Painfull muscles, problems with seeing at night
  1034    No side effects of any kind were noted or appa...
  1035    Have stopped using it and have also learned th...
  Name: sideEffectsReview, Length: 4143, dtype: object
  ```
  Berdasarkan _output_ diatas bahwa karakteristik data pada kolom `sideEffectsReview` adalah berbentuk _Review_ atau _Natural Language_ , sehingga pada proses **Data Preparation** kolom ini tidak digunakan

- Menampilkann total unique value di kolom `commentsReview`
  ``` python
  print(df['commentsReview'].nunique())
  ```
  Dengan _output_
  ```
  4056
  ```
  Berdasarkan output diatas nilai `nunique()` dari kolom `commentsReview` adalah sebanyak **4056 data unik** atau hampir keseluruhan data merupakan data unik.

  ```python
  df['commentsReview']
  ```
  Dengan _output_
  ```
  0       monitor blood pressure , weight and asses for ...
  1       I Hate This Birth Control, I Would Not Suggest...
  2       I took 2 pills at the onset of my menstrual cr...
  3       I was given Prilosec prescription at a dose of...
  4                                               See above
                                ...                        
  1031    Treatment period is 3 months/12 weeks. Dosage ...
  1032    I use the albuterol as needed because of aller...
  1033    This drug is highly teratogenic ,females must ...
  1034    Divigel is a topically applied Bio-Identical H...
  1035                 Stopped using it for the time being.
  Name: commentsReview, Length: 4143, dtype: object
  ```

  Berdasarkan _output_ diatas bahwa karakteristik data pada kolom `commentsReview` adalah berbentuk _Review_ atau _Natural Language_ , sehingga pada proses **Data Preparation** kolom ini tidak digunakan

- Pengecekan data duplikat pada dataframe `df`
  ```python
  df.duplicated().sum()
  ```
  Dengan _output_
  ```
  0
  ```
  Berdasarkan _output_ diatas bahwa tidak terdapat data yang duplikat terhadap dataset `df`
    
  


  

  

# Referensi
[1]	R. D. Zaafira and Y. Yardi, “Analysis of the Effectiveness of Drug Management Systems in Tangerang Selatan General Hospital in 2021,” J. Farm. Galen. (Galenika J. Pharmacy), vol. 10, no. 1, pp. 62–72, Mar. 2024, doi: 10.22487/j24428744.2024.v10.i1.16479.

[2]	A. S. Pertiwi and T. N. Rochmah, “Implementation of Theory of Constraint on Waiting Time of Prescription Service,” J. Adm. Kesehat. Indones., vol. 7, no. 1, p. 1, Jun. 2019, doi: 10.20473/jaki.v7i1.2019.1-8.

[3]	A. Sae-Ang, S. Chairat, N. Tansuebchueasai, O. Fumaneeshoat, T. Ingviya, and S. Chaichulee, “Drug Recommendation from Diagnosis Codes: Classification vs. Collaborative Filtering Approaches,” Int. J. Environ. Res. Public Health, vol. 20, no. 1, p. 309, Dec. 2022, doi: 10.3390/ijerph20010309.

[4]	R. F. T. Ceskoutsé, A. B. Bomgni, D. R. Gnimpieba Zanfack, D. D. M. Agany, T. Bouetou Bouetou, and E. Gnimpieba Zohim, “Sub-clustering based recommendation system for stroke patient: Identification of a specific drug class for a given patient,” Comput. Biol. Med., vol. 171, p. 108117, Mar. 2024, doi: 10.1016/j.compbiomed.2024.108117.

[5]	C. Fiarni and H. Maharani, “Product Recommendation System Design Using Cosine Similarity and Content-based Filtering Methods,” IJITEE (International J. Inf. Technol. Electr. Eng., vol. 3, no. 2, p. 42, Sep. 2019, doi: 10.22146/ijitee.45538.

[6]	Q. Y. Shambour, M. M. Al-Zyoud, A. H. Hussein, and Q. M. Kharma, “A doctor recommender system based on collaborative and content filtering,” Int. J. Electr. Comput. Eng., vol. 13, no. 1, p. 884, Feb. 2023, doi: 10.11591/ijece.v13i1.pp884-893.

[7]	D. Çelik Ertuğrul and A. Elçi, “A survey on semanticized and personalized health recommender systems,” Expert Syst., vol. 37, no. 4, Aug. 2020, doi: 10.1111/exsy.12519.

[8]	E. Saad et al., “Determining the Efficiency of Drugs Under Special Conditions From Users’ Reviews on Healthcare Web Forums,” IEEE Access, vol. 9, pp. 85721–85737, 2021, doi: 10.1109/ACCESS.2021.3088838.






