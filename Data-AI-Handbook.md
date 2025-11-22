---
tone: |-
  pendekatannya kayak teman yang saling berbagi, memang kami berasa jago gitu tapi kita gak terlalu menggurui, dan ngasih space buat eksplor di hal lain, selagi kita ngasih reousrce yang cukup

  ringan, padat, mudah diaplikasikan
desain: Modern, Neon, Sederhana
tags:
  - ML
  - AI
  - Mahasiswa
version: "0.9"
progress: 75
isi tiap bab: |-
  paragraf: penjelasan, penjabaran teori
  bullet list : resources, checklist
  table: perbandingan, penjelasan dari jenis jenis
  gambar: intinya gambar punya orang untuk mempermudah memahami sesuatu
  kode: kode contoh penggunaan library/framework, penggunaan
  quotes: bisa berupa recommendation, question,
---
## Table of Contents
- [[#Table of Contents|Table of Contents]]
- [[#Pengantar|Pengantar]]
- [[#Data and Digital Use case|Data and Digital Use case]]
	- [[#Data and Digital Use case#Understanding Data in AI Contexts|Understanding Data in AI Contexts]]
	- [[#Data and Digital Use case#How AI work with Data|How AI work with Data]]
	- [[#Data and Digital Use case#Where to Get Data|Where to Get Data]]
- [[#Data Analysis and Preprocessing|Data Analysis and Preprocessing]]
	- [[#Data Analysis and Preprocessing#Data Preprocessing|Data Preprocessing]]
## Pengantar
Materi tentang Data dan ML ini aku kurasi dari berbagai sumber dan pengalaman pribadi—mulai dari kuliah teknik sains data di kampus hingga pengalaman kerja—supaya kamu bisa langsung belajar hal-hal yang benar-benar penting, applicable, dan yang paling penting ber Bahasa Indonesia. Cocok untuk:

- Memulai proyek ML pertama
- Persiapan lomba AI/data science
- Persiapan magang di bidang Data/AI
- Implementasi solusi AI untuk tugas kuliah atau skripsi
- Membuat Proof of Concept untuk startup/ide bisnis AI

> This ebook is 75% written by human, 25% AI-assisted. The percentage reflects time allocation in resource curation, structuring, and sentence crafting.

---
## Data and Digital Use case
### Understanding Data in AI Contexts

Dalam dunia AI, data adalah bahan baku utama yang memberi “pengetahuan” bagi mesin untuk belajar dan membuat keputusan.

Data digunakan untuk melatih model AI agar mengenali pola, memprediksi hasil, dan memberikan solusi berbasis pengalaman sebelumnya. Formatnya tidak hanya berupa angka, tetapi juga mencakup teks, gambar, suara, dan video. Semakin banyak dan semakin berkualitas data yang digunakan, semakin baik pula performa model AI.

| Jenis Data           | Penjelasan                                                      | Contoh Penggunaan                                                    |
| -------------------- | --------------------------------------------------------------- | -------------------------------------------------------------------- |
| Structured Data      | Data tabel dengan format tetap (misalnya CSV, database).        | Data pelanggan untuk prediksi churn atau transaksi keuangan.         |
| Semi-structured Data | Tidak seketat tabel, masih punya struktur (misalnya JSON, XML). | Log API, file sensor IoT, data chatbot.                              |
| Unstructured Data    | Tidak punya format tetap seperti teks, gambar, audio, video.    | Analisis sentimen media sosial, pengenalan wajah, transkripsi suara. |
| Synthetic Data       | Data buatan yang menyerupai data asli untuk melatih model.      | Latihan AI generatif tanpa melibatkan data pengguna nyata.           |

Sumber data sangat beragam, mulai dari sensor IoT, media sosial, hingga sistem internal. Tiap sumber punya format dan kualitas berbeda yang perlu dikelola dengan tepat.

Dari sisi engineering, pembuat proyek perlu memahami cara data dikumpulkan dan diolah agar bisa dimanfaatkan secara efektif dalam pengembangan AI.

### How AI work with Data

If the data is already, How AI Leverages data to learn, adapt, and deliver the best outcomes?

- Training AI Models: Untuk membangun model AI yang efektif, data digunakan sebagai bahan pelatihan. Contohnya pada NLP, model dilatih menggunakan kumpulan besar data teks agar mampu memahami tata bahasa, makna kata, hingga analisis sentimen.
- Data-Driven Decision Making: Data yang berkualitas memungkinkan sistem AI membuat keputusan secara akurat dan real-time. Sebagai contoh, mobil otonom memanfaatkan data dari berbagai sensor untuk mengenali lingkungan dan menavigasi jalan dengan aman. Contoh lainnya, market analysis bot menggunakan data transaksi, tren harga, dan sentimen pasar untuk memberikan rekomendasi investasi.
- Personalization and Recommendations - Algoritma AI menggunakan data perilaku dan preferensi pengguna untuk menghasilkan pengalaman yang lebih personal. Contohnya Algoritma konten di Tiktok.
%% %%
lorem ipsum lorem ipsum lorem ipsum

Dalam dunia AI, data adalah bahan baku utama yang memberi “pengetahuan” bagi mesin untuk belajar dan membuat keputusan.

Data digunakan untuk melatih model AI agar mengenali pola, memprediksi hasil, dan memberikan solusi berbasis pengalaman sebelumnya. Formatnya tidak hanya berupa angka, tetapi juga mencakup teks, gambar, suara, dan video. Semakin banyak dan semakin berkualitas data yang digunakan, semakin baik pula performa model AI.

### Where to Get Data

Public Datasets: Free and accessible.

- [Kaggle](https://www.kaggle.com/datasets): Nyediain ribuan dataset unggahan pengguna dan kompetisi analisis data, untuk pemula serta dataset lanjutan untuk ML dan AI tingkat lanjut.​
- [UCI Machine Learning Repository](https://archive.ics.uci.edu): Menampilkan banyak dataset klasik untuk keperluan riset dan pelatihan model, seperti MNIST dan Iris
- [Google Dataset Search](https://datasetsearch.research.google.com): Mesin pencari khusus untuk dataset lintas bidang, sehingga mudah menemukan dataset dari berbagai sumber open source
- DataHub & Open Data Hub: Katalog dataset dari berbagai sektor skala internasional.
- [World Bank](https://data.worldbank.org): Kumpulan dataset lintas negara dan bidang pembangunan.
- Portal Pemerintah ([Data.go.id](https://data.go.id), [DataIndonesia.id](https://dataindonesia.id), [BPS](https://www.bps.go.id)):
- [Huggingface datasets](https://huggingface.co/datasets)
- [Stanford Large Network Dataset Collection (SNAP)](https://snap.stanford.edu/data)
- [Amazon AWS Public Datasets](https://registry.opendata.aws)

API-Based Sources: Real-time data pulls.

- Twitter API for sentiment analysis data; OpenWeatherMap for climate data.
- Custom Sources: Collect your own via surveys, sensors, or web scraping (ethically and legally).

Checklist for Data

□ Define your project's data needs (e.g., volume, type).
□ Assess data quality: Is it accurate, complete, and unbiased?
□ Document sources and preprocessing steps.

Resources and whats next

- [https://mitsloan.mit.edu/ideas-made-to-matter/machine-learning-and-generative-ai-what-are-they-good-for](https://mitsloan.mit.edu/ideas-made-to-matter/machine-learning-and-generative-ai-what-are-they-good-for)
- [https://www.potterclarkson.com/insights/what-data-is-used-to-train-an-ai-where-does-it-come-from-and-who-owns-it](https://www.potterclarkson.com/insights/what-data-is-used-to-train-an-ai-where-does-it-come-from-and-who-owns-it/)
## Data Analysis and Preprocessing

### Data Preprocessing

Data preprocessing merupakan tahap krusial pada tahap awal dari persiapan data. Sebelum membangun model AI yang efektif, data harus benar-benar siap dan telah melalui serangkaian proses pembersihan serta transformasi.

Pada tahap ini, data mentah diolah menjadi bentuk yang lebih bersih, konsisten, dan terstruktur agar model dapat belajar dengan optimal dan menghasilkan prediksi yang akurat.

“If 80 percent of our work is data preparation, then ensuring data quality is the important work of a machine learning team.”
– Andrew Ng

Proses preprocessing mencakup berbagai aktivitas seperti penanganan missing values, deteksi outlier, normalisasi data, encoding kategorikal, hingga feature engineering. Setiap dataset memiliki karakteristik unik, sehingga teknik yang diterapkan perlu disesuaikan dengan konteks dan tujuan analisis spesifik.

Lorem ipsum lorem ipsum Lorem ipsum lorem ipsum Lorem ipsum lorem ipsum Lorem ipsum lorem ipsum Lorem ipsum lorem ipsum Lorem ipsum lorem ipsum Lorem ipsum lorem ipsum Lorem ipsum lorem ipsum

Use Case Prompt for Analysis Scenario

- “Summarize this dataset for me. What trends or anomalies do you see?”
- “Review this analysis. Are there any missing considerations or biases?”
- “Create visualization ideas for this dataset.”
- “Suggest SQL queries for this scenario.”
- “Explain this complex metric in simple terms.”
- “Draft an executive summary for my findings.”

## Model and Evaluation

### Core Concepts of Machine Learning

Machine learning (ML) adalah bagian dari artificial intelligence (AI) yang memungkinkan komputer belajar dari data untuk membuat prediksi atau keputusan tanpa diprogram secara manual untuk setiap kasus.

Dalam praktiknya, proses training ML melibatkan tiga komponen utama: data, model, dan algoritma. Data digunakan sebagai bahan untuk melatih model agar mampu mengenali pola, sedangkan algoritma bertugas mengoptimalkan parameter model berdasarkan umpan balik dari hasil prediksi. Semakin banyak dan berkualitas data yang digunakan, semakin baik model dalam melakukan generalisasi terhadap data baru.

proses modelling cenderung singkat/cepat, dibandingkan dengan pemrosesan data, proses modelling makin cepat, karena tinggal memanggil library bla bla bla, adapun untuk kasus kasus tertentu seperti data yang besar atau data komputasi yang kompleks bisa nggunain deep learning/neural network

```python
import numpy as np

import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 1. Import dataset built-in

data = fetch_california_housing(as_frame=True)

X = data.data[["AveRooms"]] # Fitur: rata-rata kamar per rumah
y = data.target # Target: harga rumah (dalam $100,000)

# 2. Bagi data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 3. Latih model
model = LinearRegression().fit(X_train, y_train)

# 4. Prediksi untuk rumah dengan rata-rata 120 kamar (ekstrem, untuk demo)
prediksi = model.predict([[1201])
print(prediksi) # Output: sekitar 240000
```

>[!INFO]
>- Regresi ini fondasi dari hampir semua model ML ke depannya. Bahkan cara kerja prediksi pada neural network tetap pakai prinsip yang sama—bedanya cuma di tingkat kompleksitas dan jumlah layer
>- 📚 Baca lebih lanjut: Linear Regression Tutorial - hands-on notebook buat praktik langsung

### Types of Machine Learning


Ada tiga jenis penerapan model ML berdasarkan jenis data:
- Supervised Learning: Belajar dari data berlabel untuk prediksi. Cocok untuk CV (klasifikasi gambar) dan NLP (analisis sentimen teks). Contoh: Model memprediksi penyakit dari data medis berlabel.
- Unsupervised Learning: Menemukan pola tanpa label, seperti mengelompokkan data gambar di CV atau topik teks di NLP. Contoh: Mengelompokkan pelanggan berdasarkan perilaku belanja.
- Reinforcement Learning: Belajar melalui trial-error dengan reward, jarang digunakan di CV/NLP dasar tapi bisa untuk game AI yang melibatkan visi atau bahas

Pemahaman terhadap jenis-jenis machine learning menjadi penting guna memilih teknik yang tepat dalam penerapannya. Scikit-learn merupakan salah satu library Python paling populer yang menyediakan berbagai algoritma dan fungsi siap pakai untuk melatih, menguji, serta mengevaluasi model.

Scikit-learn adalah library Python open source yang banyak digunakan dalam pengembangan ML klasik. Library ini menyediakan beragam algoritma seperti classification, regression, clustering, serta alat bantu untuk model selection dan preprocessing.


>[!INFO]
>💡 Coba https://github.com/andre1araujo/Supervised-and-Unsupervised-Learning-Examples/. Kode ini ngejelasin lebih dalam implementasi ML unsupervised dan supervised sapa tahu mau dibaca lebih lanjut

Sering kali, bagian tersulit dalam menyelesaikan masalah machine learning adalah menentukan algoritma atau estimator yang paling tepat untuk digunakan. Setiap estimator memiliki keunggulan dan keterbatasan tergantung pada jenis data serta tujuan analisis.

| Library                   | Primary Use                         | Key Features                               | Best For                                  |
| ------------------------- | ----------------------------------- | ------------------------------------------ | ----------------------------------------- |
| TensorFlow                | Deep learning model building/deploy | High/low-level APIs, scalability           | Production with unstructured data         |
| PyTorch                   | Flexible DL prototyping             | Dynamic graphs, easy debugging             | Research, NLP/computer vision             |
| Keras                     | Simplified neural networks          | User-friendly interface, community support | Beginners, quick prototyping              |
| Hugging Face Transformers | NLP/LLM tasks                       | Pre-trained models, transfer learning      | Text generation, chatbots                 |
| OpenAI API                | Generative AI integration           | Multimodal capabilities, fine-tuning       | Rapid app development without hosting     |
| Scikit-learn              | Traditional ML                      | Data processing, evaluation tools          | Small datasets, classification/regression |
| XGBoost                   | Gradient boosting                   | High performance, scalability              | Tabular data, competitions                |
| LangChain                 | LLM apps                            | Prompt chaining, memory management         | Complex workflows, agents                 |


Pada penerapannya itu tergantung masalah nyata—jangan pakai yang berat jika sederhana cukup. Start simple, scale when proven.

- Gunakan ML Klasik untuk prediksi tabular, klasifikasi, atau clustering: cepat, ringan, dan mudah dijelaskan. Cocok untuk churn prediction, fraud detection, sales forecasting. Contoh: [Customer Churn](https://github.com/khanhnamle1994/customer-churn-prediction) | [Fraud Detection](https://github.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection)
- Gunakan Deep Learning untuk gambar, audio, atau video: butuh GPU dan data besar, tapi akurasi tinggi. Cocok untuk computer vision, speech recognition, medical imaging. Contoh: [YOLO Object Detection](https://github.com/ultralytics/yolov5) | [Face Recognition](https://github.com/ageitgey/face_recognition)
- Gunakan LLM & NLP untuk teks seperti chatbot atau summarization: mulai pre-trained, fine-tune secukupnya. Cocok untuk customer support, document QA, content generation. Contoh: [ChatBot LangChain](https://github.com/hwchase17/chat-your-data) | [Document QA](https://github.com/deepset-ai/haystack)

### Model Training and Validation data

Terima kasih kepada Python dan library seperti scikit-learn, sekarang nerapin model ML tinggal beberapa baris kode. Fokus kita lebih ke penyesuaian parameter yang tepat untuk model dan memahami bagaimana data dibagi dengan benar. Pemilihan parameter dan strategi validation inilah yang menentukan apakah model kita benar-benar reliable atau hanya bagus di data training.

```python
# ... import model scikit-learn
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)
```

Dalam praktiknya, data dibagi menjadi tiga bagian dengan fungsi yang berbeda

1. Training Dataset
Data yang digunakan untuk melatih model—di sinilah model belajar mengenali pola.

2. Validation Dataset:
Data yang digunakan untuk evaluasi model selama proses tuning hyperparameter. Dataset ini membantu kita menyesuaikan konfigurasi model tanpa "menyentuh" test data, sehingga evaluasi tetap objektif.

3. Test Dataset:
Data yang digunakan untuk evaluasi final model. Dataset ini sama sekali tidak digunakan selama training atau tuning, sehingga memberikan gambaran performa model yang paling objektif.

>[!INFO]
>💡 Coba https://github.com/andre1araujo/Supervised-and-Unsupervised-Learning-Examples/. Kode ini ngejelasin lebih dalam implementasi ML unsupervised dan supervised sapa tahu mau dibaca lebih lanjut

==perlu satu halaman lagi==

Pemilihan strategi validation bergantung pada tiga hal: ukuran dataset, tipe problem, dan resource komputasi yang tersedia. Dataset besar bisa pakai holdout untuk efisiensi, dataset kecil butuh K-Fold atau bahkan LOOCV untuk maksimalkan data usage. Kalau datanya imbalanced, stratified approach jadi penting untuk jaga distribusi class. Berikut perbandingan dalam tabel:

| Strategy                | When to Use                     | Pros                   | Cons                               |
| ----------------------- | ------------------------------- | ---------------------- | ---------------------------------- |
| Holdout                 | Quick baseline, large datasets  | Fast, simple           | High variance, data waste          |
| K-Fold Cross-Validation | Most general supervised tasks   | Robust, efficient      | Slower than holdout                |
| Stratified K-Fold       | Imbalanced classification tasks | Preserves class ratios | Slightly more complex              |
| LOOCV                   | Small datasets, high precision  | Maximum data use       | Very slow, overkill for large sets |

### Hyperparameter tuning

Dalam pengembangan model machine learning, terdapat dua jenis parameter yang perlu dipahami dengan baik:

Model Parameters (Weights): Parameter yang dipelajari oleh model secara otomatis selama proses training, seperti koefisien dalam regresi linear atau weight dalam neural network.

Hyperparameters: Parameter yang tidak dipelajari oleh model tetapi harus diatur secara manual sebelum training dimulai. Hyperparameter memiliki pengaruh signifikan terhadap performa model dan bagaimana model belajar dari data.

Hyperparameter tuning adalah proses sistematis mencari nilai-nilai optimal untuk hyperparameter agar model mencapai performa terbaik. Proses ini merupakan langkah krusial dalam mengembangkan model machine learning yang efektif dan dapat diandalkan.

==perlu quotes==

### Ensemble and Stacking

Kalau kamu sering eksplor kompetisi Kaggle, kamu pasti notice bahwa top performers di leaderboard sering bukan pakai deep learning—mereka pakai ensemble methods. Teknik ini consistently menghasilkan akurasi tinggi dan sering jadi game changer dalam kompetisi.

Tapi seberapa powerful sebenarnya ensemble ini? Surprisingly, konsepnya cukup straightforward: kita combine beberapa model untuk menghasilkan prediksi yang lebih baik. Ide dasarnya adalah "wisdom of crowds"—beberapa model yang decent, kalau digabung dengan cara yang tepat, bisa menghasilkan performa yang jauh lebih baik dibanding satu model terbaik sekalipun.

Prosesnya begini: setelah kita train beberapa model dengan performa yang bagus (bisa berbeda algoritma atau hyperparameter), kita combine prediksi mereka. Yang sering terjadi adalah hasil ensemble ini lebih robust dan akurat dibanding individual models.

==jenis-jenis ensemble dalam gambar==


Great Books on Everything Data and Machine Learning

lorem ipsum lorem ipsum

<!--makan nasi-->