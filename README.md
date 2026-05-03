# SignBridge - Traducător Live pentru Limbajul Semnelor

SignBridge este un proiect de Computer Vision și Deep Learning care traduce în timp real gesturile mâinii în litere din alfabetul limbajului semnelor. Aplicația folosește MediaPipe pentru extragerea celor 21 de landmark-uri ale mâinii, TensorFlow/Keras pentru clasificare și OpenCV pentru captură video și afișare live.

## Tehnologii

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-0F9D58?style=for-the-badge)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Preprocessing-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)

- Python
- MediaPipe Hands
- TensorFlow / Keras
- OpenCV
- scikit-learn
- NumPy

## Ce face proiectul

- Parcurge imaginile din dataset și extrage automat coordonatele relevante ale mâinii.
- Normalizează landmark-urile relativ la încheietură pentru consistență între train și inference.
- Antrenează un model neural pentru clasificarea literelor.
- Rulează recunoașterea live prin webcam, cu smoothing temporal pentru predicții mai stabile.

## Instalare

1. Clonează repository-ul:

```bash
git clone <URL_REPOSITORY>
cd SignLanguage
```

2. Creează și activează un mediu virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Instalează dependențele:

```bash
pip install -r requirements.txt
```

4. Adaugă dataset-ul local în structura:

```text
data/raw/semne/
  Train/
    A/ ... Y/
  Test/
    A/ ... Y/
```

## Mod de utilizare

Ordinea corectă de rulare este:

1. Extracția landmark-urilor din dataset:

```bash
python3 src/extract_landmarks.py
```

2. Antrenarea modelului:

```bash
python3 src/train_model.py
```

3. Recunoașterea live din webcam:

```bash
python3 src/live_recognition.py
```

Aplicația live se închide cu tasta `q`.

## Highlights Tehnice

- Arhitectura folosește modulul comun `sign_language_utils.py` pentru aceeași logică de normalizare în train și live inference.
- Această abordare reduce riscul de data drift între datele preprocesate și cadrele capturate în timp real.
- Pentru hardware limitat, inferența live folosește `model_complexity=0` în MediaPipe.
- Cadrul video este oglindit pentru o experiență naturală, iar detecția se face după `cv2.flip(...)`, astfel încât AI-ul procesează exact imaginea afișată utilizatorului.
- Predicțiile sunt stabilizate prin buffer și majoritate statistică, pentru a reduce flicker-ul între litere.
- Resursele sunt gestionate corect prin `with` pentru fișiere și `cap.release()` / `cv2.destroyAllWindows()` pentru webcam.

## Structura Proiectului

```text
SignLanguage/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   │   └── semne/
│   └── processed/
│       ├── train_data.pickle
│       └── test_data.pickle
├── models/
│   └── sign_language_model.h5
├── src/
│   ├── extract_landmarks.py
│   ├── train_model.py
│   ├── live_recognition.py
│   └── sign_language_utils.py
└── notebooks/
```

Această separare face clară diferența dintre codul sursă, datele brute, datele procesate și modelele antrenate.

## Autor

**Numele Tău**

- LinkedIn: `https://www.linkedin.com/in/linkul-tau/`
