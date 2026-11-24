# Reproducible Experiment Environment: Sexism Detection in Social Media

Відтворюване обчислювальне середовище для дослідження методів виявлення гендерної дискримінації (сексизму) у текстовому контенті соціальних медіа (Хмельницький національний університет, ФІТ, кафедра комп'ютерних наук).

## Опис експерименту та артефактів

- **Датасет:** очищений аугментований корпус на основі SDET та SSMB (18 409 унікальних записів: Train — 14 727, Validation — 3 682; зовнішній Test — 4 000 записів). Контроль версій даних реалізовано через **DVC**.
- **Досліджувані конфігурації (`SEED = 42`):**
  - `baseline_M0`: `TF-IDF + LogisticRegression(C=1.0)` — Validation $F_1 = 0.7407$, External Test $F_1 = 0.5796$.
  - `optimized_M1`: `TF-IDF + SelectKBest(chi2, k=2000) + LogisticRegression(C=10.0)` — Validation $F_1 = 0.7646$, External Test $F_1 = 0.5972$.
- **Стек відтворюваності:** Git, DVC, Docker, Docker Compose, MLflow, Zenodo (`zenodo.json`, ліцензія MIT).

## Структура репозиторію

    ├── data/                   # DVC-вказівники датасетів (*.dvc) та .gitignore
    ├── Dockerfile              # Опис ізольованого образу (Python 3.10-slim)
    ├── docker-compose.yml      # Оркестрація контейнера експерименту та MLflow UI
    ├── requirements.txt        # Зафіксовані версії Python-бібліотек
    ├── train.py                # Пайплайн підготовки даних, навчання та трекінгу MLflow
    ├── zenodo.json             # Машинозчитувані наукові метадані (FAIR Data)
    └── README.md               # Інструкція відтворення експерименту

## Покрокова інструкція відтворення результатів

### 1. Клонування репозиторію

    git clone https://github.com/Twinker25/sexism-detection-reproducible.git
    cd sexism-detection-reproducible

### 2. Підготовка набору даних (DVC)

За наявності налаштованого віддаленого сховища DVC завантажте дані командою:

    dvc pull

Або розмістіть очищені файли датасету (`train.csv`, `dev.csv`, `test.csv`) у директорії `data/`.

### 3. Запуск експерименту та сервера MLflow UI в один клік (Docker Compose)

    docker compose up --build

Після виконання контейнера експерименту веб-інтерфейс моніторингу результатів **MLflow UI** доступний у браузері за адресою: `http://localhost:5000`.

### 4. Альтернативний запуск через окремий Docker-контейнер

Збірка образу:

    docker build -t cs-experiment:v1.0 .

Запуск обчислень (Linux / macOS / Git Bash):

    docker run --rm -v "$(pwd)/data:/app/data:ro" -v "$(pwd)/mlruns:/app/mlruns" cs-experiment:v1.0

Запуск обчислень (Windows PowerShell):

    docker run --rm -v "${PWD}/data:/app/data:ro" -v "${PWD}/mlruns:/app/mlruns" cs-experiment:v1.0

Запуск локального веб-інтерфейсу MLflow:

    mlflow ui --port 5000