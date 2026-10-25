# ДЗ 3. MLflow

В этой части проекта мы используем дневные цены Brent и проверяем, можно ли предсказать направление следующего дневного закрытия по предыдущей истории цен. Задача бинарная: `1` означает, что следующее закрытие выше текущего, `0` — что оно ниже или равно текущему.

## Что находится в проекте

- `docker-compose.yml` поднимает PostgreSQL, MinIO и MLflow Tracking Server.
- PostgreSQL используется как Backend Store для экспериментов, метрик, параметров и Model Registry.
- MinIO используется как S3-совместимое хранилище артефактов.
- `app/eda.py` строит графики и сохраняет отчёт о качестве данных.
- `app/train.py` запускает EDA, baseline и две модели, логирует метрики и артефакты, регистрирует модели и назначает alias `champion`.
- `app/main.py` содержит асинхронный API. Модель загружается один раз при старте приложения.
- `tests/` содержит unit-тесты для подготовки данных и API.
- `data/brent_prices.csv` — датасет, который используется для запуска проекта без дополнительных скачиваний.

## Данные

В CSV 1 840 исходных строк за период с 2017-11-02 по 2025-04-03. При подготовке данные сортируются по дате, а дубликаты дат удаляются с сохранением последней строки. Итоговое число строк после очистки — 1 837.

У датасета есть ограничения: примерно у 86% строк значения Open, High, Low и Close совпадают, а примерно у 84% строк объём равен нулю. Поэтому объём не используется как признак, а модель строится на лагах доходности цены закрытия, скользящей волатильности и отклонении цены от скользящего среднего. Эти особенности также записываются в `data_quality.json` в артефактах EDA. Результаты эксперимента не следует трактовать как готовую торговую стратегию.

Источник, указанный для файла: [Historical Crude Oil Futures Prices (WTI & Brent)](https://www.kaggle.com/code/nikitamanaenkov/financial-performance-analysis-brent-and-wti/input). Перед публикацией датасета в открытом репозитории стоит проверить условия его лицензии.

## Запуск

Нужны Docker Desktop с Docker Compose и Python 3.11 для локального запуска тестов.

Перейди в папку `mlflow_hw3`. Для первого запуска создай `.env`.

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

macOS/Linux:

```bash
cp .env.example .env
```

Затем подними Tracking Server и хранилища:

```bash
docker compose up -d --build
```

Открой:

- MLflow UI: http://localhost:5000
- MinIO Console: http://localhost:9003

Локальные учётные данные MinIO находятся в `.env.example`. Они предназначены только для учебного запуска.

### EDA и обучение

```bash
docker compose --profile train run --rm trainer
```

После завершения в эксперименте `brent-direction` должны быть видны Runs `eda`, `baseline_most_frequent`, `logistic_regression`, `random_forest` и `model-comparison`. В Runs доступны параметры, метрики и диагностические артефакты. В Model Registry появится `BrentDirectionModel` с версиями зарегистрированных моделей и alias `champion`.

Основная метрика — balanced accuracy: она учитывает качество распознавания обоих классов и не позволяет baseline, который всегда предсказывает один класс, выглядеть лучше только за счёт F1. Дополнительно записываются F1 для класса роста, accuracy, precision, recall и ROC-AUC. Данные делятся по времени в пропорции примерно 80/20. Между обучающей и тестовой частями оставляется одна строка, чтобы целевая цена последней обучающей записи не пересекала границу тестового периода.

### Запуск inference

После успешного обучения выполни:

```bash
docker compose --profile inference up -d --build inference
```

Документация API: http://localhost:8000/docs

Проверка загруженной версии модели:

```bash
curl http://localhost:8000/model
```

Проверка состояния приложения:

```bash
curl http://localhost:8000/healthz
```

`POST /process` принимает историю минимум из 11 последовательных свечей. Пример тела запроса:

```json
{
  "candles": [
    {"date": "2024-01-01", "open": 80.0, "high": 80.3, "low": 79.7, "close": 80.0, "volume": 100},
    {"date": "2024-01-02", "open": 80.0, "high": 80.7, "low": 79.8, "close": 80.4, "volume": 100},
    {"date": "2024-01-03", "open": 80.4, "high": 80.6, "low": 79.9, "close": 80.1, "volume": 100},
    {"date": "2024-01-04", "open": 80.1, "high": 81.0, "low": 80.0, "close": 80.8, "volume": 100},
    {"date": "2024-01-05", "open": 80.8, "high": 81.2, "low": 80.7, "close": 81.0, "volume": 100},
    {"date": "2024-01-06", "open": 81.0, "high": 81.1, "low": 80.5, "close": 80.7, "volume": 100},
    {"date": "2024-01-07", "open": 80.7, "high": 81.4, "low": 80.6, "close": 81.2, "volume": 100},
    {"date": "2024-01-08", "open": 81.2, "high": 81.6, "low": 81.0, "close": 81.4, "volume": 100},
    {"date": "2024-01-09", "open": 81.4, "high": 81.5, "low": 80.9, "close": 81.1, "volume": 100},
    {"date": "2024-01-10", "open": 81.1, "high": 81.8, "low": 81.0, "close": 81.6, "volume": 100},
    {"date": "2024-01-11", "open": 81.6, "high": 82.2, "low": 81.5, "close": 82.0, "volume": 100}
  ]
}
```

В ответе будут предсказанное направление, вероятность роста (если модель поддерживает `predict_proba`), имя и версия модели, alias, Run ID и число переданных свечей. Даты можно указывать у всех свечей или не указывать ни у одной.

## Model Registry и alias

Обе обучаемые модели регистрируются в `BrentDirectionModel` как отдельные версии. Alias `champion` назначается модели с наибольшей balanced accuracy; F1 используется как дополнительный критерий при равенстве основной метрики. Повторный запуск обучения создаёт новые версии, а не перезаписывает предыдущие.

Чтобы переключить alias на существующую версию, замени `1` на номер версии из Model Registry:

```bash
docker compose --profile train run --rm trainer python -c "from mlflow import MlflowClient; client = MlflowClient(tracking_uri='http://mlflow:5000'); client.set_registered_model_alias('BrentDirectionModel', 'champion', '1')"
```

Затем перезапусти inference:

```bash
docker compose --profile inference restart inference
```

Приложение читает alias при старте и загружает конкретную версию модели в память. Изменение alias в MLflow не заменяет модель в уже работающем процессе. В продакшене лучше загрузить и проверить новую модель отдельно, а затем атомарно переключить ссылку на модель или развернуть новый экземпляр и переключить на него трафик после проверки готовности.

## Dataset Tracking: source, digest и lineage

- `source` — путь к CSV, который использовался при подготовке данных.
- `digest` — вычисляемый MLflow отпечаток содержимого датасета; он помогает отличать версии данных.
- `lineage` — связь между датасетом и Runs, которые его использовали. Она видна в карточке Run в разделе входных данных.

В проекте отслеживаются исходный очищенный датасет `brent-daily-prices` и таблица признаков с целевой переменной `brent-daily-features`. Оба связаны с экспериментами. В Runs также сохранён URL страницы источника данных.

## Тесты и проверки

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
ruff check app tests
ruff format --check app tests
pytest -q --cov=app --cov-report=term-missing
```

macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
ruff check app tests
ruff format --check app tests
pytest -q --cov=app --cov-report=term-missing
```

GitHub Actions запускает линтер, проверку форматирования, unit-тесты, проверку конфигурации Compose и интеграционный сценарий с реальным MLflow, PostgreSQL, MinIO, обучением и `POST /process`.

## Остановка

```bash
docker compose down
```

Данные PostgreSQL и MinIO сохраняются в Docker volumes. Для удаления хранилищ вместе с их содержимым используй `docker compose down -v` только если они больше не нужны.
