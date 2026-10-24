# Trading Chaos AI

Проект для live-сбора Brent H1 из MetaTrader 5, подготовки данных, запуска торгового контура обычного бектеста без ML и отправки уведомлений/команд через Telegram.

Сейчас рабочий контур выглядит так:

```text
MetaTrader 5
-> input/data/BRENT_H1.csv
-> input/Cleaner.py
-> input/clean_df/BRENT_H1_clean.csv
-> src/basic_backtest
-> заявка в MetaTrader 5
-> Telegram
```

## Структура

- `input` - входные данные, запись свечей из MT5 и очистка датасета.
- `docs` - документация по структуре проекта и MT5-потоку.
- `src/basic_backtest` - обычный бектест и live-контур без ML.
- `src/ml_backtest` - зона для бектеста с ML-фильтром.
- `src/opt_backtest` - зона для ML-фильтра с оптимизацией.
- `output` - Telegram-уведомления и Telegram-бот для управления торговлей.
- `test/basic_backtest` - тесты для базового live-контура.

## Установка

Рекомендуемый запуск для торговли через MT5 - Windows с установленным MetaTrader 5.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
set PYTHONPATH=src;.
```

На macOS/Linux можно запускать тесты и проверять код, но пакет `MetaTrader5` и реальное подключение к терминалу работают на Windows.

## Запись Brent H1

Свечи пишутся в `input/data/BRENT_H1.csv`.

```bash
set MT5_LOGIN=123456
set MT5_PASSWORD=your_password
set MT5_SERVER=Broker-Server
set MT5_PATH=C:\Program Files\MetaTrader 5\terminal64.exe

python -m input.record_brent_h1 --symbol BRENT
```

Если у брокера другой тикер Brent, замени только `--symbol`, например `BRN` или `XBRUSD`.

## Live-контур basic_backtest

Проверка одного шага без реальных заявок:

```bash
python -m basic_backtest.live_brent_h1_pipeline --symbol BRENT --volume 1 --once --dry-run
```

Запуск постоянного контура:

```bash
python -m basic_backtest.live_brent_h1_pipeline --symbol BRENT --volume 1
```

Контур добавляет новые закрытые H1-свечи, прогоняет их через `input/Cleaner.py`, проверяет базовый сигнал и при наличии сигнала выставляет stop-entry заявку.

## Telegram

Для уведомлений о заявках и сделках:

```bash
set TELEGRAM_BOT_TOKEN=123456:token
set TELEGRAM_CHAT_ID=123456789
```

Для управления торговлей из Telegram запусти:

```bash
set TELEGRAM_BOT_TOKEN=123456:token
set PYTHONPATH=src;.
python -m output.telegram_trading_bot
```

Сначала отправь боту:

```text
/whoami
```

Потом задай разрешенный chat id и перезапусти:

```bash
set TELEGRAM_ALLOWED_CHAT_ID=123456789
set TELEGRAM_CHAT_ID=123456789
python -m output.telegram_trading_bot
```

Основные команды:

```text
/connect <login> <server> <password>
/set_path <путь к terminal64.exe>
/set_symbol BRENT
/set_volume 1
/set_risk 0.02
/test_mt5
/start_basic_dry
/start_basic_live
/stop_basic
/status
```

Реальные заявки дополнительно заблокированы защитой. Чтобы разрешить `/start_basic_live`, на Windows-машине нужно явно задать:

```bash
set TELEGRAM_ALLOW_LIVE_TRADING=1
```

Сначала проверяй запуск через `/start_basic_dry`.

## Проверка

```bash
PYTHONPATH=src:. .venv/bin/python -m pytest -q test/basic_backtest
```

На Windows с активированным `.venv`:

```bash
set PYTHONPATH=src;.
python -m pytest -q test/basic_backtest
```

## Документация

- `docs/project_lanes.md` - разделение проекта на обычный бектест, ML-бектест и оптимизацию.
- `docs/mt5_data_recorder.md` - запись Brent H1 из MetaTrader 5.
- `output/README.md` - Telegram-уведомления и управление торговлей.

## MLOps: веб-приложение (ДЗ 2)

Веб-приложение лежит в `trading_chaos_api`. Оно запускается отдельно от торгового контура и использует PostgreSQL для проверки доступности внешнего компонента. Зависимости приложения вынесены в `pyproject.toml`; версии для сборки образа закреплены в `requirements-api.lock`.

Для локального запуска нужны Docker и Docker Compose:

```bash
cp .env.example .env
# Замени POSTGRES_PASSWORD в .env
docker compose up --build
```

После запуска:

- `GET http://localhost:8000/healthz` — быстрая проверка приложения без обращения к БД;
- `GET http://localhost:8000/api/v1/version` — версия приложения и ревизия сборки;
- `GET http://localhost:8000/api/v1/health` — версия PostgreSQL и время ответа. При ошибке БД возвращается 503.

Для разработки без Docker:

```bash
python -m venv .venv-api
.venv-api/bin/pip install -e '.[dev]'
.venv-api/bin/pre-commit install
.venv-api/bin/ruff check trading_chaos_api test/mlops
.venv-api/bin/ruff format --check trading_chaos_api test/mlops
.venv-api/bin/pytest -q test/mlops --cov=trading_chaos_api
```

CI запускает линтер, проверку форматирования и тесты с покрытием. После успешного push в `hw2` или `main` CD собирает образ и публикует его в GitHub Container Registry с тегом `sha-<12 символов коммита>`. Используется GHCR, потому что он доступен из GitHub Actions через `GITHUB_TOKEN`, без отдельного ключа реестра. Лог сборки и адрес опубликованного образа видны в Actions → API → publish.
