# GPTHub

Единый мультимодальный чат поверх: текст, голос, файлы, изображения, поиск, проекты и память в одном интерфейсе.

## Запуск

```bash
cp .env.example .env
```

После этого откройте `.env` и заполните:

```env
MWS_GPT_API_KEY=
```

Далее:

```bash
docker compose build --no-cache
docker compose up
```

## Вход

- Основной вход: [http://localhost](http://localhost)
- Логин администратора: `admin@gpthub.local`
- Пароль администратора: `change_me_admin`

## Порты

- [http://localhost:3000](http://localhost:3000) — frontend напрямую
- [http://localhost:8080](http://localhost:8080) — Traefik dashboard, роуты и сервисы
- [http://localhost:3001](http://localhost:3001) — Langfuse, трассировка запросов и наблюдаемость
- [http://localhost:5555](http://localhost:5555) — Flower, фоновые задачи и Celery workers
- [http://localhost:9001](http://localhost:9001) — MinIO Console, файлы и артефакты
- [http://localhost:15672](http://localhost:15672) — RabbitMQ Management, очереди и брокер
