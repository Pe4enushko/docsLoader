# Как работает приложение

## Коротко
Система проверяет качество заполнения медицинских приёмов на основе клинических рекомендаций в knowledge graph.

## Поток
1. `evaluateVerdict.py` делает HTTP-запрос в 1C и получает массив `appointments`.
2. Для каждого приёма извлекаются МКБ-коды из диагноза (если есть).
3. Выполняется базовая LLM-оценка корректности заполнения.
4. Если МКБ найден и сопоставлен в `manifest.csv`, выполняется дополнительная проверка через KG по `doc_id`.
5. Результаты объединяются, формируется `human_readable` (raw JSON приёма), и запись сохраняется в Postgres `MedKard`.

## Что хранится в MedKard
- score-поля по секциям
- `score_overall`
- `risk_level` (юридический риск)
- `issues`, `summary`
- raw JSON-поля: `Inspection_data`, `diagnosis_data`, `services_data`, `patient`
- `visit_guid_1c`, `visit_date`, `human_readable`

## Подготовка
- `init_knowledge_graph.py` — загрузка рекомендаций в KG.
- `init_medkard_table.py` — создание таблицы в Postgres.
