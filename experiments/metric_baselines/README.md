# Сравнение обучающих сигналов на Qilin

В раннем пилоте пять моделей `CatBoostRegressor` переставляют одни и те же посты с исходных позиций 1–10. У моделей одинаковые лексические признаки запроса и текста поста, настройки обучения и выборки. Отличается только обучающая метка. Короткие выдачи остаются в выборке. Порядок Qilin служит исходным бейзлайном.

| Модель | Обучающая метка |
| --- | --- |
| `click` | Индикатор клика |
| `skip_penalty` | Клик, а для некликнутого поста выше клика штраф `−1/log₂(1 + позиция)` |
| `long_view` | Клик и `page_time` от 10 секунд |
| `dwell_log` | Для клика `log(1 + min(page_time, 60))/log(61)` при известном неотрицательном времени, иначе 0 |
| `collect` | Индикатор сохранения поста |

Считаем ClickNDCG@10 и LongViewNDCG@10 по всем запросам validation, включая нулевые значения у запросов без целевого действия. CollectNDCG@10 считаем на запросах, где хотя бы один кандидат был сохранён. Парные 95%-е интервалы для разностей получаем бутстрэпом по пользователям. Реакции и исходная позиция служат для построения меток и оценки. В модель подаются только текстовые признаки и тип поста. Выдачи с одинаковыми оценками модели сохраняют исходный порядок. В официальном test используем только идентификаторы для исключения пересекающихся сессий.

## Запуск

Нужен Python 3.12. Выполните из корня репозитория:

```bash
cd experiments/metric_baselines
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Скачайте файлы из [зафиксированной версии Qilin](https://huggingface.co/datasets/THUIR/Qilin/tree/2d79000cef450b99f6423357ce86e5c132d32dd5). Общий объём parquet около 1,1 ГБ. Команды выполняются из `experiments/metric_baselines`.

```bash
qilin_revision=2d79000cef450b99f6423357ce86e5c132d32dd5
qilin_source="https://huggingface.co/datasets/THUIR/Qilin/resolve/$qilin_revision"
for split in search_train search_test; do
  mkdir -p "data/qilin/$split"
  path="$split/train-00000-of-00001.parquet"
  curl --fail --location --retry 3 "$qilin_source/$path" --output "data/qilin/$path"
done
mkdir -p data/qilin/notes
for shard in 00000 00001 00002 00003 00004; do
  path="notes/train-$shard-of-00005.parquet"
  curl --fail --location --retry 3 "$qilin_source/$path" --output "data/qilin/$path"
done
```

Из официального `search_test` выгрузите только три колонки с идентификаторами. Они нужны для исключения пересекающихся сессий из `search_train`. Файлы в `data/` и результаты в `results/` храните локально.

```bash
.venv/bin/python - "$qilin_revision" <<'PY'
import json
import sys
from pathlib import Path

from pyarrow.parquet import ParquetFile

test = ParquetFile('data/qilin/search_test/train-00000-of-00001.parquet')
rows = test.read(columns=['search_idx', 'user_idx', 'session_idx']).to_pylist()
out = Path('data/qilin/test_ids.json')
out.write_text(json.dumps({'revision': sys.argv[1], 'rows': rows}), encoding='utf-8')
PY

train=data/qilin/search_train/train-00000-of-00001.parquet
ids=data/qilin/test_ids.json
```

## Воспроизведение основных сравнений

Запустите четыре скрипта в этом порядке. В первом определяется рецепт исходного композита. Во втором строятся позиционные кривые, которые понадобятся двум следующим прогонам.

```bash
.venv/bin/python run_engagement.py --train "$train" --test-identifiers "$ids" \
  --notes data/qilin/notes/train-*-of-00005.parquet --output results/engagement
.venv/bin/python position_avatar_fit.py --train "$train" --test-identifiers "$ids" \
  --output results/avatar/calibration
.venv/bin/python run_avatar_rerank.py --train "$train" --test-identifiers "$ids" \
  --notes data/qilin/notes/train-*-of-00005.parquet \
  --curves results/avatar/calibration/curves.json \
  --engagement-summary results/engagement/summary.json \
  --output results/avatar/evaluation
.venv/bin/python run_position_free.py --train "$train" --test-identifiers "$ids" \
  --notes data/qilin/notes/train-*-of-00005.parquet \
  --curves results/avatar/calibration/curves.json --output results/position_free
```

При глубине кандидатов 30 общая выборка validation содержит 6 811 запросов и 91 011 пар «запрос — пост». У 610 запросов есть хотя бы один лайк. Все основные оценки ниже сделаны на этом validation.

| Сравнение | Где искать результат |
| --- | --- |
| Исходный порядок и CatBoost на кликах с признаком исходной позиции | `results/engagement/summary.csv`, модели `qilin` и `point_click` |
| CatBoost на композите с сырыми реакциями, исходной и обновлённой позиционными кривыми | `results/avatar/evaluation/summary.csv` и `weighted_dcg.csv`, модели `point_composite`, `point_composite_original`, `point_composite_avatar`. Для взвешенных оценок сравнивайте строки с одной `evaluation_curve` |
| PairLogit без позиции во входах, обычные пары и усиление пар с вероятным пропуском | `results/position_free/comparisons.csv`, модели `blind_pair_control` и `blind_pair_skip` |
| Смешивание независимого скора с порядком Qilin или CatBoost на кликах | `results/position_free/comparisons.csv`, суффиксы `_fixed` и `_tuned` для порядка Qilin, `_on_catboost` для порядка CatBoost. Выбранные на внутренней части train коэффициенты есть в `summary.json` и `calibration.csv` |

`LikeNDCG@10` включает все 6 811 запросов, а `LikePositiveNDCG@10` усредняется по 610 запросам с лайком. Взвешенный DCG представлен в собственной шкале и считается по всем запросам.

## Ранний пилот на кандидатах из топ-10

```bash
.venv/bin/python run.py \
  --train data/qilin/search_train/train-00000-of-00001.parquet \
  --test-identifiers "$ids" \
  --notes data/qilin/notes/train-*-of-00005.parquet \
  --output results/metric_pilot
```

Программа запишет агрегированные результаты в `summary.json` и `summary.csv`, признаки и версии окружения в `run.json`. Для проверки запуска используйте `--smoke` и отдельный каталог результатов. В полном прогоне должно быть 27 963 запроса для обучения и 6 811 для validation.

## Глубина выдачи и позиционная поправка

```bash
.venv/bin/python run_depths.py --train "$train" --test-identifiers "$ids" \
  --notes data/qilin/notes/train-*-of-00005.parquet --output results/depths
.venv/bin/python position_audit.py --train "$train" --test-identifiers "$ids" \
  --output results/position_audit
.venv/bin/python position_sensitivity.py --train "$train" --test-identifiers "$ids" \
  --notes data/qilin/notes/train-*-of-00005.parquet --output results/position_sensitivity.json
```

В `run_depths.py` все модели оцениваются на одних 6 811 запросах. Для сравнения глубин используйте строки `fixed_deepest_slate` с общими метками и идеальным DCG по топ-30. Строки `fixed_deepest_model` показывают, что меняется при расширении пула без нового обучения. Строки `within_depth` подходят для сравнения моделей внутри одной глубины.

`position_audit.py` сравнивает CTR по позициям и по повторяющимся парам «запрос — пост» в train. `position_sensitivity.py` проверяет, как меняется результат, если в обучении и оценке дать клику на исходной позиции `p` вес `min(3, p^0.25)`. Обычный ClickNDCG считается отдельно. Убывающая вероятность просмотра `p^−0.25` принята для проверки чувствительности. Из сопоставления повторных показов нельзя выделить эффект позиции независимо от остальных условий показа.

Для диагностики разрыва с исходной выдачей запустите ту же модель на кликах с текстовыми признаками, с одной позицией и с обоими видами признаков. Скрипт сравнит две реализации ClickNDCG@10 и измерит вклад перестановки первой десятки и допуска кандидатов с более низких мест.

```bash
.venv/bin/python diagnose_gap.py --train "$train" --test-identifiers "$ids" \
  --notes data/qilin/notes/train-*-of-00005.parquet --output results/gap_diagnosis.json
```

## Реакции пользователей с учётом исходной позиции

`run_engagement.py` фиксирует глубину кандидатов 30 и добавляет исходную позицию Qilin ко всем обучаемым моделям. Он сравнивает отдельные сигналы `click`, просмотр от 10 секунд, длительность, `like`, `collect`, любое активное действие и штраф за пропуск выше клика. Поля скролла в логах нет. Просмотр логарифмируем и ограничиваем 60 секундами. Активное действие означает лайк, сохранение, комментарий или отправку.

Составную метку выбираем из трёх рецептов по LikeNDCG на поздней внутренней части train. Сессии на границе убираем. После выбора обучаем модели на полном train и сравниваем с исходным порядком на общей validation. Основная оценка LikeNDCG@10 включает все запросы, с нулём для запросов без лайков. Отдельно выводится LikeNDCG по запросам с лайками. Для оценки архитектуры обучения сравниваем CatBoost RMSE, парный CatBoost PairLogit и линейную Ridge на кликах и на выбранной составной метке.

Рецепты, ход выбора, метрики, парные интервалы и время обучения с предсказанием записываются в `summary.json`. Сравнение метрик доступно в `summary.csv`.

## Калибровка веса исходной позиции

`position_head.py` проверяет идею оценить внимание по CTR частых запросов. `position_fit.py` отдельно для кликов и лайков подбирает кривую `e(p) = p^(−α)` по повторным показам одной пары «нормализованный запрос — пост» на разных местах в train. Вариант со сравнением внутри пользователя проверяет чувствительность. Это оценки по наблюдательным логам, где исходное размещение зависит от ранжирования.

`position_eval.py` повторно получает скоры тех же двух моделей на validation. В поисковом DCG историческая реакция получает обратный вес `1/e(p)` по исходной позиции `p`. Логарифмический дисконт зависит от нового места `r`, а вклад за пределом первых десяти мест равен нулю. Прогон сравнивает обычный DCG, взвешенный DCG и вариант с ограничением обратного веса на уровне 3. Интервалы по пользователям условны на выбранной кривой. Этот прогон оценивает прежние модели и сохраняет их цели обучения.

```bash
.venv/bin/python position_head.py --train "$train" --test-identifiers "$ids" \
  --output results/position_calibration/head
.venv/bin/python position_fit.py --train "$train" --test-identifiers "$ids" \
  --output results/position_calibration/fit
.venv/bin/python position_eval.py --train "$train" --test-identifiers "$ids" \
  --notes data/qilin/notes/train-*-of-00005.parquet \
  --curves results/position_calibration/fit/curves.json \
  --engagement-summary results/engagement/summary.json \
  --output results/position_calibration/eval
```

## Переобучение с учётом переформулировок

`position_avatar_fit.py` заново оценивает гладкую кривую для клика и лайка на том же train. В запросах на аватарки заменяем эквивалентные формулировки одним названием интента. Так можно сравнить показы одного и того же поста по разным формулировкам. Входные показы используются один раз, а сравнение по-прежнему опирается на посты, встречавшиеся на разных исходных местах.

`run_avatar_rerank.py` сравнивает обучение на сырых реакциях, на метках с исходной кривой и на метках с обновлённой кривой. Для каждого клика и лайка берём обратный вес `min(1/e(p), 3)` по его исходной позиции. В составной метке корректируем только слагаемые клика и лайка, для которых оценены свои кривые. Время просмотра, сохранения, репосты и комментарии идут в неё с прежними весами. Кандидаты, признаки и validation общие для всех моделей. Взвешенный DCG оценивает все модели по одной кривой в каждом сценарии оценки.

В `calibration/curves.json` находятся обе кривые, состав аватарных групп и число сопоставимых показов. В `evaluation/summary.csv` находится обычный NDCG, в `evaluation/weighted_dcg.csv` — позиционно-взвешенный DCG, в `evaluation/paired_changes.csv` — сравнение моделей на одной цели обучения с парным 95% интервалом по пользователям. При совпадении меток используем один обученный вариант. Официальный test задействован только для исключения пересечений сессий по идентификаторам.

Для проверки чувствительности калибровали кривую клика только на аватарных запросах. В `point_avatar_only_click` этой кривой взвешены клики, для лайков сохранена общая кривая. В `comparisons.csv` все модели, включая Qilin, оцениваются по одной общей кривой.

```bash
.venv/bin/python run_avatar_only_click.py --train "$train" --test-identifiers "$ids" \
  --notes data/qilin/notes/train-*-of-00005.parquet \
  --curves results/avatar/calibration/curves.json --output results/avatar/only-avatar-click
```

## Мягкий бонус, пропуски и сходный повторный запрос

`run_reward_ablation.py` сравнивает сырой композит, полную позиционную поправку, четверть бонуса за глубокие клики и тот же мягкий бонус только для кликов с просмотром от 10 секунд или сохранением. Для сигнала перезапроса добавляем штраф в четверть базового веса клика, если короткий просмотр без активных действий предшествует похожему поиску в той же сессии. Сходство считаем по символьным биграммам. Следующий запрос участвует в метке на train.

Отдельно сравниваем два PairLogit с одинаковыми парами. Во втором повышаем вес предпочтений удачного клика перед карточкой без реакции на одной из двух ближайших исходных позиций выше. Вес зависит от позиционной кривой и нормируется по запросу. В `comparisons.csv` есть сравнения с Qilin и с соответствующим контрольным вариантом.

```bash
.venv/bin/python run_reward_ablation.py --train "$train" --test-identifiers "$ids" \
  --notes data/qilin/notes/train-*-of-00005.parquet \
  --curves results/avatar/calibration/curves.json --output results/reward_ablation
```

## Метаданные изображений и видео

`run_media_metadata.py` обучает одну и ту же составную цель с текстом и исходной позицией и с дополнительными полями `image_num`, числом путей в `image_path`, длительностью и размерами видео. Дополнительно считаем метрики на срезе запросов про аватарки, обои, фоны, пейзажи, маникюр и образы. Это эксперимент с метаданными. Для оценки по содержимому обложки нужны файлы изображений из отдельного архива, указанного [авторами Qilin](https://huggingface.co/datasets/THUIR/Qilin/discussions/4).

```bash
.venv/bin/python run_media_metadata.py --train "$train" --test-identifiers "$ids" \
  --notes data/qilin/notes/train-*-of-00005.parquet \
  --curves results/avatar/calibration/curves.json --output results/media_metadata
```

## Скор запроса и поста без позиции во входах

`run_position_free.py` учит пять моделей с одинаковыми признаками текста и типа поста. Сравниваем сырые клики, клики с обратным весом позиции, модель клика с вероятностью просмотра и два варианта PairLogit. В парном контроле посты с реакцией ставятся выше постов без реакции. Второй вариант усиливает пары, где пропущенный пост стоял на одну-две исходные позиции выше лайка, сохранения или просмотра от 10 секунд. Общий вес пар одного запроса совпадает в обоих вариантах.

Оцениваем порядок напрямую по скору каждой модели, а также мягкую перестановку поверх исходного Qilin или обычного CatBoost на кликах. Для такой перестановки складываем `−log(ранг в опорной выдаче)` и относительное место поста по скору без позиции, умноженное на коэффициент. Фиксированное смешивание `0.16` позволяет сравнить модели при одном коэффициенте. Другие коэффициенты выбираем на поздней части train по обычному ClickNDCG@10 при LikeNDCG@10 на запросах с лайком не ниже опорной выдачи на той же части. На общем validation сравниваем замороженные варианты.

`calibration.csv` содержит сравнение коэффициентов смешивания внутри train. `comparisons.csv` показывает парные сравнения с Qilin, обычным CatBoost на кликах и соответствующим вариантом без усиления сигнала.
