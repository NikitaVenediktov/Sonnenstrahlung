<p align="center">
  <img src="https://sites.google.com/site/sunactiv/24-cikl/fc37c43-13-sunspot-predict-l.jpg?attredirects=0" align="middle"  width="600" />
</p>


<h1 align="center">
 Sonnenstrahlung
</h1S>

<h4 align="center">

![1](https://img.shields.io/badge/python-3.10.6+-aff.svg)
![2](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg)
![3](https://img.shields.io/github/stars/NikitaVenediktov/Sonnenstrahlung?color=ccf)
![4](https://img.shields.io/github/v/release/NikitaVenediktov/Sonnenstrahlung?color=ffa)

</h4>

-----------------------------------------------

<h4 align="center">
  <a href=#features> Features </a> |
  <a href=#installation> Installation </a> |
  <a href=#quick-start> Quick Start </a> |
  <a href=#community> Community </a>
</h4>

-----------------------------------------------

Солнечная энергия, в настоящее время является одним из наиболее популярных альтернативных источников энергии. Некоторые районы, поселения,частные территории бывают полностью зависимы от этого вида энергии и логично,что нужно рационально накапливать и использовать ее. В этом может помочь предсказание интенсивности солнечного излучения (солнечной радиации). Знание того,когда условия наиболее благоприятны для падающего солнечного излучения, имеет решающее значение для принятия решения о том, когда и где разместить солнечные панели и батареи и после наиболее эффективно использовать полученную энергию.


## &#128204;Дз3 по инженерным практикам в ML

- [x] Отформатировать код с помощью isort и black/autopep8/yapf.
- [x] Выбрать набор плагинов для flake8 (от 5 штук).
```py
-   repo: https://github.com/PyCQA/flake8
    rev: 6.0.0
    hooks:
    -   id: flake8
        args: [--max-line-length=131]
        additional_dependencies: [
            'flake8-bugbear>=19.8.0',
            'flake8-isort>=2.7.0',
            'flake8-bandit>=3.0.0',
            'flake8-builtins>=1.5.3',
            'flake8-annotations-complexity>=0.0.7',
            'flake8-requirements>=1.5.3',
        ]
```

- [x] Записать выбранные формтеры, линтеры и плагины в readme.md. (1 балла) \
  Пишу их сюда) trailing-whitespace, end-of-file-fixer, check-yaml, check-added-large-files, black и flake8 с допами
- [x] Зафиксировать настройки форматера и линтера в pyproject.toml или setup.cfg (1 балла)
- [x] Настроить и добавить pre-commit в проект. (1 балла)
- [x] Провести анализ кода с помощью flake8 и плагинов и зафиксировать проблемы в файле linting.md (1 балла)
- [] Провести рефакторинг выявленных проблем. (3 балла)

Стандартная установка и запуск pre-commit:

```py
poetry add pre-commit
pre-commit sample-config > .pre-commit-config.yaml
pre-commit install
pre-commit run --all-files
```

если ошибки

```py
pre-commit clean
pre-commit autoupdate
```

Пример работы

```sh
(.venv) (base) nikivene@DESKTOP-78NOBF0:~/ITMO_Projects/Sonnenstrahlung$ pre-commit run --all-files
trim trailing whitespace.................................................Passed
fix end of files.........................................................Passed
check yaml...........................................(no files to check)Skipped
check for added large files..............................................Passed
file contents sorter.................................(no files to check)Skipped
black....................................................................Passed
flake8...................................................................Passed
```

```
poetry export -f requirements.txt --output requirements.txt
isort train.py
```

## &#128204;Features

### &#128642;Под капотом  (Мы предоставляем сценарий):

1. Обработка данных
2. Отсев признаков
3. Обучение модели
4. Прогноз

#### ⚡ Пример работы

##### Визуализация данных
<p align="center">
  <img src="https://github.com/NikitaVenediktov/Sonnenstrahlung/blob/main/experiments/output1.png" align="middle"  width="600" />
</p>

##### Корреляция признаков
<p align="center">
  <img src="https://github.com/NikitaVenediktov/Sonnenstrahlung/blob/main/experiments/output2.png" align="middle"  width="600" />
</p>

##### Графики с прогнозом
<p align="center">
  <img src="https://github.com/NikitaVenediktov/Sonnenstrahlung/blob/main/experiments/output3.png" align="middle"  width="600" />
</p>




#### 🚀 Метрики

Коэффициент детерминации($R^2$) = 0.94
#### Модели

Случайный лес



## &#128204;Installation

### Используемые библиотеки

* python >= 3.10
* numpy >= 1.23.4
* pandas >= 1.5.0
* scikit-learn >= 1.1.2
* notebook >= 6.5.1
* seaborn>=0.12.1
* pytz>=2022.6

Через requirements.txt для pip:
```python
pip install -r requirements.txt
```
С помощью [Poetry](https://python-poetry.org) устанавливаются все зависимости. Кроме pip можно использовать [Homebrew](https://formulae.brew.sh/formula/poetry) или [Conda](https://anaconda.org/conda-forge/poetry).
``` python
git clone https://github.com/NikitaVenediktov/Sonnenstrahlung.git
pip install poetry
poetry install
```


## &#128204; Quick Start

В разработке

## &#128204;Community

### Расти вместе с AI Talent Hub!
На базе [AI Talent Hub](https://ai.itmo.ru/) Университет ИТМО совместно с компанией Napoleon IT запустил образовательную программу «Инженерия машинного обучения». Это не краткосрочные курсы без практического применения, а онлайн-магистратура нового формата, основанная на реальном рабочем процессе в компаниях.

Этот проект создан в рамках второго задания по курсу: "Глубокое обучение на практике"

Мы команда ViN:
* [Виктор](https://t.me/anoninf)
* [Илья](https://t.me/sadinhead)
* [Никита](https://t.me/space_apple)

<details><summary> &#128516; Шутейка </summary>
<p>

![Jokes Card](https://readme-jokes.vercel.app/api)

</p>
</details>

## Цитирование

Если вы используете GennaDIY в своих исследованиях, рассмотрите возможность цитирования
```python
@misc{=Sonnenstrahlung,
    title={=Sonnenstrahlung: An Easy-to-use and High Performance CLI},
    author={ViN Contributors},
    howpublished = {\url{https://github.com/NikitaVenediktov/Sonnenstrahlung}},
    year={2022}
}
```

## Благодарность

- [Даннные](https://www.kaggle.com/datasets/dronio/SolarEnergy)
## Лицензия

 [The MIT License](https://opensource.org/licenses/mit-license.php).
