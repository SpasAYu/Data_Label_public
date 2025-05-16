# Data Label - Инструмент для разметки изображений

![Project Logo](https://via.placeholder.com/150x50?text=DataLabel)

Инструмент для разметки датасетов изображений с поддержкой ручной и автоматической разметки с использованием YOLOv8.

## 🚀 Возможности

- 📤 Загрузка изображений для разметки
- 🏷️ Определение классов объектов
- ✏️ Ручная разметка bounding boxes
- 🤖 Автоматическая разметка с помощью YOLOv8
- 🔍 Просмотр и корректировка разметки
- 💾 Сохранение в формате YOLO

## 📦 Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/yourusername/Data_Label.git
cd Data_Label
2. Создайте и активируйте виртуальное окружение:

bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.\.venv\Scripts\activate  # Windows
Установите зависимости:

bash
pip install -r requirements.txt
🖥️ Запуск
bash
streamlit run main.py
Приложение будет доступно по адресу: http://localhost:8501

🛠️ Структура проекта
Data_Label/
├── main.py                - Главный файл приложения
├── config.py              - Конфигурационные параметры
├── requirements.txt       - Зависимости
├── data/                  - Данные (создается автоматически)
│   ├── uploads/           - Загруженные изображения
│   ├── annotations/       - Аннотации в формате YOLO
│   └── models/            - Модели YOLO
├── utils/                 - Вспомогательные модули
│   ├── file_utils.py      - Работа с файлами
│   ├── annotation_utils.py- Утилиты аннотаций
│   └── model_utils.py     - Утилиты для работы с моделями
├── components/            - Компоненты интерфейса
│   ├── uploader.py        - Загрузчик изображений
│   ├── annotator.py       - Интерфейс разметки
│   └── autolabel.py       - Автоматическая разметка
└── models/                - Модели
    └── yolo_model.py      - Класс для работы с YOLO
📝 Использование
Загрузка изображений:

Перейдите в раздел "Upload Images"

Выберите файлы или перетащите их в область загрузки

Определение классов:

В разделе "Class Definition" укажите названия классов (по одному на строку)

Ручная разметка:

Во вкладке "Manual Annotation" используйте инструменты для создания bounding boxes

Навигация между изображениями - кнопки "Previous/Next"

Автоматическая разметка:

Во вкладке "Auto-Labeling" выберите модель YOLOv8

Настройте порог уверенности

Запустите разметку для текущего или всех изображений

Корректировка:

После автоматической разметки проверьте и отредактируйте bounding boxes

🤝 Участие в разработке
Форкните репозиторий

Создайте ветку для вашей фичи (git checkout -b feature/AmazingFeature)

Сделайте коммит изменений (git commit -m 'Add some AmazingFeature')

Запушьте в ветку (git push origin feature/AmazingFeature)

Откройте Pull Request

📜 Лицензия
Распространяется под лицензией MIT. См. файл LICENSE.

✉️ Контакты
Ваше имя - your.email@example.com
Проект на GitHub: https://github.com/yourusername/Data_Label