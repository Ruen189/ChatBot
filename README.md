Для работы необходима версия python 3.10.18. 

Файлы main.py и courses.json являются основными, а functest.py - для тестирования не LLM функций. 
saved.py для тестирования LLM, взятой из https://huggingface.co/TheBloke/Llama-2-7B-Chat-AWQ

UPD (18.07.2025): перешли на Qwen

Колеса для vllm на Windows не поддерживаются. Рекомендую использовать Ubuntu на WSL, либо просто на Ubuntu. 

Модель TheBloke/Llama-2-7b-Chat-AWQ весит 4gb. Доля используемой памяти определяется gpu_memory_utilization

Возможно потребуется установка pip install accelerate 

Для запуска на fastapi следует установить pip install "fastapi[standard]" потом использовать fastapi dev main.py
