from typing import Union

def build_block(data: Union[dict, list], block_name: str) -> str:
    """
    Полностью универсальный формовщик блоков для Real-IT.
    Пользователю не нужно писать кастомные форматтеры.
    
    :param data: список или словарь с данными
    :param block_name: имя блока (например, "курсы", "филиалы", "преподаватели")
    """
    if not data:
        return f"Информация про {block_name} Real-IT отсутствует."

    def auto_formatter(item: dict) -> str:
        return "; ".join(f"{k}: {v}" for k, v in item.items())

    if isinstance(data, dict):
        blocks = []
        for key, items in data.items():
            if not items:
                blocks.append(f"Информация про {block_name} в разделе {key} отсутствует.")
                continue
            block_text = f"Блок про {block_name} ({key}):\n" + "\n".join(auto_formatter(item) for item in items)
            block_text += f"\nКонец блока {block_name} ({key})"
            blocks.append(block_text)
        return "\n".join(blocks)

    elif isinstance(data, list):
        block_text = f"Блок про {block_name} Real-IT:\n" + "\n".join(auto_formatter(item) for item in data)
        block_text += f"\nКонец блока про {block_name} Real-IT"
        return block_text

    return f"Неверный формат данных для {block_name}."