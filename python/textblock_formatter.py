def build_courses_block(courses_data: dict) -> str:
    """Формирование текстового блока с курсами."""
    courses = courses_data.get("courses", [])
    if not courses:
        return "Информация про курсы в Real-IT отсутствует."

    return "Блок курсы Real-IT:\n" + "\n".join(
        f"Название:{c.get('title', 'Без названия')}."
        f"Рекомендуемый возраст:(от {c.get('min_age', '?')} до {c.get('max_age', '?')} лет)."
        f"Описание: {c.get('description', '')} Ссылка на курс:({c.get('url', '')})."
        for c in courses
    ) + "\nКонец блока про курсы Real-IT"

def build_locations_block(locations: dict) -> str:
    """Формирование текстового блока с филиалами по всем городам."""
    if not locations:
        return "Информация про филиалы Real-IT отсутствует."

    blocks = []
    for city, city_locations in locations.items():
        if not city_locations:
            blocks.append(f"Информация про филиалы Real-IT в городе {city} отсутствует.")
            continue

        city_block = f"Блок про филиалы в городе {city}:\n" + "\n".join(
            f"Название филиала:{l.get('title', 'Без названия')}, Улица:{l.get('street', '')}."
            f"Как пройти:{l.get('entrance', '')}."
            for l in city_locations
        ) + f"\nКонец блока {city}"
        blocks.append(city_block)

    return "\n".join(blocks)
