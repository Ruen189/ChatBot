def build_courses_block(courses_data: dict) -> str:
    """Формирование текстового блока с курсами."""
    courses = courses_data.get("courses", [])
    if not courses:
        return "Информация про курсы в Real-IT отсутствует."

    return "Блок курсы Real-IT:\n" + "\n".join(
        f"- {c.get('title', 'Без названия')} "
        f"(от {c.get('min_age', '?')} до {c.get('max_age', '?')} лет): "
        f"{c.get('description', '')} ({c.get('url', '')})"
        for c in courses
    ) + "\nКОНЕЦ БЛОКА"

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
            f"- {l.get('title', 'Без названия')} ({l.get('street', '')}): "
            f"{l.get('entrance', '')}."
            for l in city_locations
        ) + f"\nКОНЕЦ БЛОКА {city}"
        blocks.append(city_block)

    return "\n".join(blocks)
