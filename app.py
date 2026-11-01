import pyautogui
import cv2
import numpy as np
from ultralytics import YOLO
import time
import keyboard
import threading

pyautogui.FAIL_SAFE = True

model = YOLO('runs/detect/cookie_detector-2/weights/best.onnx')

CLASS_AD = 'advertising'
CLASS_GAME = 'game'
CLASS_UP = 'upgrade'

print('Классы модели:', model.names)

stop_flag = threading.Event()

# Кулдаун на покупку апгрейда (секунды между покупками)
UPGRADE_COOLDOWN = 20.0
last_upgrade_time = 0.0


def stop_program():
    print('\n[!] Остановка...')
    stop_flag.set()


keyboard.add_hotkey('space', stop_program)


def click_cookie(cx, cy):
    """Кликает по печенью 30 раз."""
    for _ in range(30):
        if stop_flag.is_set():
            return
        pyautogui.click(cx, cy)
        time.sleep(0.05)
    print(f'>>> Клик по предмету: ({cx}, {cy})')


def find_and_click():
    global last_upgrade_time

    if stop_flag.is_set():
        return False

    screenshot = pyautogui.screenshot()
    screenshot_np = np.array(screenshot)
    screenshot_np = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)

    results = model(screenshot_np, verbose=False)

    # Собираем все найденные боксы
    all_boxes = []
    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf <= 0.4:
                continue
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            x1, y1, x2, y2 = box.xyxy[0]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            all_boxes.append((cls_name, cx, cy))
            print(f'Найден: {cls_name} conf={conf:.2f} at ({cx}, {cy})')

    if not all_boxes:
        print('Ничего не найдено')
        return False

    # 1) Приоритет: закрываем рекламу
    for cls_name, cx, cy in all_boxes:
        if cls_name == CLASS_AD:
            pyautogui.click(cx, cy)
            print(f'>>> Закрыл рекламу: ({cx}, {cy})')
            time.sleep(0.3)  # ждём, пока реклама исчезнет
            return True

    # 2) Покупаем апгрейд, но не чаще чем раз в UPGRADE_COOLDOWN секунд
    now = time.time()
    if now - last_upgrade_time >= UPGRADE_COOLDOWN:
        for cls_name, cx, cy in all_boxes:
            if cls_name == CLASS_UP:
                pyautogui.click(cx, cy)
                last_upgrade_time = now
                print(f'>>> Купил апгрейд: ({cx}, {cy})')
                time.sleep(0.3)
                return True

    # 3) Кликаем по печенью
    for cls_name, cx, cy in all_boxes:
        if cls_name == CLASS_GAME:
            click_cookie(cx, cy)
            return True

    return False


while True:
    try:
        if stop_flag.is_set():
            print('[!] Программа остановлена.')
            break

        if find_and_click():
            time.sleep(0.1)
        else:
            time.sleep(0.5)

    except Exception as e:
        print(f'Ошибка: {e}')
        time.sleep(1)