"""
Google Image Scraper  - query+" cat"
Author : ChatGPT
Date   : 2025-07-03
Note   : 遵守 Google TOS，仅限个人/科研
"""

import os, time, random, urllib.parse, requests
from io import BytesIO
from PIL import Image
from tqdm import tqdm

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import (
    StaleElementReferenceException, ElementClickInterceptedException,
    NoSuchElementException, TimeoutException
)

# ====== 全局参数 ======
IMG_PER_QUERY   = 200      # 目标张数
SCROLL_PAUSE    = 1.2      # 滚动间隔(s)
DOWNLOAD_TIMEOUT= 6        # 单图下载超时
HEADLESS        = False    # 调试时可设 False
MAX_RETRIES     = 3
SAVE_ROOT       = "cats_dataset"  # 根目录

# ————————————— Selenium 初始化 —————————————
def get_driver(headless: bool = HEADLESS) -> webdriver.Chrome:
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    )
    driver = webdriver.Chrome(options=chrome_options)
    driver.set_page_load_timeout(30)
    return driver


# ————————————— 核心爬取流程 —————————————
def fetch_image_urls(driver: webdriver.Chrome, query: str, max_links: int):
    search_url = (
        "https://www.google.com/search?"
        + urllib.parse.urlencode(
            {"q": f"{query} cat", "tbm": "isch", "hl": "en", "safe": "off"}
        )
    )
    driver.get(search_url)
    image_urls, img_count, results_start = set(), 0, 0

    while img_count < max_links:
        # 向下滚动加载更多缩略图
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE)

        # 如果有“显示更多”按钮则点击
        try:
            show_more = driver.find_element(By.CSS_SELECTOR, ".mye4qd")
            show_more.click()
            time.sleep(1)
        except NoSuchElementException:
            pass  # 没找到说明不用点

        # 提取缩略图元素
        thumbnails = driver.find_elements(By.CSS_SELECTOR, "img.Q4LuWd")
        for thumb in thumbnails[results_start:]:
            try:
                thumb.click()
                time.sleep(0.8)
            except (ElementClickInterceptedException, StaleElementReferenceException):
                continue

            # 找到侧边或弹窗里的大图
            images = driver.find_elements(By.CSS_SELECTOR, "img.n3VNCb")
            for img in images:
                src = img.get_attribute("src")
                if src and src.startswith("http"):
                    image_urls.add(src)
            img_count = len(image_urls)
            if len(image_urls) >= max_links:
                break
        results_start = len(thumbnails)

    return image_urls


# ————————————— 下载工具 —————————————
def download_image(url: str, save_path: str) -> bool:
    try:
        resp = requests.get(url, timeout=DOWNLOAD_TIMEOUT)
        img = Image.open(BytesIO(resp.content)).convert("RGB")

        # 过滤竖图 & 小图
        w, h = img.size
        if h / w > 1.3 or min(w, h) < 100:
            return False

        img.save(save_path, "JPEG", quality=92)
        return True
    except Exception:
        return False


# ————————————— 主爬取函数 —————————————
def scrape_google_images(breed: str, num_images: int = IMG_PER_QUERY):
    save_dir = os.path.join(SAVE_ROOT, breed.replace(" ", "_").capitalize())
    os.makedirs(save_dir, exist_ok=True)

    driver = get_driver()
    try:
        urls = fetch_image_urls(driver, breed, num_images * 2)  # 抓多一点做过滤
    finally:
        driver.quit()

    kept = 0
    for i, url in enumerate(tqdm(urls, desc=f"Downloading {breed}")):
        if kept >= num_images:
            break
        fname = f"{breed.replace(' ', '_')}_{kept+1}.jpg"
        if download_image(url, os.path.join(save_dir, fname)):
            kept += 1

    print(f"[DONE] {breed}: 目标 {num_images}，实际保存 {kept} 张 → {save_dir}")


# ————————————— 批量入口 —————————————
if __name__ == "__main__":
    breeds = ["Singapura", "Ragdoll", "Persian", "Sphynx", "Pallas"]
    for b in breeds:
        scrape_google_images(b, num_images=300)   # 每类抓 300 张
