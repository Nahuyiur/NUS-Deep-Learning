from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from PIL import Image
from io import BytesIO
import os, time, urllib.parse, requests

def scrape_yandex_images(query, num_images, parent_folder):
    # ---------- 1. 浏览器启动 ----------
    options = Options()
    options.add_argument("--disable-gpu")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36")
    driver = webdriver.Chrome(options=options)

    # ---------- 2. 打开 Yandex 图片 ----------
    full_query = f"{query} cat"
    query_encoded = urllib.parse.quote_plus(full_query)
    driver.get(f"https://yandex.com/images/search?text={query_encoded}")
    wait = WebDriverWait(driver, 10)
    print(f"[INFO] Searching for '{full_query}' on Yandex...")

    # ---------- 3. 创建保存目录 ----------
    save_dir = os.path.join(parent_folder, query.replace(" ", "_"))
    os.makedirs(save_dir, exist_ok=True)

    saved, seen_urls, tried = 0, set(), 0
    MAX_RETRIES = 1000

    while saved < num_images and tried < MAX_RETRIES:
        thumbnails = driver.find_elements(By.CSS_SELECTOR, "div.serp-item__thumb")
        if tried >= len(thumbnails):
            # 滚动加载更多
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            thumbnails = driver.find_elements(By.CSS_SELECTOR, "div.serp-item__thumb")

        if tried >= len(thumbnails):
            break

        try:
            thumb = thumbnails[tried]
            tried += 1
            thumb.click()
            time.sleep(1)

            # 获取大图元素
            img_elem = None
            for _ in range(5):  # 最多等待 5 次
                try:
                    img_elem = wait.until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "img.MMImage-Origin"))
                    )
                    url = img_elem.get_attribute("src")
                    if url and url.startswith("http"):
                        break
                except:
                    time.sleep(1)

            if not img_elem or not url or url in seen_urls:
                continue

            ext = url.split(".")[-1].split("?")[0].lower()
            if ext not in ['jpg', 'jpeg', 'png']:
                continue

            # 下载并检查尺寸
            try:
                r = requests.get(url, timeout=8)
                if r.status_code == 200 and r.content:
                    img = Image.open(BytesIO(r.content))
                    width, height = img.size
                    if height / width > 1.3:
                        print(f"[SKIP] 竖图跳过：{url} 尺寸={width}x{height}")
                        continue

                    ext = 'jpg' if ext == 'jpeg' else ext
                    fname = f"{saved+1:05d}.{ext}"
                    img.convert("RGB").save(os.path.join(save_dir, fname),
                                            format="JPEG" if ext == "jpg" else "PNG")
                    saved += 1
                    print(f"[{saved}] {url}")
            except Exception as e:
                print(f"[DOWNLOAD FAIL] {e}")
            seen_urls.add(url)

        except Exception as e:
            print(f"[SKIP] 处理失败：{e}")
            continue

    driver.quit()
    print(f"[DONE] Saved {saved} images to {save_dir}")


# ✅ 示例调用
if __name__ == "__main__":
    query = "singapura"
    parent_folder = "/Users/ruiyuhan/Desktop/NUS Deep Learning/SWS3009Assg/datasets"
    scrape_yandex_images(query=query, num_images=1000, parent_folder=parent_folder)
