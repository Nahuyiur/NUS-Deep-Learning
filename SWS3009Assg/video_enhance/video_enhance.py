import cv2, numpy as np, threading, queue, time

# ==== LUT：高亮线性压制 ====
def create_bright_lut(thresh=220, strength=0.45):
    lut = np.arange(256).astype(np.uint8)
    for i in range(256):
        if i > thresh:
            delta = i - thresh
            lut[i] = int(thresh + delta * strength)
    return lut

# ==== 大面积区域专用 LUT 压制 ====
def apply_large_area_lut(v_channel, area_thresh=2000):
    mask = (v_channel > 230).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    v_mod = v_channel.copy()
    
    # 自定义 LUT
    lut = create_bright_lut(thresh=200, strength=0.3)
    lut_applied = cv2.LUT(v_channel, lut)  # 对整图预处理一次 LUT

    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > area_thresh:
            region_mask = (labels == i)
            v_mod[region_mask] = lut_applied[region_mask]
    return v_mod
# ==== 主增强函数 ====
def soft_deglare_retinex(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Step 1: 全图轻压制（LUT）
    lut_base = create_bright_lut(thresh=220, strength=0.45)
    v_base = cv2.LUT(v, lut_base)

    # Step 2: 大块区域进一步压制
    v_final = apply_large_area_lut(v_base)

    # Step 3: 重新组合
    hsv_mod = cv2.merge((h, s, v_final))
    result = cv2.cvtColor(hsv_mod, cv2.COLOR_HSV2BGR)

    # Step 4: 轻锐化（不放大亮度）
    blur = cv2.GaussianBlur(result, (5, 5), 0)
    sharp = cv2.addWeighted(result, 1.15, blur, -0.15, 0)
    return sharp

# ==== 多线程采集 ====
def grab_thread(cap, buf, stop_flag):
    while not stop_flag.is_set():
        ret, f = cap.read()
        if not ret: stop_flag.set(); break
        if not buf.full(): buf.put(f)

# ==== 主循环 ====
def main(cam=0, show_w=640):
    cap = cv2.VideoCapture(cam)
    if not cap.isOpened(): print("❌ 摄像头打开失败"); return

    buf, stop = queue.Queue(3), threading.Event()
    threading.Thread(target=grab_thread, args=(cap,buf,stop), daemon=True).start()
    print("🎥 Retinex + 大块反光 LUT 降亮 (ESC 退出)")

    while not stop.is_set():
        try: frame = buf.get(timeout=1.0)
        except queue.Empty: continue

        t0 = time.time()
        enhanced = soft_deglare_retinex(frame)
        t_ms = (time.time() - t0) * 1000

        h = int(frame.shape[0]*show_w/frame.shape[1])
        vis = np.hstack((
            cv2.resize(frame,   (show_w,h)),
            cv2.resize(enhanced,(show_w,h))
        ))
        cv2.putText(vis, f"{t_ms:.1f} ms", (10,28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.imshow("Original | Smart De-glare", vis)
        if cv2.waitKey(1)&0xFF==27: stop.set()

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
