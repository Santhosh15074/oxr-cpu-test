import cv2, numpy as np
img = np.full((1400, 1000, 3), 255, np.uint8)
f = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, "Quarterly Health Report 2026", (120, 110), f, 1.4, (0,0,0), 3)
y = 200
for p in range(3):
    for l in range(6):
        cv2.putText(img, "Lorem ipsum dolor sit amet, consectetur adipiscing elit sed do eiusmod.", (80, y), f, 0.6, (0,0,0), 1); y += 30
    y += 30
x0, y0 = 120, y + 20
for r in range(6):
    cv2.line(img, (x0, y0 + r*45), (x0+760, y0 + r*45), (0,0,0), 2)
for c in range(5):
    cv2.line(img, (x0 + c*190, y0), (x0 + c*190, y0 + 225), (0,0,0), 2)
for r in range(5):
    for c in range(4):
        cv2.putText(img, f"R{r}C{c} 12.{r}{c}", (x0 + c*190 + 15, y0 + r*45 + 30), f, 0.6, (0,0,0), 1)
cv2.putText(img, "Page 1 of 1", (440, 1360), f, 0.6, (0,0,0), 1)
cv2.imwrite("page.png", img)
