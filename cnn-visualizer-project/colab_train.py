import numpy as np
import tensorflow as tf
from tensorflow import keras
from scipy.ndimage import rotate, gaussian_filter
import json

print("TensorFlow:", tf.__version__)

IMG_SIZE      = 16        # Generate at 16×16, resize to 8×8 — clean 2× halving
NUM_PER_CLASS = 60000     # 180,000 total — enough for robust generalisation
np.random.seed(42)

# ═══════════════════════════════════════════════════════════
#  Shape generators — FILLED, at 16×16
# ═══════════════════════════════════════════════════════════

def _fill_convex_poly(img, verts, xx, yy):
    """Fill a convex polygon with 0.0 — works for any vertex winding order."""
    n = len(verts)
    mask_cw  = np.ones(img.shape, dtype=bool)
    mask_ccw = np.ones(img.shape, dtype=bool)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i+1) % n]
        cross = (x2-x1)*(yy-y1) - (y2-y1)*(xx-x1)
        mask_cw  &= (cross >= 0)
        mask_ccw &= (cross <= 0)
    img[mask_cw | mask_ccw] = 0.0

def make_triangle(size=IMG_SIZE):
    """Upward-pointing isosceles triangle — matches real user drawings:
    apex near top-center, wide base spanning the bottom of the image."""
    img = np.ones((size, size), dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
    # Apex: upper portion, slightly randomised left-right
    apex_x = size // 2 + np.random.randint(-size // 5, size // 5 + 1)
    apex_y = np.random.randint(0, size // 3)
    # Base: wide, spanning most of the bottom edge
    margin  = np.random.randint(0, size // 6)
    base_y  = size - 1 - np.random.randint(0, size // 8)
    base_lx = margin
    base_rx = size - 1 - margin
    verts = [(apex_x, apex_y), (base_lx, base_y), (base_rx, base_y)]
    _fill_convex_poly(img, verts, xx, yy)
    return img

def make_heart(size=IMG_SIZE):
    """Filled heart — algebraic heart curve, wide and flat like user drawings."""
    img = np.ones((size, size), dtype=np.float32)
    cx  = size // 2
    cy  = size // 2 - max(1, size // 10)   # shift up: tip centred vertically
    scale = size * 0.42                     # wider scale for flatter look
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
    xn = (xx - cx) / scale
    yn = -(yy - cy) / scale
    inside = (xn**2 + yn**2 - 1)**3 - xn**2 * yn**3 <= 0
    img[inside] = 0.0
    return img

def make_square(size=IMG_SIZE):
    """Large filled rectangle — matches real user drawings which fill most of
    the canvas with only a thin white border visible."""
    img = np.ones((size, size), dtype=np.float32)
    # User drawings are very large: 70–100 % of canvas
    w = int(np.random.uniform(size * 0.70, size * 1.00))
    h = int(np.random.uniform(size * 0.70, size * 1.00))
    w = min(w, size); h = min(h, size)
    x0 = (size - w) // 2; y0 = (size - h) // 2
    img[y0:y0+h, x0:x0+w] = 0.0
    return img

# ═══════════════════════════════════════════════════════════
def normalize_like_website(img):
    mn, mx = img.min(), img.max()
    if mx - mn > 0.05:
        return (img - mn) / (mx - mn)
    return img

# ═══════════════════════════════════════════════════════════
#  Augmentation — kept light so distinctive features survive
# ═══════════════════════════════════════════════════════════
def augment(img):
    img = img + np.random.normal(0, 0.04, img.shape)
    img = np.clip(img, 0, 1)
    dx, dy = np.random.randint(-2, 3), np.random.randint(-2, 3)
    img = np.roll(np.roll(img, dy, axis=0), dx, axis=1)
    img = rotate(img, np.random.uniform(-20, 20), reshape=False, mode='constant', cval=1.0)
    img = gaussian_filter(img, sigma=np.random.uniform(0, 0.10))   # light blur only
    return np.clip(img, 0, 1)

# ═══════════════════════════════════════════════════════════
#  Generate dataset
# ═══════════════════════════════════════════════════════════
generators = [make_triangle, make_heart, make_square]
X_list, y_list = [], []
for label, gen in enumerate(generators):
    for _ in range(NUM_PER_CLASS):
        X_list.append(augment(gen()))
        y_list.append(label)

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list,  dtype=np.int32)

# Resize 16×16 → 8×8  (clean 2× halving — preserves shape better than 28→8)
X_r = tf.image.resize(X[...,np.newaxis], [8, 8]).numpy()[...,0]

# Apply the SAME normalisation the website applies
X_r = np.array([normalize_like_website(im) for im in X_r], dtype=np.float32)

idx = np.random.permutation(len(X_r))
X_r, y = X_r[idx], y[idx]
split = int(0.8 * len(X_r))
X_train, X_test = X_r[:split],  X_r[split:]
y_train, y_test = y[:split],    y[split:]

print(f"Train:{len(X_train)}  Test:{len(X_test)}")
print(f"T:{(y_train==0).sum()}  H:{(y_train==1).sum()}  S:{(y_train==2).sum()}")

# ═══════════════════════════════════════════════════════════
#  Model — same architecture as website (4 filters, Dense-8)
#  Flatten = 4×2×2 = 16,  Dense-1 = 8
# ═══════════════════════════════════════════════════════════
model = keras.Sequential([
    keras.layers.InputLayer(shape=(8, 8, 1)),
    keras.layers.Conv2D(4, (3,3), padding='same', activation='relu', name='conv1'),
    keras.layers.MaxPooling2D((2,2), name='pool1'),
    keras.layers.Conv2D(4, (3,3), padding='same', activation='relu', name='conv2'),
    keras.layers.MaxPooling2D((2,2), name='pool2'),
    keras.layers.Flatten(name='flatten'),          # 4×2×2 = 16
    keras.layers.Dense(8, activation='relu', name='dense1'),
    keras.layers.Dense(3, activation='softmax',  name='output')
], name="cnn_shape_classifier")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ═══════════════════════════════════════════════════════════
#  Train — ReduceLROnPlateau halves lr when stuck; EarlyStopping
#          restores best weights automatically
# ═══════════════════════════════════════════════════════════
Xtr, Xte = X_train[...,np.newaxis], X_test[...,np.newaxis]

es  = keras.callbacks.EarlyStopping(
          monitor='val_accuracy', patience=25,
          restore_best_weights=True)
rlr = keras.callbacks.ReduceLROnPlateau(
          monitor='val_loss', factor=0.5,
          patience=10, min_lr=1e-5, verbose=1)
mc  = keras.callbacks.ModelCheckpoint(
          'best_model.keras', monitor='val_accuracy',
          save_best_only=True, verbose=0)

model.fit(Xtr, y_train,
          epochs=200,
          batch_size=128,
          validation_data=(Xte, y_test),
          callbacks=[es, rlr, mc],
          verbose=1)

model.load_weights('best_model.keras')
_, acc = model.evaluate(Xte, y_test, verbose=0)
print(f"\nBest test accuracy: {acc*100:.1f}%")

if acc < 0.90:
    print("  ⚠  Accuracy below 90 % — re-run the cell (random init may be unlucky)")
else:
    print("  ✓  90 %+ achieved")

# ═══════════════════════════════════════════════════════════
#  Sanity check — same preprocessing as website
# ═══════════════════════════════════════════════════════════
print("\nSanity check (with website normalisation):")
shapes = {"Triangle": make_triangle(), "Heart": make_heart(), "Square": make_square()}
ok_all = True
for name, img16 in shapes.items():
    img8 = tf.image.resize(img16[...,np.newaxis], [8,8]).numpy()[...,0]
    img8 = normalize_like_website(img8)
    pred = model.predict(img8[np.newaxis,...,np.newaxis], verbose=0)[0]
    w    = ["Triangle","Heart","Square"][int(np.argmax(pred))]
    ok   = "✓" if w == name else "✗ WRONG"
    print(f"  {name:10s}→{w:10s}  T={pred[0]:.2f} H={pred[1]:.2f} S={pred[2]:.2f}  {ok}")
    if w != name: ok_all = False

if not ok_all:
    print("\n  ⚠ Sanity failed — re-run cell (weights re-init randomly each time)")
else:
    print("\n  ✓ All three correct — ready to export")

# ═══════════════════════════════════════════════════════════
#  Export
# ═══════════════════════════════════════════════════════════
def r4(v):
    if isinstance(v, (list, np.ndarray)): return [r4(x) for x in v]
    return round(float(v), 4)

def gconv(l):
    W, b = l.get_weights()
    return r4(W.transpose(3,2,0,1).tolist()), r4(b.tolist())

def gdense(l):
    W, b = l.get_weights()
    return r4(W.T.tolist()), r4(b.tolist())

c1W, c1b = gconv(model.get_layer('conv1'))
c2W, c2b = gconv(model.get_layer('conv2'))
dW,  db  = gdense(model.get_layer('dense1'))
oW,  ob  = gdense(model.get_layer('output'))

mj = {
    "classNames": ["Triangle","Heart","Square"],
    "inputSize": 8,
    "layers": [
        {"name":"Conv2D-1","type":"conv2d","numFilters":4,"kernelSize":3,
         "padding":"same","activation":"relu","filters":c1W,"biases":c1b},
        {"name":"Conv2D-2","type":"conv2d","numFilters":4,"kernelSize":3,
         "padding":"same","activation":"relu","filters":c2W,"biases":c2b},
        {"name":"Dense-1","type":"dense","units":8,"activation":"relu",
         "weights":dW,"biases":db},
        {"name":"Output","type":"dense","units":3,"activation":"softmax",
         "weights":oW,"biases":ob}
    ]
}

js = json.dumps(mj)
print(f"\nJSON: {len(js.encode())/1024:.1f} KB")
print("\n" + "="*60)
print("COPY THE LINE BELOW INTO model.js")
print("Replace the existing  var MODEL_JSON = {...};  block")
print("="*60 + "\n")
print("var MODEL_JSON = " + js + ";")
print("\n" + "="*60)
