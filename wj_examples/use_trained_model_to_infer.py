from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO(r"..\runs\detect\train2\weights\best.pt")

results = model(r"wj_images\Untitled3.png")  # Predict on an image
results[0].show()  # Display results

# Export the model to ONNX format for deployment
# path = model.export(format="onnx")  # Returns the path to the exported model