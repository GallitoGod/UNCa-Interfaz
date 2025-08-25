import tensorflow as tf
from pathlib import Path
import numpy as np

def inspect_tflite_model(model_path: str):
    model_path = str(model_path)
    
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("\n===== INPUT DETAILS =====")
    for idx, d in enumerate(input_details):
        print(f"Input {idx}:")
        print(f"  Name: {d['name']}")
        print(f"  Shape: {d['shape']}")
        print(f"  Dtype: {d['dtype']}")
        print(f"  Quantization: {d['quantization']}")

    print("\n===== OUTPUT DETAILS =====")
    for idx, d in enumerate(output_details):
        print(f"Output {idx}:")
        print(f"  Name: {d['name']}")
        print(f"  Shape: {d['shape']}")
        print(f"  Dtype: {d['dtype']}")
        print(f"  Quantization: {d['quantization']}")

    
    dummy_input = np.zeros(input_details[0]['shape'], dtype=input_details[0]['dtype'])
    interpreter.set_tensor(input_details[0]['index'], dummy_input)
    interpreter.invoke()

    for idx, d in enumerate(output_details):
        out = interpreter.get_tensor(d['index'])
        print(f"\nOutput tensor {idx} sample values:")
        print(out.flatten()[:10])  


if __name__ == "__main__":
    model_file = Path("models/efficientdet-lite0.tflite")
    inspect_tflite_model(model_file)
