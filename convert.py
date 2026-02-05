import base64

with open("samples/test.wav", "rb") as f:
    encoded = base64.b64encode(f.read()).decode()

print(encoded)
