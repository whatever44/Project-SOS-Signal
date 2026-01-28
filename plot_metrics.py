import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("metrics.csv")

plt.figure()
plt.plot(df["frame"], df["confidence"])
plt.xlabel("Frame")
plt.ylabel("SOS Confidence")
plt.title("Confidence Over Time")
plt.show()

plt.figure()
plt.plot(df["frame"], df["inference_time_sec"])
plt.xlabel("Frame")
plt.ylabel("Inference Time (sec)")
plt.title("DL Inference Runtime")
plt.show()

print("Average inference time:", df["inference_time_sec"].mean(), "sec")
print("Max inference time:", df["inference_time_sec"].max(), "sec")
