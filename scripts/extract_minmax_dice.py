import json

with open("/mnt/ito/diffusion-anomaly/out/segment_diffmap.json", "r") as file:
    data = json.load(file)

sum_dice = 0
for key, value in data.items():
    dice_value = value["dice_mean"]
    sum_dice += dice_value

mean_dice = sum_dice / len(data)
print(f"mean dice: {mean_dice}")
