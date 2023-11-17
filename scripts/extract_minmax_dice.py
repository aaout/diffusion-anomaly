import json

with open("/mnt/ito/diffusion-anomaly/out/dice_and_auroc.json", "r") as file:
    data = json.load(file)

min_dice_value = float("inf")
max_dice_value = float("-inf")
min_dice_element = None
max_dice_element = None

for key, value in data.items():
    dice_value = value["dice"]
    if dice_value < min_dice_value:
        min_dice_value = dice_value
        min_dice_element = {key: value}
    if dice_value > max_dice_value:
        max_dice_value = dice_value
        max_dice_element = {key: value}

print(f"min dice element: {min_dice_element}")
print(f"max dice element: {max_dice_element}")
