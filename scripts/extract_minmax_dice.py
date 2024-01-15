import json

with open("/mnt/ito/diffusion-anomaly/out/dice_and_auroc.json", "r") as file:
    data = json.load(file)

# 'average' キーを除外する
data.pop("average", None)

# 'dice' の値に基づいてソートし、上位5件を取得
top_five = sorted(data.items(), key=lambda x: x[1]["dice"], reverse=True)

# 結果の表示
for key, value in top_five:
    print(f"ID: {key}, Dice: {value['dice']}")

# sum_dice = 0
# for key, value in data.items():
#     dice_value = value["dice_mean"]
#     sum_dice += dice_value

# mean_dice = sum_dice / len(data)
# print(f"mean dice: {mean_dice}")
