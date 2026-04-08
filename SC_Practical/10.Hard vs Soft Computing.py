def hard_temperature(temp):
    if temp < 25:
        return "Cold"
    else:
        return "Hot"

def soft_temperature(temp):
    if temp <= 20:
        return "Cold"
    elif 20 < temp <= 30:
        return "Warm"
    else:
        return "Hot"

temperature = [18, 25, 28, 35]

print("Temperature | Hard Computing | Soft Computing")
print("---------------------------------------------")

for t in temperature:
    print(f"{t}°C         | {hard_temperature(t):13} | {soft_temperature(t)}")
