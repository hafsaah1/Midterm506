import qrcode

commands = ['LEFT', 'RIGHT', 'FORWARD', 'BACK']

for cmd in commands:
    img = qrcode.make(cmd)
    img.save(f"{cmd}.png")
