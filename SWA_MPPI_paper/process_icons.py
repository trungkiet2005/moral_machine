import os
from PIL import Image

output_dir = r"d:\AI_RESEARCH\moral_machine\Coding\SWA_MPPI_paper\icons"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

images = {
    r"C:\Users\huynh\.gemini\antigravity\brain\da4c7c28-6156-4540-9966-22d0ff168854\moral_scenario_icon_1775409740824.png": "moral_scale.png",
    r"C:\Users\huynh\.gemini\antigravity\brain\da4c7c28-6156-4540-9966-22d0ff168854\llm_network_icon_1775409755721.png": "llm_chip.png",
    r"C:\Users\huynh\.gemini\antigravity\brain\da4c7c28-6156-4540-9966-22d0ff168854\mppi_gears_icon_1775409768938.png": "mppi_gears.png",
    r"C:\Users\huynh\.gemini\antigravity\brain\da4c7c28-6156-4540-9966-22d0ff168854\culture_globe_icon_1775409783199.png": "culture_globe.png",
}

for src, dst in images.items():
    if os.path.exists(src):
        img = Image.open(src).convert('RGBA')
        datas = img.getdata()
        new_data = []
        for item in datas:
            # Change near-white pixels to transparent
            if item[0] > 240 and item[1] > 240 and item[2] > 240:
                new_data.append((255, 255, 255, 0))
            else:
                new_data.append(item)
        img.putdata(new_data)
        out_path = os.path.join(output_dir, dst)
        img.save(out_path, "PNG")
        print(f"Saved {out_path}")
    else:
        print(f"File not found: {src}")
