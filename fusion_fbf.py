from frame_by_frame_constant_depth import run 

output_folder_name = "fusion"

resolution = 768

scenes = [
    ['beautiful islandic ghost, female, woman, tribal face painting', ''],
    ['robot head, digital information processing, leds', ''],
    ['pale face, forest druid, glowing eyes', ''],
    ['rusty robot, gears, steam, clockwork', ''],
    ['insiting cyberpunk girl, neon lights, hologram, futuristic', ''],
    ['desert raven, skull, orange feathers', ''],
    ['terminator, skull, metal, red eyes', ''],
    ['peacock profile face, feathers, colorful', ''],
    ['batteldroid, metal, gears, nuclear power', ''],

    # ab hier chatGPT
    # extend this list: 
    # also use alternatingly: 
    # A) something female, natural, friendly
    # B) something male, technical, powerful
    ['enchanted fairy, wings, glowing dust', ''],
    ['cyborg warrior, exoskeleton, laser guns', ''],
    ['mermaid, ocean waves, coral crown', ''],
    ['mech suit, titanium, plasma cannon', ''],
    ['mystic shaman, feathers, earth magic', ''],
    ['space marine, jetpack, blasters', ''],
    ['goddess of nature, vines, blooming flowers', ''],
    ['android assassin, stealth mode, nanotech', ''],
    ['moonlit witch, spellcasting, crescent moon', ''],
    ['combat drone, aerial surveillance, missiles', ''],
    ['forest nymph, floral dress, serene', ''],
    ['cybernetic samurai, katana, augmented reality', ''],
    ['ethereal elf, silver hair, forest guardian', ''],
    ['futuristic tank, armored, heavy artillery', ''],
    ['magical unicorn, rainbow mane, sparkling horn', ''],
    ['robotic engineer, circuit board, holographic display', ''],
    ['gentle healer, herbal remedies, light aura', ''],
    ['steampunk inventor, goggles, mechanical arm', '']
]

scenes.append(scenes[0]) # make it loop

prompt_texts = []
for i, scene in enumerate(scenes):
    print(i)
    prompt_texts.append([
        scene, 
        i * 5
    ])

print("Scenes:", prompt_texts)


fps = 6

prompt_prefix = ""
prompt_postfix = ", biomechanical, skull, human-machine fusion, solar punk, organic meets artificial, highly detailed"

neg_prompt_prefix = "text, letters, watermark"
neg_prompt_postfix = ", natural, deformed"

first_frame = '/home/felix/work/projects/fusion-visuals/4K/head-ai-video/head-depth-widened-125percent.png'

run(output_folder_name, prompt_texts, fps, 
    prompt_prefix, prompt_postfix, 
    neg_prompt_prefix, neg_prompt_postfix, 
    None, first_frame, resolution)
