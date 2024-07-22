import random

from PIL import Image
from diffusers import EulerDiscreteScheduler
import gradio as gr
import numpy as np
import spaces
import torch

# torch._inductor.config.conv_1x1_as_mm = True
# torch._inductor.config.coordinate_descent_tuning = True
# torch._inductor.config.epilogue_fusion = False
# torch._inductor.config.coordinate_descent_check_all_directions = True
from evo_ukiyoe_v1 import load_evo_ukiyoe


DESCRIPTION = """# 🐟 Evo-Ukiyoe
🤗 [モデル一覧](https://huggingface.co/SakanaAI) | 📝 [ブログ](https://sakana.ai/evo-ukiyoe/) | 🐦 [Twitter](https://twitter.com/SakanaAILabs)

[Evo-Ukiyoe](https://huggingface.co/SakanaAI/Evo-Ukiyoe-v1)は[Sakana AI](https://sakana.ai/)が教育目的で開発した浮世絵に特化した画像生成モデルです。
入力した日本語プロンプトに沿った浮世絵風の画像を生成することができます。より詳しくは、上記のブログをご参照ください。
"""
if not torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CPU 🥶 This demo may not work on CPU.</p>"

MAX_SEED = np.iinfo(np.int32).max

device = "cuda" if torch.cuda.is_available() else "cpu"

NUM_IMAGES_PER_PROMPT = 1
SAFETY_CHECKER = True
if SAFETY_CHECKER:
    from safety_checker import StableDiffusionSafetyChecker
    from transformers import CLIPFeatureExtractor

    safety_checker = StableDiffusionSafetyChecker.from_pretrained(
        "CompVis/stable-diffusion-safety-checker"
    ).to(device)
    feature_extractor = CLIPFeatureExtractor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    def check_nsfw_images(
        images: list[Image.Image],
    ) -> tuple[list[Image.Image], list[bool]]:
        safety_checker_input = feature_extractor(images, return_tensors="pt").to(device)
        has_nsfw_concepts = safety_checker(
            images=[images], clip_input=safety_checker_input.pixel_values.to(device)
        )

        return images, has_nsfw_concepts


pipe = load_evo_ukiyoe(device)
pipe.scheduler = EulerDiscreteScheduler.from_config(
    pipe.scheduler.config, use_karras_sigmas=True,
)
pipe.to(device=device, dtype=torch.float16)
# pipe.unet.to(memory_format=torch.channels_last)
# pipe.vae.to(memory_format=torch.channels_last)
# # Compile the UNet and VAE.
# pipe.unet = torch.compile(pipe.unet, mode="max-autotune", fullgraph=True)
# pipe.vae.decode = torch.compile(pipe.vae.decode, mode="max-autotune", fullgraph=True)


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


@spaces.GPU
@torch.inference_mode()
def generate(
    prompt: str,
    negative_prompt: str = "",
    seed: int = 0,
    randomize_seed: bool = False,
    progress=gr.Progress(track_tqdm=True),
):
    seed = int(randomize_seed_fn(seed, randomize_seed))
    generator = torch.Generator().manual_seed(seed)

    images = pipe(
        prompt=prompt + "最高品質の輻の浮世絵。超詳細。",
        negative_prompt=negative_prompt,
        width=1024,
        height=1024,
        guidance_scale=8.0,
        num_inference_steps=40,
        generator=generator,
        num_images_per_prompt=NUM_IMAGES_PER_PROMPT,
        output_type="pil",
    ).images

    if SAFETY_CHECKER:
        images, has_nsfw_concepts = check_nsfw_images(images)
        if any(has_nsfw_concepts):
            gr.Warning("NSFW content detected.")
            return Image.new("RGB", (512, 512), "WHITE"), seed
    return images[0], seed


examples = [
    "植物と花があります。蝶が飛んでいます。",
    "鶴が庭に立っています。雪が降っています。",
    "着物を着ている猫が庭でお茶を飲んでいます。",
]

css = """
.gradio-container{max-width: 690px !important}
h1{text-align:center}
"""
with gr.Blocks(css=css) as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Group():
        with gr.Row():
            prompt = gr.Textbox(placeholder="日本語でプロンプトを入力してください。", show_label=False, scale=8)
            submit = gr.Button(scale=0)
        result = gr.Image(label="Evo-Ukiyoeからの生成結果", type="pil", show_label=False)
    with gr.Accordion("詳細設定", open=False):
        negative_prompt = gr.Textbox(placeholder="日本語でネガティブプロンプトを入力してください。(空白可)", show_label=False)
        seed = gr.Slider(label="シード値", minimum=0, maximum=MAX_SEED, step=1, value=0)
        randomize_seed = gr.Checkbox(label="ランダムにシード値を決定", value=True)
    gr.Examples(examples=examples, inputs=[prompt], outputs=[result, seed], fn=generate)
    gr.on(
        triggers=[
            prompt.submit,
            submit.click,
        ],
        fn=generate,
        inputs=[
            prompt,
            negative_prompt,
            seed,
            randomize_seed,
        ],
        outputs=[result, seed],
        api_name="run",
    )
    gr.Markdown("""⚠️ 本モデルは実験段階のプロトタイプであり、教育および研究開発の目的でのみ提供されています。商用利用や、障害が重大な影響を及ぼす可能性のある環境（ミッションクリティカルな環境）での使用には適していません。
                本モデルの使用は、利用者の自己責任で行われ、その性能や結果については何ら保証されません。
                Sakana AIは、本モデルの使用によって生じた直接的または間接的な損失に対して、結果に関わらず、一切の責任を負いません。
                利用者は、本モデルの使用に伴うリスクを十分に理解し、自身の判断で使用することが必要です。""")

demo.queue().launch()
