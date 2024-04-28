import traceback

from diffusers import DiffusionPipeline, StableDiffusionPipeline
import torch

from device import device
from output import output


def models():
    if not hasattr(models, 'models'):
        base = DiffusionPipeline.from_pretrained(
            'stabilityai/stable-diffusion-xl-base-1.0',
            torch_dtype=torch.float16,
            variant='fp16',
            use_safetensors=True,
        )
        base.to(device)

        refiner = DiffusionPipeline.from_pretrained(
            'stabilityai/stable-diffusion-xl-refiner-1.0',
            text_encoder_2=base.text_encoder_2,
            vae=base.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant='fp16',
        )
        refiner.to(device)

        models.models = (base, refiner)

    return models.models


def img(prompt, n_steps=60):
    base, refiner = models()

    # run both experts
    image = base(
        prompt=prompt,
        num_inference_steps=n_steps,
        output_type='latent',
    ).images

    image = refiner(
        prompt=prompt,
        num_inference_steps=n_steps,
        image=image,
    ).images[0]

    return image


def img_to(to, *args, **kwargs):
    try:
        image = img(*args, **kwargs)
        image.save(to)
        return True
    except Exception as e:
        traceback.print_exc()
        print(f'Error while creating image to {to}: {e}')
        return False


if __name__ == '__main__':
    prompt_test = 'beautiful woman in dubai'
    img_to(output(f'imgs/{prompt_test} .9 no fraction.jpg'), prompt=prompt_test)
