from string import Template

flythrough = Template("""
{
  "4": {
    "inputs": {
      "ckpt_name": "$ckpt_name"
    },
    "class_type": "CheckpointLoaderSimple",
    "_meta": {
      "title": "Load Checkpoint"
    }
  },
  "5": {
    "inputs": {
      "width": [
        "26",
        0
      ],
      "height": [
        "26",
        0
      ],
      "batch_size": 1
    },
    "class_type": "EmptyLatentImage",
    "_meta": {
      "title": "Empty Latent Image"
    }
  },
  "6": {
    "inputs": {
      "text": "$prompt_to",
      "clip": [
        "4",
        1
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Prompt)"
    }
  },
  "7": {
    "inputs": {
      "text": "text, watermark, missing limbs, extra limbs, cartoon, anime, illustration",
      "clip": [
        "4",
        1
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Prompt)"
    }
  },
  "8": {
    "inputs": {
      "samples": [
        "14",
        0
      ],
      "vae": [
        "4",
        2
      ]
    },
    "class_type": "VAEDecode",
    "_meta": {
      "title": "VAE Decode"
    }
  },
  "9": {
    "inputs": {
      "filename_prefix": "$output_folder/f",
      "images": [
        "16",
        0
      ]
    },
    "class_type": "SaveImage",
    "_meta": {
      "title": "Save Image"
    }
  },
  "11": {
    "inputs": {
      "conditioning_to_strength": $progress,
      "conditioning_to": [
        "6",
        0
      ],
      "conditioning_from": [
        "12",
        0
      ]
    },
    "class_type": "ConditioningAverage",
    "_meta": {
      "title": "ConditioningAverage"
    }
  },
  "12": {
    "inputs": {
      "text": "$prompt_from",
      "clip": [
        "4",
        1
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Prompt)"
    }
  },
  "14": {
    "inputs": {
      "seed": 417748201902973,
      "steps": 20,
      "cfg": 7.55,
      "sampler_name": "euler",
      "scheduler": "normal",
      "denoise": 0.9,
      "model": [
        "35",
        0
      ],
      "positive": [
        "29",
        0
      ],
      "negative": [
        "29",
        1
      ],
      "latent_image": [
        "5",
        0
      ]
    },
    "class_type": "KSampler",
    "_meta": {
      "title": "KSampler"
    }
  },
  "15": {
    "inputs": {
      "model_name": "RealESRGAN_x2.pth"
    },
    "class_type": "UpscaleModelLoader",
    "_meta": {
      "title": "Load Upscale Model"
    }
  },
  "16": {
    "inputs": {
      "upscale_model": [
        "15",
        0
      ],
      "image": [
        "8",
        0
      ]
    },
    "class_type": "ImageUpscaleWithModel",
    "_meta": {
      "title": "Upscale Image (using Model)"
    }
  },
  "21": {
    "inputs": {
      "image": "$previous_frame",
      "upload": "image"
    },
    "class_type": "LoadImage",
    "_meta": {
      "title": "Load Image"
    }
  },
  "41": {
    "inputs": {
      "width": [
        "26",
        0
      ],
      "height": [
        "26",
        0
      ],
      "alignment": "center",
      "resampling": "lanczos",
      "supersample": "false",
      "image": [
        "21",
        0
      ]
    },
    "class_type": "ImageResizeAndCropNode",
    "_meta": {
      "title": "Image Resize And Crop Node"
    }
  }, 
  "26": {
    "inputs": {
      "value": $size
    },
    "class_type": "Integer Variable [n-suite]",
    "_meta": {
      "title": "Integer Variable [🅝-🅢🅤🅘🅣🅔]"
    }
  },
  "27": {
    "inputs": {
      "low_threshold": 0.4,
      "high_threshold": 0.8,
      "image": [
        "41",
        0
      ]
    },
    "class_type": "Canny",
    "_meta": {
      "title": "Canny"
    }
  },
  "28": {
    "inputs": {
      "images": [
        "27",
        0
      ]
    },
    "class_type": "PreviewImage",
    "_meta": {
      "title": "Preview Image"
    }
  },
  "29": {
    "inputs": {
      "strength": $control_strength,
      "start_percent": 0,
      "end_percent": 1,
      "positive": [
        "11",
        0
      ],
      "negative": [
        "7",
        0
      ],
      "control_net": [
        "31",
        0
      ],
      "image": [
        "27",
        0
      ]
    },
    "class_type": "ControlNetApplyAdvanced",
    "_meta": {
      "title": "Apply ControlNet (Advanced)"
    }
  },
  "31": {
    "inputs": {
      "control_net_name": "control-lora-depth-rank128.safetensors"
    },
    "class_type": "ControlNetLoader",
    "_meta": {
      "title": "Load ControlNet Model"
    }
  },
  "35": {
    "inputs": {
      "weight_style": $style_weight,
      "weight_composition": $composition_weight,
      "expand_style": false,
      "combine_embeds": "average",
      "start_at": 0,
      "end_at": 1,
      "embeds_scaling": "V only",
      "model": [
        "4",
        0
      ],
      "ipadapter": [
        "37",
        0
      ],
      "image_style": [
        "41",
        0
      ],
      "image_composition": [
        "41",
        0
      ],
      "clip_vision": [
        "38",
        0
      ]
    },
    "class_type": "IPAdapterStyleComposition",
    "_meta": {
      "title": "IPAdapter Style & Composition SDXL"
    }
  },
  "37": {
    "inputs": {
      "ipadapter_file": "ip-adapter-plus_sdxl_vit-h.bin"
    },
    "class_type": "IPAdapterModelLoader",
    "_meta": {
      "title": "IPAdapter Model Loader"
    }
  },
  "38": {
    "inputs": {
      "clip_name": "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
    },
    "class_type": "CLIPVisionLoader",
    "_meta": {
      "title": "Load CLIP Vision"
    }
  }
}
""")
