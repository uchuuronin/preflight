"""The torch 2.7.1 probe image's pins (probes.torch_new), kept free of Modal: runner/modal_app.py builds the
image from them, and pipeline.cache_key hashes them into its probes' key, so changing a pin measures afresh
while the default image's cache stays as it is (review, Nov 3)."""

PROBE_NEW_PACKAGES = [
	"torch==2.7.1",
	"torchvision==0.22.1",  # torch 2.7.1's
	"uv==0.5.11",
	"transformers==4.52.3",
	"accelerate==1.7.0",
	"peft==0.15.2",
	"trl==0.18.0",
	"datasets==3.6.0",
	"safetensors==0.5.3",
	"psutil==6.1.0",
	"nvidia-ml-py==12.560.30",
	"pydantic>=2.8,<3",
	"numpy>=1.26,<2.3",
	"httpx>=0.27",
]
