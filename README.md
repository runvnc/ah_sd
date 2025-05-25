# AH Stable Diffusion Plugin

This plugin provides Stable Diffusion image generation capabilities for the MindRoot agent.

It includes:
- `image` command: Generate an image from a text prompt.
- `text_to_image` service: Backend service for generating images, used by the `image` command.
- `warmup` hook: Initializes the Stable Diffusion pipeline on startup.
