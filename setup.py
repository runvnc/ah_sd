from setuptools import setup, find_packages

setup(
    name="ah_sd",
    version="1.0.0",
    author="AH",
    description="AH Stable Diffusion Plugin for MindRoot",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "diffusers",
        "nanoid",
        "transformers",
        "invisible-watermark",
        "accelerate"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    # If MindRoot uses entry points for plugin discovery:
    # entry_points={
    #     'mindroot.plugins': [
    #         'ah_sd = ah_sd',
    #     ]
    # }
)
