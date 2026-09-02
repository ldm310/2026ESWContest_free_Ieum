from pathlib import Path

from setuptools import setup

package = "doa_separation_node"

setup(
    name=package,
    version="0.1.0",
    packages=[package, f"{package}.model"],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package}"]),
        (f"share/{package}", ["package.xml"]),
        (f"share/{package}/launch", [str(p) for p in Path("launch").glob("*.py")]),
        (f"share/{package}/weights",
         [str(p) for p in Path("weights").glob("*.pt")]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="JunHyeong Kwon",
    maintainer_email="meoruu00@gmail.com",
    description="ReSpeaker 4채널에서 화자별 분리 음성과 방향을 실시간 발행",
    license="MIT",
    entry_points={
        "console_scripts": [
            f"doa_separation = {package}.node:main",
        ],
    },
)
