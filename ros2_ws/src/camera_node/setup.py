from pathlib import Path

from setuptools import setup

package = "camera_node"

setup(
    name=package,
    version="0.1.0",
    packages=[package],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package}"]),
        (f"share/{package}", ["package.xml"]),
        (f"share/{package}/launch", [str(p) for p in Path("launch").glob("*.py")]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="JunHyeong Kwon",
    maintainer_email="meoruu00@gmail.com",
    description="음원 방향의 얼굴을 RealSense 프레임에서 찾아 발행하는 노드",
    license="MIT",
    entry_points={"console_scripts": [f"camera = {package}.node:main"]},
)
