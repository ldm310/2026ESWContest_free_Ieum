from pathlib import Path

from setuptools import setup

package = "ui_bridge_node"

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
    description="ROS 토픽을 자막 UI 화면에 연결하는 다리 노드",
    license="MIT",
    entry_points={"console_scripts": [f"ui_bridge = {package}.node:main"]},
)
