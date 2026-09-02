from pathlib import Path

from setuptools import setup

package = "ui_bridge_node"


def _vendor_files(root: str):
    entries = []
    base = Path(root)
    for path in sorted(base.rglob("*")):
        if path.is_dir() or "__pycache__" in path.parts:
            continue
        entries.append((f"share/{package}/{path.parent.as_posix()}", [path.as_posix()]))
    return entries


setup(
    name=package,
    version="0.1.0",
    packages=[package],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package}"]),
        (f"share/{package}", ["package.xml"]),
        (f"share/{package}/launch", [str(p) for p in Path("launch").glob("*.py")]),
    ] + _vendor_files("jetson_ui"),
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="JunHyeong Kwon",
    maintainer_email="meoruu00@gmail.com",
    description="ROS 토픽을 자막 UI 화면에 연결하는 다리 노드",
    license="MIT",
    entry_points={"console_scripts": [f"ui_bridge = {package}.node:main"]},
)
