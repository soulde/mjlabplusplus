from setuptools import find_packages, setup

setup(
  name="mjlabplusplus",
  version="0.1.0",
  description="Private MJLab extension package for additional robots and tasks.",
  long_description=open("README.md", encoding="utf-8").read(),
  long_description_content_type="text/markdown",
  python_requires=">=3.13",
  package_dir={"": "src"},
  packages=find_packages("src"),
  package_data={
    "mjlabplusplus": [
      "assets/robots/dr02/urdf/*.urdf",
      "assets/robots/dr02/mjcf/*.xml",
      "assets/robots/dr02/assets/*.STL",
    ]
  },
  entry_points={
    "mjlab.tasks": [
      "mjlabplusplus = mjlabplusplus",
    ]
  },
)
