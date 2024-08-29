from setuptools import find_packages, setup
from typing import List

hypen_e_dot = "-e ."

# return the list of requirements
def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if hypen_e_dot in requirements:
            requirements.remove(hypen_e_dot)
    
    return requirements

setup(
name="predictive_maintenance_project",
version="0.0.1",
author="Mahdi Kriaa",
author_email="mahdikriaa@gmail.com",
packages=find_packages(),
install_requires=get_requirements("requirements.txt")
)