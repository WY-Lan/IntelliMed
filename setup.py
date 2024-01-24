from setuptools import setup, find_namespace_packages,find_packages


# with open("requirements.txt") as requirements_file:
#     requirements = requirements_file.read().splitlines()
    
setup(
    name='intellimed',  # 打包后的包文件名
    version='0.0.1',    #版本号
    keywords=["pip", "RE","NER","AE","EE"],    # 关键字
    description='IntelliMed Nexus Toolchain is a plug-and-play model and can be used for medical entity recognition, medical relation extraction, and medical event extraction',  # 说明
    license="MIT",  # 许可
    url='',
    author='WY_Lan',
    author_email='892827956@qq.com',
    include_package_data=True,
    platforms="any",
    package_dir={"": "src"},
    packages=find_packages("src"),
    # install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ]
)