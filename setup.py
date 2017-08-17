from setuptools import setup, find_packages

setup(
    name='comicolorization',
    version='1.0.0',
    packages=find_packages(),
    url='https://github.com/DwangoMediaVillage/Comicolorization',
    author='Chie Furusawa, Kazuyuki Hiroshiba, Keisuke Ogaki, Yuri Odagiri',
    author_email='chie_furusawa@dwango.co.jp, kazuyuki_hiroshiba@dwango.co.jp, keisuke_ogaki@dwango.co.jp, yuri_odagiri@dwango.co.jp',
    description='Code for paper Comicolorization: Semi-automatic Manga Colorization',
    license='MIT License',
    install_requires=[
        'numpy',
        'scikit-image',
        'chainer>=1.14,<2.0.0',
    ]
)
