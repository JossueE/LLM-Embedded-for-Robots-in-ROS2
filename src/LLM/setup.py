from setuptools import find_packages, setup

package_name = 'LLM'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/LLM.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='snorlix',
    maintainer_email='kadavetar10@icloud.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'llm_main = LLM.llm_main:main',
            'audio_listener = LLM.audio_listener:main',
            'wake_word = LLM.wake_word_detector:main',
            'stt = LLM.speech_to_text:main',
        ],
    },
)
